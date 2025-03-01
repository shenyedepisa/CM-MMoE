import torch
import torch.nn as nn
from transformers import CLIPModel, AutoModel
from models.imageModels import UNet


class CDModel(nn.Module):
    def __init__(
            self,
            config,
            question_vocab,
            input_size,
            textHead,
            imageHead,
            trainText,
            trainImg,
    ):
        super(CDModel, self).__init__()
        self.config = config
        self.num_epochs = config["num_epochs"]
        self.question_vocab = question_vocab
        self.maskHead = config["maskHead"]
        self.textHead = textHead
        self.imageHead = imageHead
        self.imageModelPath = config["imageModelPath"]
        self.textModelPath = config["textModelPath"]
        self.fusion_in = config["FUSION_IN"]
        self.MoE = config['MoE']
        self.fusion_hidden = config["FUSION_HIDDEN"]
        self.num_classes = config["answer_number"]
        self.clipList = config["clipList"]
        self.vitList = config["vitList"]
        self.oneStep = config["one_step"]
        self.layer_outputs = {}
        self.attConfig = self.config["attnConfig"]

        saveDir = config["saveDir"]
        from models import RouterGate
        if self.MoE:
            self.router = RouterGate(config)
        if self.maskHead:
            if not self.oneStep:
                self.maskNet = torch.load(f"{saveDir}maskModel.pth")
                for param in self.maskNet.parameters():
                    param.requires_grad = False
            else:
                self.maskNet = UNet(n_channels=3, n_classes=3, bilinear=False)
                state_dict = torch.load(config["maskModelPath"])
                del state_dict["outc.conv.weight"]
                del state_dict["outc.conv.bias"]
                self.maskNet.load_state_dict(state_dict, strict=False)
        if self.imageHead == "siglip_512":
            siglip_model = AutoModel.from_pretrained(self.imageModelPath)
            self.imgModel = siglip_model.vision_model
            self.lineV = nn.Linear(768, 768)
        elif self.imageHead in self.clipList:
            clip = CLIPModel.from_pretrained(self.imageModelPath)
            self.imgModel = clip.vision_model
            self.lineV = nn.Linear(768, 768)

        if self.textHead == "siglip_512":
            siglip_model = AutoModel.from_pretrained(self.textModelPath)
            self.textModel = siglip_model.text_model
        elif self.textHead in self.clipList:
            clip = CLIPModel.from_pretrained(self.textModelPath)
            self.textModel = clip.text_model
            self.lineQ = nn.Linear(512, 768)
        self.linear_classify1 = nn.Linear(self.fusion_in, self.fusion_hidden)
        self.linear_classify2 = nn.Linear(self.fusion_hidden, self.num_classes)
        self.dropout = torch.nn.Dropout(config["DROPOUT"])
        if not trainText:
            for param in self.textModel.parameters():
                param.requires_grad = False
        if not trainImg:
            for param in self.imglModel.parameters():
                param.requires_grad = False

    def forward(self, input_v, input_q, mask=None, epoch=0):
        predict_mask = self.maskNet(input_v)
        m0 = predict_mask[:, 0, :, :].unsqueeze(1) / 255  # source
        m1 = predict_mask[:, 1, :, :].unsqueeze(1) / 255  # target
        m2 = predict_mask[:, 2, :, :].unsqueeze(1) / 255  # background
        v = self.imgModel(pixel_values=input_v)["pooler_output"]
        v = self.dropout(v)
        v = self.lineV(v)
        v = nn.SiLU()(v)
        if self.textHead == "siglip_512":
            q = self.textModel(input_ids=input_q["input_ids"])["pooler_output"]
        elif self.textHead in self.clipList:
            q = self.textModel(**input_q)["pooler_output"]
            q = self.dropout(q)
            q = self.lineQ(q)
            q = nn.SiLU()(q)
        else:
            q = self.textModel(input_q)
        moe, gate_prob, importance_loss = self.router(m0, m1, m2, v, q)
        x = torch.mul(moe, q)
        x = nn.SiLU()(x)
        x = self.dropout(x)
        x = self.linear_classify1(x)
        x = nn.SiLU()(x)
        x = self.dropout(x)
        x = self.linear_classify2(x)

        return x, predict_mask, gate_prob, importance_loss
