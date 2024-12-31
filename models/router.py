import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import models
from .modules import CrossAttention
from torch import softmax


class Expert(nn.Module):
    def __init__(self, emb_size):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
        )

    def forward(self, x):
        return self.seq(x)


# Mixture of Experts
class MoE(nn.Module):
    def __init__(self, config, experts, top, emb_size, w_importance=0.01):
        super().__init__()
        self.attConfig = config["attnConfig"]
        self.experts = nn.ModuleList([Expert(emb_size) for _ in range(experts)])
        self.top = top
        self.crossAtt = CrossAttention(
            self.attConfig["embed_size"],
            self.attConfig["heads"],
            self.attConfig["attn_dropout"],
        )
        self.gate = nn.Linear(emb_size, experts)
        self.noise = nn.Linear(emb_size, experts)  # add noise to gate
        self.w_importance = w_importance  # expert balance(for loss)

    def forward(self, x, q):  # x: (batch,seq_len,emb)
        x_shape = x.shape

        x = x.reshape(-1, x_shape[-1])  # (batch*seq_len,emb)

        # gates
        att = self.crossAtt(x.unsqueeze(1), q.unsqueeze(1)).squeeze(1)
        gate_logits = self.gate(att)  # (batch*seq_len,experts)
        gate_prob = softmax(gate_logits, dim=-1)  # (batch*seq_len,experts)

        # 2024-05-05 Noisy Top-K Gating
        if self.training:
            noise = torch.randn_like(gate_prob) * nn.functional.softplus(
                self.noise(x))  # https://arxiv.org/pdf/1701.06538 , StandardNormal()*Softplus((x*W_noise))
            gate_prob = gate_prob + noise

        # top expert
        top_weights, top_index = torch.topk(gate_prob, k=self.top,
                                            dim=-1)  # top_weights: (batch*seq_len,top), top_index: (batch*seq_len,top)
        top_weights = softmax(top_weights, dim=-1)

        top_weights = top_weights.view(-1)  # (batch*seq_len*top)
        top_index = top_index.view(-1)  # (batch*seq_len*top)

        x = x.unsqueeze(1).expand(x.size(0), self.top, x.size(-1)).reshape(-1, x.size(-1))  # (batch*seq_len*top,emb)
        y = torch.zeros_like(x)  # (batch*seq_len*top,emb)

        # run by per expert
        for expert_i, expert_model in enumerate(self.experts):
            x_expert = x[top_index == expert_i]  # (...,emb)
            y_expert = expert_model(x_expert)  # (...,emb)

            add_index = (top_index == expert_i).nonzero().flatten()
            y = y.index_add(dim=0, index=add_index,
                            source=y_expert)  # y[top_index==expert_i]=y_expert

        # weighted sum experts
        top_weights = top_weights.view(-1, 1).expand(-1, x.size(-1))  # (batch*seq_len*top,emb)
        y = y * top_weights
        y = y.view(-1, self.top, x.size(-1))  # (batch*seq_len,top,emb)
        y = y.sum(dim=1)  # (batch*seq_len,emb)

        # experts balance loss
        # https://arxiv.org/pdf/1701.06538 BALANCING EXPERT UTILIZATION
        if self.training:
            importance = gate_prob.sum(dim=0)  # sum( (batch*seq_len,experts) , dim=0)
            # Coefficient of Variation(CV), CV = standard deviation / mean
            importance_loss = self.w_importance * (torch.std(importance) / torch.mean(importance)) ** 2
        else:
            importance = gate_prob.sum(dim=0)
            importance_loss = self.w_importance * (torch.std(importance) / torch.mean(importance)) ** 2
            # importance_loss = None
        return y.view(x_shape), gate_prob, importance_loss


class RouterGate(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(RouterGate, self).__init__()
        self.config = config
        self.embed_size = self.config["FUSION_IN"]
        self.attConfig = self.config["attnConfig"]
        self.output = int(self.attConfig["embed_size"] / 4)
        experts = self.config["EXPERTS"]
        top = self.config["TOP"]
        self.moe = MoE(config, experts, top, self.attConfig["embed_size"])
        self.cnnEncoder = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.cnnEncoder.fc.in_features
        self.cnnEncoder.fc = torch.nn.Linear(num_ftrs, self.output)
        self.cnnEncoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.linerImg = nn.Linear(self.attConfig["embed_size"], self.output)
        self.out = nn.Linear(int(self.embed_size * 2), self.embed_size)

    def forward(self, source, target, background, image, text):
        s = self.cnnEncoder(source)
        t = self.cnnEncoder(target)
        b = self.cnnEncoder(background)
        img = self.linerImg(image)
        img = nn.functional.relu(img)
        visionFeatures = torch.cat((s, t, b, img), dim=1)

        moeFeatures, gate_prob, importance_loss = self.moe(visionFeatures, text)

        return moeFeatures, gate_prob, importance_loss
