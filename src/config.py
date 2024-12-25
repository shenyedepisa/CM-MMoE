from sacred import Experiment
import os

ex = Experiment("CM-MMoE", save_git_info=False)


@ex.config
def config():
    # save
    use_wandb = False
    subDir = "MoE_6_4"

    # Wandb Config
    # https://docs.wandb.ai/quickstart/
    wandbName = subDir
    wandbKey = ""
    project = "CM-MMoE"
    job_type = "train"

    MoE = True
    answer_number = 51
    question_classes = 14
    # if MoE:
    EXPERTS = 6
    TOP = 4

    sga = False
    BLIP = False
    Co_att = False
    sga1 = False
    mac = False
    mcan = False
    dvqa = False
    earthVQA = False
    san = False
    mqvqa = False
    rsivqa = False
    eth = False

    normalize = False
    opts = True
    one_step = True
    num_epochs = 30
    thread_epoch = 20

    learning_rate = 5e-5  # 5e-5

    saveDir = "./outputs/"
    saveDir = os.path.join(saveDir, subDir + '/')
    # new_data_path = "datasets/CM_dataset/"
    new_data_path = './datasets/CM_dataset/'
    source_image_size = 512

    image_resize = 224

    FUSION_IN = 768
    FUSION_HIDDEN = 512
    DROPOUT = 0.3

    add_mask = False
    mask_only = False
    learnable_mask = False
    img_only = False
    if img_only or eth or rsivqa or mqvqa or san or dvqa or mac or mcan or img_only:
        opts = False
        one_step = True
        add_mask = False
        mask_only = False
        learnable_mask = False

    resample = False
    pin_memory = True
    persistent_workers = True
    num_workers = 4

    real_batch_size = 32
    batch_size = 32  # batch_size * steps == real_batch_size
    steps = int(real_batch_size / batch_size)
    weight_decay = 0
    opt = "Adam"
    scheduler = True
    CosineAnnealingLR = True
    warmUp = False
    L1Reg = False
    trainText = True
    trainImg = True
    finetuneMask = True

    if scheduler:
        end_learning_rate = 1e-6

    json_path = os.path.join(new_data_path, 'JsonFiles')
    DataConfig = {
        "images_path": os.path.join(new_data_path, "image"),
        "sourceMask_path": os.path.join(new_data_path, "source"),
        "targetMask_path": os.path.join(new_data_path, "target"),
        "backgroundMask_path": os.path.join(new_data_path, "background"),
        "seg_path": os.path.join(new_data_path, "segmentation"),
        "answersJson": os.path.join(json_path, "Answers.json"),
        "allQuestionsJSON": os.path.join(json_path, "All_Questions.json"),
        "train": {
            "imagesJSON": os.path.join(json_path, "All_Images.json"),
            "questionsJSON": os.path.join(json_path, "Train_Questions.json"),
        },
        "val": {
            "imagesJSON": os.path.join(json_path, "All_Images.json"),
            "questionsJSON": os.path.join(json_path, "Val_Questions.json"),
        },
        "test": {
            "imagesJSON": os.path.join(json_path, "All_Images.json"),
            "questionsJSON": os.path.join(json_path, "Test_Questions.json"),
        },
    }
    MAX_ANSWERS = 100
    LEN_QUESTION = 30
    clipList = [
        "clip",
        "rsicd",
        "clip_b_32_224",
        "clip_b_16_224",
        "clip_l_14_224",
        "clip_l_14_336",
    ]
    vitList = ["vit-b", "vit-s", "vit-t"]
    maskHead = "unet"
    if maskHead == "unet":
        maskModelPath = (
            "models/imageModels/milesial_UNet/unet_carvana_scale1.0_epoch2.pth"
        )

    imageHead = "clip_b_32_224"
    if imageHead == "clip_b_32_224":
        imageModelPath = "models/clipModels/openai_clip_b_32"
        VISUAL_OUT = 768
        imageSize = 224
    elif imageHead == "siglip_512":
        imageModelPath = "models/clipModels/siglip_512"
        imageSize = 512
    else:
        imageSize = 256
    textHead = "clip_b_32_224"
    if textHead == "clip_b_32_224":
        textModelPath = "models/clipModels/openai_clip_b_32"
        QUESTION_OUT = 512
    elif textHead == "siglip_512":
        textModelPath = "models/clipModels/siglip_512"
        imageSize = 512
    elif textHead == "skipthoughts":
        textModelPath = "models/textModels/skip-thoughts"
        QUESTION_OUT = 2400
    attnConfig = {
        "embed_size": FUSION_IN,
        "heads": 6,
        "mlp_input": 768,
        "mlp_ratio": 4,
        "mlp_output": 768,
        "attn_dropout": 0.1,
    }

    MAC_cfg = {
        "TFLAG": True,
        "LEARNING_RATE": 0.0001,
        "BATCH_SIZE": 64,
        "MAX_EPOCHS": 25,
        "SNAPSHOT_INTERVAL": 5,
        "WEIGHT_INIT": "xavier_uniform",
        "CLIP_GRADS": True,
        "CLIP": 8,
        "MAX_STEPS": 4,
        "EALRY_STOPPING": True,
        "PATIENCE": 5,
        "VAR_DROPOUT": False,
        "DATA_DIR": '',
        "NUM_ANS": answer_number,
    }

