import subprocess

import torch

subprocess.call(
    "pip install git+https://github.com/facebookresearch/fvcore.git", shell=True
)
net = torch.hub.load("zhanghang1989/ResNeSt", "resnest50", pretrained=True)
