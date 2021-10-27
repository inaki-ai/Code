from models.segmentors.unet import UNet, UNet2
from models.segmentors.rdau_net import RDAU_NET
from models.segmentors.unetpp import UNetpp
from self_attention_cv.transunet import TransUnet
from torchsummary import summary

import torch

model = UNet(3, 1, True)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"UNET: {total_params}")

model = RDAU_NET()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"RDA-UNET: {total_params}")

model = UNetpp(1)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"UNET++: {total_params}")

model = TransUnet(in_channels=3, img_dim=128, vit_blocks=8,
                                    vit_dim_linear_mhsa_block=512,
                                    classes=1)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"TRANSUNET: {total_params}")
