from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GroupNorm

from .encoder import build_encoder


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        num_groups = 32
        if out_channels % num_groups != 0 or out_channels == num_groups:
            num_groups = out_channels // 2

        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3),
                      stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.blocks(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample=True, **_):
        super().__init__()
        self.up_sample = up_sample

        self.block = nn.Sequential(
                Conv3x3GNReLU(in_channels, out_channels),
                Conv3x3GNReLU(out_channels, out_channels),
        )

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x, skip = x
        else:
            skip = None

        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        return self.block(x)


class Decoder(nn.Module):
    def __init__(self,
                 num_classes,
                 encoder_channels,
                 dropout=0.2,
                 out_channels=[256, 128, 64, 32, 16],
                 **_):
        super().__init__()
        in_channels = encoder_channels
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.center = DecoderBlock(in_channels=in_channels[0], out_channels=in_channels[0],
                                   up_sample=False)

        self.layer1 = DecoderBlock(in_channels[0]+in_channels[1], out_channels[0])
        self.layer2 = DecoderBlock(in_channels[2]+out_channels[0], out_channels[1])
        self.layer3 = DecoderBlock(in_channels[3]+out_channels[1], out_channels[2])
        self.layer4 = DecoderBlock(in_channels[4]+out_channels[2], out_channels[3])
        self.layer5 = DecoderBlock(out_channels[3], out_channels[4])

        self.final = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_channels[-1], num_classes, kernel_size=1)
        )

    def forward(self, x, encodes):
        skips = encodes[1:]

        center = self.center(encodes[0])

        decodes = self.layer1([center, skips[0]])
        decodes = self.layer2([decodes, skips[1]])
        decodes = self.layer3([decodes, skips[2]])
        decodes = self.layer4([decodes, skips[3]])
        decodes = self.layer5([decodes, None])

        decodes4 = F.interpolate(decodes, x.size()[2:], mode='bilinear')
        outputs = self.final(decodes4)
        
        return outputs


class UNet(nn.Module):
    def __init__(self, encoder, num_classes, **kwargs):
        super().__init__()
        self.encoder = build_encoder(encoder)
        self.decoder = Decoder(num_classes=num_classes, encoder_channels=self.encoder.channels, **kwargs)

    def forward(self, x):
        encodes = self.encoder(x)
        return self.decoder(x, encodes)


class DecoderWithClassificationHead(nn.Module):
    def __init__(self,
                 num_classes,
                 encoder_channels,
                 dropout=0.2,
                 out_channels=[256, 128, 64, 32, 16],
                 use_cls_head=False,
                 **_):
        super().__init__()
        in_channels = encoder_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.use_cls_head = use_cls_head

        self.center = DecoderBlock(in_channels=in_channels[0], out_channels=in_channels[0],
                                   up_sample=False)

        self.layer1 = DecoderBlock(in_channels[0]+in_channels[1], out_channels[0])
        self.layer2 = DecoderBlock(in_channels[2]+out_channels[0], out_channels[1])
        self.layer3 = DecoderBlock(in_channels[3]+out_channels[1], out_channels[2])
        self.layer4 = DecoderBlock(in_channels[4]+out_channels[2], out_channels[3])
        self.layer5 = DecoderBlock(out_channels[3], out_channels[4])

        self.final = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_channels[-1], num_classes, kernel_size=1)
        )

        if self.use_cls_head:
            feature_size = in_channels[0] // 4
            self.cls_head = nn.Sequential(
                    nn.Linear(in_channels[0], feature_size),
                    nn.BatchNorm1d(feature_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(feature_size, num_classes))

    def forward(self, x, encodes):
        skips = encodes[1:]

        center = self.center(encodes[0])

        decodes = self.layer1([center, skips[0]])
        decodes = self.layer2([decodes, skips[1]])
        decodes = self.layer3([decodes, skips[2]])
        decodes = self.layer4([decodes, skips[3]])
        decodes = self.layer5([decodes, None])

        decodes4 = F.interpolate(decodes, x.size()[2:], mode='bilinear')
        outputs = self.final(decodes4)

        if self.use_cls_head:
            p = torch.sigmoid(outputs)
            p = torch.max(p, dim=1, keepdims=True)[0]
            p = F.interpolate(p, center.size()[2:], mode='bilinear')
            p = p.detach()
            c5_pool = center * p
            c5_pool = F.adaptive_avg_pool2d(c5_pool, 1)
            c5_pool = c5_pool.squeeze(-1).squeeze(-1)

            assert len(c5_pool.size()) == 2
            cls_logits = self.cls_head(c5_pool)

            outputs = {
                'logits': outputs, 
                'cls_logits': cls_logits
            }
        
        return outputs


class UNetWithClassificationHead(nn.Module):
    def __init__(self, encoder, num_classes, **kwargs):
        super().__init__()
        self.encoder = build_encoder(encoder)
        self.decoder = DecoderWithClassificationHead(num_classes=num_classes, encoder_channels=self.encoder.channels, **kwargs)

    def forward(self, x):
        encodes = self.encoder(x)
        return self.decoder(x, encodes)
