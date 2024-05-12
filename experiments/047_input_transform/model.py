# Several SqueezeFormer components where copied/ adapted from https://github.com/upskyy/Squeezeformer/

import json
import math
import typing
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parameter import Parameter


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})"
        )


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=4, kernel_size=3):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class CNN(nn.Module):
    def __init__(self, F1: int, D: int = 2, final_dropout=0.25):
        super(CNN, self).__init__()
        self.drop_out = final_dropout
        self.att = cbam_block(D * F1)
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((7, 7, 0, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 16),
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(F1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 8)),
        )
        self.block_2 = nn.Sequential(
            nn.ZeroPad2d((7, 7, 0, 0)),
            nn.Conv2d(
                in_channels=F1,
                out_channels=D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            Swish(),  # GLU(dim=1),)
            nn.Conv2d(
                in_channels=D * F1,
                out_channels=D * F1,
                kernel_size=(1, 16),
                stride=(1, 2),
                bias=False,
                groups=D * F1,
            ),
            nn.BatchNorm2d(D * F1),
            Swish(),
            nn.Conv2d(
                in_channels=D * F1,
                out_channels=D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(D * F1),
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=D * F1,
                out_channels=D * D * F1,  # D * D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=4,
                bias=False,
            ),
            Swish(),  # GLU(dim=1),)
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * D * F1,
                kernel_size=(3, 1),
                stride=(1, 1),
                groups=D * D * F1,
                bias=False,
            ),
            nn.BatchNorm2d(D * D * F1),
            Swish(),
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * F1,  # D * D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=4,
                bias=False,
            ),
            nn.BatchNorm2d(D * F1),  # D * D * F1
        )
        self.block_4 = nn.Sequential(
            nn.ZeroPad2d((4, 3, 0, 0)),
            nn.Conv2d(
                in_channels=D * F1,
                out_channels=D * D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            Swish(),  # GLU(dim=1),)
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * D * F1,
                kernel_size=(1, 8),
                stride=(1, 1),
                bias=False,
                groups=D * D * F1,
            ),
            nn.BatchNorm2d(D * D * F1),
            Swish(),
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
                groups=4,
            ),
            nn.BatchNorm2d(D * F1),
            nn.AvgPool2d((1, 2)),
            # nn.ReLU(inplace=True),
        )
        """
        self.block_4 = nn.Sequential(
            nn.ZeroPad2d((4, 3, 0, 0)),
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * D * F1,
                kernel_size=(1, 8),
                stride=(1, 1),
                groups=D * D * F1,
                bias=False
            ),
            nn.BatchNorm2d(D * D * F1),
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * D * D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=4,
                bias=False
            ),
            nn.BatchNorm2d(D * D * D * F1),
            nn.ReLU(inplace=True),
#             nn.AvgPool2d((1, 16))
        )
        """

    def forward(self, x):
        # print(x.shape)
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        #         x1 = x.mean(2, keepdim=True)
        #         x2 = torch.norm(x, p=2, dim=2, keepdim=True)
        #         x3 = torch.norm(x, p=np.inf, dim=2, keepdim=True)
        #         x = torch.cat([x1, x2, x3], 2)

        # print(x.shape)
        x = self.att(x)

        # print(x.shape)
        x = self.block_3(x)

        # print(x.shape)
        x = self.block_4(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    batch_shape = (384, 1, 30, 60)

    model = CNN(F1=184, D=2, final_dropout=0.1)

    summary(model, input_size=batch_shape)
