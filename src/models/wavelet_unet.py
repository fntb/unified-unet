import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, cast

from .unet import DoubleConv2d

class WaveletUNetEncoderBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        kernel = torch.tensor([
            [1., 1., 1., 1.], 
            [1., -1., 1., -1.], 
            [1., 1., -1., -1.], 
            [1., -1., -1., 1.]
        ]) / 2.0

        self.register_buffer("weight", kernel.view(4, 1, 2, 2).repeat(in_channels, 1, 1, 1))
        
        self.in_channels = in_channels

    def forward(self, x):
        x = F.conv2d(x, cast(torch.Tensor, self.weight), stride=2, groups=self.in_channels)
        
        return x[:, :self.in_channels, :, :], x[:, self.in_channels:, :, :]

class WaveletUNetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.up = nn.Conv2d(in_channels, in_channels * 4, kernel_size=1)
        
        self.conv = DoubleConv2d(in_channels * 4, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, x_residual: torch.Tensor):
        # IWT Approximation
        x = self.up(x)
        x = F.pixel_shuffle(x, upscale_factor=2)
        
        if x.shape[2:] != x_residual.shape[2:]:
            x = F.interpolate(x, size=x_residual.shape[2:], mode="bilinear", align_corners=True)

        x = torch.cat([x, x_residual], dim=1) 

        return self.conv(x)

class WaveletUNet(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_classes: int, 
        stem_dim: int = 32,
        depth: int = 4,
        head: Literal["segmentation", "classification"] = "segmentation"
    ) -> None:
        super().__init__()
        self.task = head
        self.stem = nn.Conv2d(in_channels, stem_dim, kernel_size=3, padding=1)

        self.encoder = nn.ModuleList([
            WaveletUNetEncoderBlock(stem_dim) for _ in range(depth)
        ])

        self.bottleneck = DoubleConv2d(stem_dim, stem_dim, kernel_size=3, padding=1)

        if self.task == "segmentation":
            self.decoder = nn.ModuleList([
                WaveletUNetDecoderBlock(stem_dim) for _ in range(depth)
            ])

            self.head = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(stem_dim, stem_dim, kernel_size=3, padding=1),
                nn.Conv2d(stem_dim, out_classes, kernel_size=1),
            )
        elif self.task == "classification":
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(stem_dim, out_classes)
            )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        
        residuals = []
        for enc in self.encoder:
            x, res = enc(x)
            residuals.append(res)

        x = self.bottleneck(x)

        if self.task == "segmentation":
            for dec, res in zip(self.decoder, reversed(residuals)):
                x = dec(x, res)
                
            return self.head(x)
        elif self.task == "classification":
            return self.head(x)
        