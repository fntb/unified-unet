from typing import (
    List,
    Optional,
    Literal
)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x

class UNetDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv = DoubleConv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor):
        x = self.conv(x)

        return F.max_pool2d(x, kernel_size=2, stride=2), x
    
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual_in_channels: Optional[int] = None) -> None:
        super().__init__()

        if residual_in_channels is None:
            residual_in_channels = out_channels

        # self.up = nn.ConvTranspose2d(in_channels, in_channels - residual_in_channels, kernel_size=2, stride=2)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, in_channels - residual_in_channels, kernel_size=1)
        )

        self.conv = DoubleConv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, x_residual: torch.Tensor):

        x = self.up(x)

        if x.shape != x_residual.shape:
            dw = x_residual.shape[3] - x.shape[3]
            dh = x_residual.shape[2] - x.shape[2]
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])

        x = torch.concat([x_residual, x], dim=1)

        x = self.conv(x)

        return x
    
class UNet(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_classes: int, 
        dims: List[int] = [64, 128, 256, 512, 1024],
        head: Literal["segmentation", "classification"] = "segmentation"
    ) -> None:
        super().__init__()

        self.task = head

        self.encoder = nn.ModuleList([
            UNetDownBlock(in_channels=in_channels, out_channels=dims[0])
        ] + [
            UNetDownBlock(dims[i], dims[i+1])
            for i in range(len(dims) - 2)
        ])

        self.embedding = DoubleConv2d(dims[-2], dims[-1], kernel_size=3, padding=1)

        if self.task == "classification":
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(dims[-1], out_classes)
            )
        elif self.task == "segmentation":
            self.decoder = nn.ModuleList([
                UNetUpBlock(dims[i], dims[i - 1])
                for i in reversed(range(1, len(dims)))
            ])

            self.head = nn.Conv2d(in_channels=dims[0], out_channels=out_classes, kernel_size=1)
        else:
            raise ValueError(f"Unknown head '{self.task}'")

    def forward(self, x: torch.Tensor):
        residuals = []

        for encoder in self.encoder:
            x, x_residual = encoder(x)
            residuals.append(x_residual)

        x = self.embedding(x)

        if self.task == "classification":
            return self.head(x)
        elif self.task == "segmentation":
            for decoder, x_residual in zip(self.decoder, reversed(residuals)):
                x = decoder(x, x_residual)

            return self.head(x)
        else:
            raise ValueError(f"Unknown head '{self.task}'")



        