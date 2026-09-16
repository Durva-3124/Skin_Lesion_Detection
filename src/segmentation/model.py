"""
src/segmentation/model.py
U-Net with swappable encoder: VGG16 (research/benchmark) or MobileNetV2 (on-device).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad x to match skip spatial dims if they differ by 1 pixel
        if x.shape[2:] != skip.shape[2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetVGG16(nn.Module):
    """U-Net with VGG16 encoder — research/benchmark segmentation model."""

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        features = vgg.features

        self.enc1 = features[:4]    # 64
        self.enc2 = features[4:9]   # 128
        self.enc3 = features[9:16]  # 256
        self.enc4 = features[16:23] # 512
        self.enc5 = features[23:30] # 512

        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = ConvBlock(512, 1024)

        self.up5 = UpBlock(1024, 512, 512)
        self.up4 = UpBlock(512, 512, 256)
        self.up3 = UpBlock(256, 256, 128)
        self.up2 = UpBlock(128, 128, 64)
        self.up1 = UpBlock(64, 64, 32)

        self.out = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)           # 64 ch
        e2 = self.enc2(self.pool(e1))  # 128 ch
        e3 = self.enc3(self.pool(e2))  # 256 ch
        e4 = self.enc4(self.pool(e3))  # 512 ch
        e5 = self.enc5(self.pool(e4))  # 512 ch

        b = self.bottleneck(e5)     # no extra pool — keeps spatial dims

        d5 = self.up5(b, e5)
        d4 = self.up4(d5, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.out(d1)


class UNetMobileNetV2(nn.Module):
    """U-Net with MobileNetV2 encoder — lightweight on-device segmentation model."""

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        mob = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        features = mob.features

        # MobileNetV2 encoder stages
        self.enc0 = features[0]         # stride 2, 32 ch
        self.enc1 = features[1:3]       # stride 2, 16->24 ch
        self.enc2 = features[3:5]       # stride 2, 32 ch
        self.enc3 = features[5:8]       # stride 2, 64 ch
        self.enc4 = features[8:14]      # stride 2, 96->160 ch
        self.enc5 = features[14:18]     # 320 ch

        self.bottleneck = ConvBlock(320, 512)

        self.up4 = UpBlock(512, 96, 256)
        self.up3 = UpBlock(256, 32, 128)
        self.up2 = UpBlock(128, 24, 64)
        self.up1 = UpBlock(64, 16, 32)
        self.up0 = UpBlock(32, 32, 16)

        self.out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        b = self.bottleneck(e5)

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        d0 = self.up0(d1, e0)

        return self.out(d0)


def get_segmentation_model(encoder="vgg16", pretrained=True):
    if encoder == "vgg16":
        return UNetVGG16(pretrained=pretrained)
    elif encoder == "mobilenetv2":
        return UNetMobileNetV2(pretrained=pretrained)
    raise ValueError(f"Unknown encoder: {encoder}. Choose 'vgg16' or 'mobilenetv2'.")
