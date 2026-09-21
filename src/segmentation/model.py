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

        # Encoder stages — verified channel/spatial map at 256x256 input:
        # features[0]:  32ch 128x128  (stride-2)
        # features[1:3]: 24ch 64x64   (stride-2 at features[2])
        # features[3:7]: 32ch 32x32   (stride-2 at features[4])
        # features[7:11]: 64ch 16x16  (stride-2 at features[7])
        # features[11:14]: 96ch 16x16 (no stride change)
        # features[14:17]: 160ch 8x8  (stride-2 at features[14])
        # features[17]: 320ch 8x8
        self.enc0 = features[0]          # 32ch,  128x128
        self.enc1 = features[1:3]        # 24ch,  64x64
        self.enc2 = features[3:7]        # 32ch,  32x32
        self.enc3 = features[7:11]       # 64ch,  16x16
        self.enc4 = features[11:14]      # 96ch,  16x16
        self.enc5 = features[14:17]      # 160ch, 8x8
        self.enc6 = features[17:18]      # 320ch, 8x8

        self.bottleneck = ConvBlock(320, 512)

        self.up6 = UpBlock(512, 160, 256)   # upsample 8->16, skip=enc5 160ch
        self.up5 = UpBlock(256,  96, 128)   # no upsample (enc4 same spatial), skip=enc4 96ch
        self.up4 = UpBlock(128,  64, 128)   # no upsample (enc3 same spatial), skip=enc3 64ch
        self.up3 = UpBlock(128,  32,  64)   # upsample 16->32, skip=enc2 32ch
        self.up2 = UpBlock( 64,  24,  32)   # upsample 32->64, skip=enc1 24ch
        self.up1 = UpBlock( 32,  32,  16)   # upsample 64->128, skip=enc0 32ch

        self.out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        e0 = self.enc0(x)    # 32ch,  128x128
        e1 = self.enc1(e0)   # 24ch,  64x64
        e2 = self.enc2(e1)   # 32ch,  32x32
        e3 = self.enc3(e2)   # 64ch,  16x16
        e4 = self.enc4(e3)   # 96ch,  16x16
        e5 = self.enc5(e4)   # 160ch, 8x8
        e6 = self.enc6(e5)   # 320ch, 8x8

        b  = self.bottleneck(e6)   # 512ch, 8x8

        d6 = self.up6(b,  e5)   # 256ch, 16x16
        d5 = self.up5(d6, e4)   # 128ch, 16x16
        d4 = self.up4(d5, e3)   # 128ch, 16x16
        d3 = self.up3(d4, e2)   # 64ch,  32x32
        d2 = self.up2(d3, e1)   # 32ch,  64x64
        d1 = self.up1(d2, e0)   # 16ch,  128x128

        # Final upsample to input resolution
        d0 = torch.nn.functional.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        return self.out(d0)


def get_segmentation_model(encoder="vgg16", pretrained=True):
    if encoder == "vgg16":
        return UNetVGG16(pretrained=pretrained)
    elif encoder == "mobilenetv2":
        return UNetMobileNetV2(pretrained=pretrained)
    raise ValueError(f"Unknown encoder: {encoder}. Choose 'vgg16' or 'mobilenetv2'.")
