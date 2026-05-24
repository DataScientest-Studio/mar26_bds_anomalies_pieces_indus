"""Reconstruction models used to learn normal-only visual representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class PixelAEParams:
    category: str
    model_type: str = "custom_autoencoder"
    backbone: str = "resnet18"
    input_size: int = 256
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 1e-3
    loss: str = "l1"
    device: str = "auto"
    seed: int = 42


def conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def deconv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ResidualConvBlock(nn.Module):
    """Small residual block used by inpainting decoders."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.activation(images + self.block(images))


class CustomConvAutoencoder(nn.Module):
    """Small convolutional autoencoder trained from scratch on normal images."""

    feature_dim = 128

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(3, 32, stride=2),
            conv_block(32, 32),
            conv_block(32, 64, stride=2),
            conv_block(64, 64),
            conv_block(64, 128, stride=2),
            conv_block(128, 128),
        )
        self.decoder = nn.Sequential(
            deconv_block(128, 64),
            conv_block(64, 64),
            deconv_block(64, 32),
            conv_block(32, 32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(images))


class CustomUNetAutoencoder(nn.Module):
    """Lightweight U-Net autoencoder for richer normal-image reconstruction."""

    feature_dim = 256

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(conv_block(3, 32), conv_block(32, 32))
        self.down1 = conv_block(32, 64, stride=2)
        self.enc2 = conv_block(64, 64)
        self.down2 = conv_block(64, 128, stride=2)
        self.enc3 = conv_block(128, 128)
        self.down3 = conv_block(128, 256, stride=2)
        self.bottleneck = nn.Sequential(conv_block(256, 256), conv_block(256, 256))

        self.up3 = deconv_block(256, 128)
        self.dec3 = nn.Sequential(conv_block(256, 128), conv_block(128, 128))
        self.up2 = deconv_block(128, 64)
        self.dec2 = nn.Sequential(conv_block(128, 64), conv_block(64, 64))
        self.up1 = deconv_block(64, 32)
        self.dec1 = nn.Sequential(conv_block(64, 32), conv_block(32, 32))
        self.out = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(images)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        return self.bottleneck(self.down3(x3))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(images)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        z = self.bottleneck(self.down3(x3))

        y = self.up3(z)
        if y.shape[-2:] != x3.shape[-2:]:
            y = nn.functional.interpolate(y, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec3(torch.cat([y, x3], dim=1))

        y = self.up2(y)
        if y.shape[-2:] != x2.shape[-2:]:
            y = nn.functional.interpolate(y, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec2(torch.cat([y, x2], dim=1))

        y = self.up1(y)
        if y.shape[-2:] != x1.shape[-2:]:
            y = nn.functional.interpolate(y, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec1(torch.cat([y, x1], dim=1))

        recon = self.out(y)
        if recon.shape[-2:] != images.shape[-2:]:
            recon = nn.functional.interpolate(
                recon, size=images.shape[-2:], mode="bilinear", align_corners=False
            )
        return recon


class CustomConstrainedUNetAutoencoder(nn.Module):
    """U-Net autoencoder with constrained skips and a noisy bottleneck."""

    feature_dim = 256

    def __init__(self, bottleneck_dropout: float = 0.2, latent_noise_std: float = 0.05):
        super().__init__()
        self.latent_noise_std = float(latent_noise_std)
        self.enc1 = nn.Sequential(conv_block(3, 32), conv_block(32, 32))
        self.down1 = conv_block(32, 64, stride=2)
        self.enc2 = conv_block(64, 64)
        self.down2 = conv_block(64, 128, stride=2)
        self.enc3 = conv_block(128, 128)
        self.down3 = conv_block(128, 256, stride=2)
        self.bottleneck = nn.Sequential(
            conv_block(256, 256),
            nn.Dropout2d(p=bottleneck_dropout),
            conv_block(256, 256),
        )

        self.up3 = deconv_block(256, 128)
        self.dec3 = nn.Sequential(conv_block(256, 128), conv_block(128, 128))
        self.up2 = deconv_block(128, 64)
        self.dec2 = nn.Sequential(conv_block(128, 64), conv_block(64, 64))
        self.up1 = deconv_block(64, 32)
        self.dec1 = nn.Sequential(conv_block(32, 32), conv_block(32, 32))
        self.out = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def _bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        if self.training and self.latent_noise_std > 0:
            z = z + torch.randn_like(z) * self.latent_noise_std
        return z

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(images)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        return self._bottleneck(self.down3(x3))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(images)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        z = self._bottleneck(self.down3(x3))

        y = self.up3(z)
        if y.shape[-2:] != x3.shape[-2:]:
            y = nn.functional.interpolate(y, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec3(torch.cat([y, x3], dim=1))

        y = self.up2(y)
        if y.shape[-2:] != x2.shape[-2:]:
            y = nn.functional.interpolate(y, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec2(torch.cat([y, x2], dim=1))

        y = self.up1(y)
        if y.shape[-2:] != x1.shape[-2:]:
            y = nn.functional.interpolate(y, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec1(y)

        recon = self.out(y)
        if recon.shape[-2:] != images.shape[-2:]:
            recon = nn.functional.interpolate(
                recon, size=images.shape[-2:], mode="bilinear", align_corners=False
            )
        return recon


class CustomMaskConditionedInpaintingAutoencoder(nn.Module):
    """Constrained U-Net inpainting AE conditioned by an explicit mask channel."""

    feature_dim = 256
    mask_conditioned = True

    def __init__(self, bottleneck_dropout: float = 0.2, latent_noise_std: float = 0.05):
        super().__init__()
        self.latent_noise_std = float(latent_noise_std)
        self.enc1 = nn.Sequential(conv_block(4, 32), conv_block(32, 32))
        self.down1 = conv_block(32, 64, stride=2)
        self.enc2 = conv_block(64, 64)
        self.down2 = conv_block(64, 128, stride=2)
        self.enc3 = conv_block(128, 128)
        self.down3 = conv_block(128, 256, stride=2)
        self.bottleneck = nn.Sequential(
            conv_block(256, 256),
            nn.Dropout2d(p=bottleneck_dropout),
            ResidualConvBlock(256),
            conv_block(256, 256),
        )

        self.up3 = deconv_block(256, 128)
        self.dec3 = nn.Sequential(conv_block(256, 128), ResidualConvBlock(128))
        self.up2 = deconv_block(128, 64)
        self.dec2 = nn.Sequential(conv_block(128, 64), ResidualConvBlock(64))
        self.up1 = deconv_block(64, 32)
        self.dec1 = nn.Sequential(conv_block(32, 32), ResidualConvBlock(32))
        self.refine = nn.Sequential(
            ResidualConvBlock(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def _with_mask(self, images: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if images.shape[1] == 4:
            if mask is not None:
                raise ValueError("Pass either a 4-channel tensor or images + mask, not both.")
            return images
        if images.shape[1] != 3:
            raise ValueError("Expected RGB images or RGB+mask tensors.")
        if mask is None:
            mask = torch.zeros(
                (images.shape[0], 1, images.shape[2], images.shape[3]),
                device=images.device,
                dtype=images.dtype,
            )
        if mask.shape[1] != 1:
            raise ValueError("Mask must have shape (B, 1, H, W).")
        if mask.shape[-2:] != images.shape[-2:]:
            mask = nn.functional.interpolate(mask, size=images.shape[-2:], mode="nearest")
        return torch.cat([images, mask.to(device=images.device, dtype=images.dtype)], dim=1)

    def _bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        if self.training and self.latent_noise_std > 0:
            z = z + torch.randn_like(z) * self.latent_noise_std
        return z

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = self._with_mask(images)
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        return self._bottleneck(self.down3(x3))

    def forward(self, images: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self._with_mask(images, mask)
        original_size = x.shape[-2:]
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        z = self._bottleneck(self.down3(x3))

        y = self.up3(z)
        if y.shape[-2:] != x3.shape[-2:]:
            y = nn.functional.interpolate(y, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec3(torch.cat([y, x3], dim=1))

        y = self.up2(y)
        if y.shape[-2:] != x2.shape[-2:]:
            y = nn.functional.interpolate(y, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec2(torch.cat([y, x2], dim=1))

        y = self.up1(y)
        if y.shape[-2:] != original_size:
            y = nn.functional.interpolate(y, size=original_size, mode="bilinear", align_corners=False)
        y = self.dec1(y)
        recon = self.out(self.refine(y))
        if recon.shape[-2:] != original_size:
            recon = nn.functional.interpolate(recon, size=original_size, mode="bilinear", align_corners=False)
        return recon


def build_resnet(backbone: str) -> nn.Module:
    name = backbone.lower()
    if name == "resnet18":
        return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    raise ValueError("Unsupported backbone. Use resnet18.")


def resnet_layer_channels(backbone: str) -> dict[str, int]:
    if backbone.lower() == "resnet18":
        return {"layer2": 128, "layer3": 256, "layer4": 512}
    raise ValueError("Unsupported backbone. Use resnet18.")


class ResNetReconstructionAutoencoder(nn.Module):
    """ResNet encoder fine-tuned with a lightweight reconstruction decoder."""

    def __init__(self, backbone: str = "resnet18", trainable_from: str = "layer3"):
        super().__init__()
        self.backbone_name = backbone
        self.encoder = build_resnet(backbone)
        channels = resnet_layer_channels(backbone)["layer4"]
        self.decoder = nn.Sequential(
            deconv_block(channels, 512),
            deconv_block(512, 256),
            deconv_block(256, 128),
            deconv_block(128, 64),
            deconv_block(64, 32),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )
        self.freeze_until(trainable_from)

    def freeze_until(self, trainable_from: str) -> None:
        order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
        train = False
        for name in order:
            if name == trainable_from:
                train = True
            module = getattr(self.encoder, name)
            for param in module.parameters():
                param.requires_grad_(train)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = self.encoder.conv1(images)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        recon = self.decoder(self.encode(images))
        if recon.shape[-2:] != images.shape[-2:]:
            recon = nn.functional.interpolate(
                recon, size=images.shape[-2:], mode="bilinear", align_corners=False
            )
        return recon


def build_pixel_autoencoder(
    model_type: str,
    backbone: str = "resnet18",
    trainable_from: str = "layer3",
) -> nn.Module:
    if model_type == "custom_autoencoder":
        return CustomConvAutoencoder()
    if model_type == "custom_unet_autoencoder":
        return CustomUNetAutoencoder()
    if model_type == "custom_constrained_unet_autoencoder":
        return CustomConstrainedUNetAutoencoder()
    if model_type == "custom_mask_conditioned_inpainting_autoencoder":
        return CustomMaskConditionedInpaintingAutoencoder()
    if model_type == "resnet_reconstruction_finetuned":
        return ResNetReconstructionAutoencoder(backbone=backbone, trainable_from=trainable_from)
    raise ValueError(
        "Unknown model_type. Use custom_autoencoder, custom_unet_autoencoder, "
        "custom_constrained_unet_autoencoder, custom_mask_conditioned_inpainting_autoencoder "
        "or resnet_reconstruction_finetuned."
    )





