"""Small segmentation models for functional casting surfaces."""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from src.models.pixel_ae import conv_block


def _resnet18_backbone(pretrained: bool = True):
    if pretrained:
        try:
            return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception as exc:
            warnings.warn(
                f"Could not load torchvision ResNet18 pretrained weights ({exc}). "
                "Falling back to a randomly initialized ResNet18 encoder.",
                RuntimeWarning,
                stacklevel=2,
            )
    return models.resnet18(weights=None)


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if int(in_channels) == conv.in_channels:
        return conv
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        channels_to_copy = min(conv.in_channels, int(in_channels))
        new_conv.weight[:, :channels_to_copy] = conv.weight[:, :channels_to_copy]
        if int(in_channels) > conv.in_channels:
            extra = conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, conv.in_channels :] = extra.repeat(1, int(in_channels) - conv.in_channels, 1, 1) * 0.25
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class FunctionalSurfaceUNetSmall(nn.Module):
    """Lightweight U-Net predicting inspectable functional-surface probability."""

    model_type = "functional_unet_small"

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(conv_block(3, 32), conv_block(32, 32))
        self.down1 = conv_block(32, 64, stride=2)
        self.enc2 = conv_block(64, 64)
        self.down2 = conv_block(64, 128, stride=2)
        self.enc3 = conv_block(128, 128)
        self.down3 = conv_block(128, 256, stride=2)
        self.bottleneck = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.Sequential(conv_block(256, 128), conv_block(128, 128))
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Sequential(conv_block(128, 64), conv_block(64, 64))
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Sequential(conv_block(64, 32), conv_block(32, 32))
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        original_size = images.shape[-2:]
        x1 = self.enc1(images)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        z = self.bottleneck(self.down3(x3))

        y = self.up3(z)
        if y.shape[-2:] != x3.shape[-2:]:
            y = F.interpolate(y, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec3(torch.cat([y, x3], dim=1))
        y = self.up2(y)
        if y.shape[-2:] != x2.shape[-2:]:
            y = F.interpolate(y, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec2(torch.cat([y, x2], dim=1))
        y = self.up1(y)
        if y.shape[-2:] != x1.shape[-2:]:
            y = F.interpolate(y, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec1(torch.cat([y, x1], dim=1))
        logits = self.out(y)
        if logits.shape[-2:] != original_size:
            logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
        return logits


class FunctionalSurfaceUNetResNet18(nn.Module):
    """U-Net decoder on top of a ResNet18 ImageNet encoder."""

    model_type = "functional_unet_resnet18"

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        backbone = self._make_backbone(pretrained=pretrained)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.dec4 = nn.Sequential(conv_block(512, 256), conv_block(256, 256))
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.Sequential(conv_block(256, 128), conv_block(128, 128))
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Sequential(conv_block(128, 64), conv_block(64, 64))
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Sequential(conv_block(128, 64), conv_block(64, 64))
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec0 = nn.Sequential(conv_block(32, 32), conv_block(32, 32))
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    @staticmethod
    def _make_backbone(pretrained: bool = True):
        return _resnet18_backbone(pretrained=pretrained)

    def set_encoder_trainable(self, trainable: bool) -> None:
        for module in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for parameter in module.parameters():
                parameter.requires_grad = bool(trainable)

    @staticmethod
    def _match_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == reference.shape[-2:]:
            return x
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = self.stem(images)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        z = self.layer4(x3)
        return x0, x1, x2, x3, z

    def decode(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], original_size: tuple[int, int]) -> torch.Tensor:
        x0, x1, x2, x3, z = features
        y = self._match_size(self.up4(z), x3)
        y = self.dec4(torch.cat([y, x3], dim=1))
        y = self._match_size(self.up3(y), x2)
        y = self.dec3(torch.cat([y, x2], dim=1))
        y = self._match_size(self.up2(y), x1)
        y = self.dec2(torch.cat([y, x1], dim=1))
        y = self._match_size(self.up1(y), x0)
        y = self.dec1(torch.cat([y, x0], dim=1))
        y = self.up0(y)
        logits = self.out(self.dec0(y))
        if logits.shape[-2:] != original_size:
            logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
        return logits

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        original_size = images.shape[-2:]
        return self.decode(self.encode(images), original_size)


class FunctionalSurfaceUNetResNet18Det1(FunctionalSurfaceUNetResNet18):
    """ResNet18 U-Net with a one-class auxiliary bbox/objectness head."""

    model_type = "functional_unet_resnet18_det1"

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__(pretrained=pretrained)
        self.det_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.det_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
        )
        self.objectness = nn.Linear(128, 1)
        self.bbox = nn.Linear(128, 4)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        original_size = images.shape[-2:]
        features = self.encode(images)
        logits = self.decode(features, original_size)
        det_features = self.det_mlp(self.det_pool(features[-1]))
        return {
            "mask_logits": logits,
            "objectness_logits": self.objectness(det_features),
            "bbox": torch.sigmoid(self.bbox(det_features)),
        }


class FunctionalSurfaceUNetResNet18Det1Context2B(FunctionalSurfaceUNetResNet18Det1):
    """Two-branch model: local crop segmentation plus global image context."""

    model_type = "functional_unet_resnet18_det1_context2b"

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__(pretrained=pretrained)
        global_backbone = self._make_backbone(pretrained=pretrained)
        global_backbone.conv1 = _adapt_first_conv(global_backbone.conv1, 4)
        self.global_stem = nn.Sequential(global_backbone.conv1, global_backbone.bn1, global_backbone.relu)
        self.global_maxpool = global_backbone.maxpool
        self.global_layer1 = global_backbone.layer1
        self.global_layer2 = global_backbone.layer2
        self.global_layer3 = global_backbone.layer3
        self.global_layer4 = global_backbone.layer4
        self.context_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.context_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )

    def set_encoder_trainable(self, trainable: bool) -> None:
        super().set_encoder_trainable(trainable)
        for module in [
            self.global_stem,
            self.global_layer1,
            self.global_layer2,
            self.global_layer3,
            self.global_layer4,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = bool(trainable)

    def encode_global(self, global_image: torch.Tensor, crop_box_mask: torch.Tensor) -> torch.Tensor:
        if crop_box_mask.ndim == 3:
            crop_box_mask = crop_box_mask[:, None, ...]
        if crop_box_mask.shape[-2:] != global_image.shape[-2:]:
            crop_box_mask = F.interpolate(crop_box_mask.float(), size=global_image.shape[-2:], mode="nearest")
        x = torch.cat([global_image, crop_box_mask.to(dtype=global_image.dtype, device=global_image.device)], dim=1)
        x = self.global_stem(x)
        x = self.global_layer1(self.global_maxpool(x))
        x = self.global_layer2(x)
        x = self.global_layer3(x)
        return self.global_layer4(x)

    def forward(
        self,
        images: torch.Tensor,
        *,
        global_image: torch.Tensor | None = None,
        crop_box_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        original_size = images.shape[-2:]
        if global_image is None:
            global_image = images
        if crop_box_mask is None:
            crop_box_mask = torch.ones(
                (images.shape[0], 1, *images.shape[-2:]),
                dtype=images.dtype,
                device=images.device,
            )
        local_features = list(self.encode(images))
        global_z = self.encode_global(global_image, crop_box_mask)
        context = self.context_proj(self.context_pool(global_z))[:, :, None, None]
        local_features[-1] = local_features[-1] + context
        features = tuple(local_features)
        logits = self.decode(features, original_size)
        det_features = self.det_mlp(self.det_pool(features[-1]))
        return {
            "mask_logits": logits,
            "objectness_logits": self.objectness(det_features),
            "bbox": torch.sigmoid(self.bbox(det_features)),
        }


class FunctionalSurfaceUNetResNet18Det1Context2BRecon(FunctionalSurfaceUNetResNet18Det1Context2B):
    """Context2B model with a bottleneck-only semantic reconstruction head.

    The main decoder keeps the U-Net skip connections for precise contours. The
    auxiliary reconstruction head upsamples only the deepest feature map, which
    encourages a global, defect-robust representation of normal surfaces and
    landmarks.
    """

    model_type = "functional_unet_resnet18_det1_context2b_recon"

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__(pretrained=pretrained)
        self.recon_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            conv_block(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            conv_block(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            conv_block(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            conv_block(32, 32),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            conv_block(32, 32),
        )
        self.recon_out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(
        self,
        images: torch.Tensor,
        *,
        global_image: torch.Tensor | None = None,
        crop_box_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        original_size = images.shape[-2:]
        if global_image is None:
            global_image = images
        if crop_box_mask is None:
            crop_box_mask = torch.ones(
                (images.shape[0], 1, *images.shape[-2:]),
                dtype=images.dtype,
                device=images.device,
            )
        local_features = list(self.encode(images))
        global_z = self.encode_global(global_image, crop_box_mask)
        context = self.context_proj(self.context_pool(global_z))[:, :, None, None]
        local_features[-1] = local_features[-1] + context
        features = tuple(local_features)
        logits = self.decode(features, original_size)
        det_features = self.det_mlp(self.det_pool(features[-1]))
        recon_logits = self.recon_out(self.recon_decoder(features[-1]))
        if recon_logits.shape[-2:] != original_size:
            recon_logits = F.interpolate(recon_logits, size=original_size, mode="bilinear", align_corners=False)
        return {
            "mask_logits": logits,
            "objectness_logits": self.objectness(det_features),
            "bbox": torch.sigmoid(self.bbox(det_features)),
            "recon_logits": recon_logits,
        }


class FunctionalSurfaceUNetResNet18Det1ContextFPN(FunctionalSurfaceUNetResNet18Det1):
    """Two-branch U-Net with gated multi-scale global context fusion."""

    model_type = "functional_unet_resnet18_det1_context_fpn"

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__(pretrained=pretrained)
        global_backbone = _resnet18_backbone(pretrained=pretrained)
        global_backbone.conv1 = _adapt_first_conv(global_backbone.conv1, 4)
        self.global_stem = nn.Sequential(global_backbone.conv1, global_backbone.bn1, global_backbone.relu)
        self.global_maxpool = global_backbone.maxpool
        self.global_layer1 = global_backbone.layer1
        self.global_layer2 = global_backbone.layer2
        self.global_layer3 = global_backbone.layer3
        self.global_layer4 = global_backbone.layer4

        self.global_proj1 = nn.Conv2d(64, 64, kernel_size=1)
        self.global_proj2 = nn.Conv2d(128, 128, kernel_size=1)
        self.global_proj3 = nn.Conv2d(256, 256, kernel_size=1)
        self.global_proj4 = nn.Conv2d(512, 512, kernel_size=1)
        self.global_gate1 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.global_gate2 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.global_gate3 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.global_gate4 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def set_encoder_trainable(self, trainable: bool) -> None:
        super().set_encoder_trainable(trainable)
        for module in [
            self.global_stem,
            self.global_layer1,
            self.global_layer2,
            self.global_layer3,
            self.global_layer4,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = bool(trainable)

    @staticmethod
    def _align_feature(feature: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if feature.shape[-2:] == reference.shape[-2:]:
            return feature
        return F.interpolate(feature, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def encode_global_features(
        self,
        global_image: torch.Tensor,
        crop_box_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if crop_box_mask.ndim == 3:
            crop_box_mask = crop_box_mask[:, None, ...]
        if crop_box_mask.shape[-2:] != global_image.shape[-2:]:
            crop_box_mask = F.interpolate(crop_box_mask.float(), size=global_image.shape[-2:], mode="nearest")
        x = torch.cat([global_image, crop_box_mask.to(dtype=global_image.dtype, device=global_image.device)], dim=1)
        g0 = self.global_stem(x)
        g1 = self.global_layer1(self.global_maxpool(g0))
        g2 = self.global_layer2(g1)
        g3 = self.global_layer3(g2)
        g4 = self.global_layer4(g3)
        return g0, g1, g2, g3, g4

    def fuse_features(
        self,
        local_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        global_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0, x1, x2, x3, z = local_features
        _g0, g1, g2, g3, g4 = global_features
        x1 = x1 + self.global_gate1 * self._align_feature(self.global_proj1(g1), x1)
        x2 = x2 + self.global_gate2 * self._align_feature(self.global_proj2(g2), x2)
        x3 = x3 + self.global_gate3 * self._align_feature(self.global_proj3(g3), x3)
        z = z + self.global_gate4 * self._align_feature(self.global_proj4(g4), z)
        return x0, x1, x2, x3, z

    def forward(
        self,
        images: torch.Tensor,
        *,
        global_image: torch.Tensor | None = None,
        crop_box_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        original_size = images.shape[-2:]
        if global_image is None:
            global_image = images
        if crop_box_mask is None:
            crop_box_mask = torch.ones(
                (images.shape[0], 1, *images.shape[-2:]),
                dtype=images.dtype,
                device=images.device,
            )
        local_features = self.encode(images)
        global_features = self.encode_global_features(global_image, crop_box_mask)
        features = self.fuse_features(local_features, global_features)
        logits = self.decode(features, original_size)
        det_features = self.det_mlp(self.det_pool(features[-1]))
        return {
            "mask_logits": logits,
            "objectness_logits": self.objectness(det_features),
            "bbox": torch.sigmoid(self.bbox(det_features)),
        }


class FunctionalSurfaceUNetResNet18Det1ContextFPNLight(FunctionalSurfaceUNetResNet18Det1ContextFPN):
    """Conservative multi-scale context: fuse only deep global features."""

    model_type = "functional_unet_resnet18_det1_context_fpn_light"

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__(pretrained=pretrained)
        self.global_gate1 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
        self.global_gate2 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
        self.global_gate3 = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.global_gate4 = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

    def fuse_features(
        self,
        local_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        global_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0, x1, x2, x3, z = local_features
        _g0, _g1, _g2, g3, g4 = global_features
        x3 = x3 + self.global_gate3 * self._align_feature(self.global_proj3(g3), x3)
        z = z + self.global_gate4 * self._align_feature(self.global_proj4(g4), z)
        return x0, x1, x2, x3, z


def build_segmentation_model(model_type: str) -> nn.Module:
    if model_type == "functional_unet_small":
        return FunctionalSurfaceUNetSmall()
    if model_type == "functional_unet_resnet18":
        return FunctionalSurfaceUNetResNet18(pretrained=True)
    if model_type == "functional_unet_resnet18_det1":
        return FunctionalSurfaceUNetResNet18Det1(pretrained=True)
    if model_type == "functional_unet_resnet18_det1_context2b":
        return FunctionalSurfaceUNetResNet18Det1Context2B(pretrained=True)
    if model_type == "functional_unet_resnet18_det1_context2b_recon":
        return FunctionalSurfaceUNetResNet18Det1Context2BRecon(pretrained=True)
    if model_type == "functional_unet_resnet18_det1_context_fpn":
        return FunctionalSurfaceUNetResNet18Det1ContextFPN(pretrained=True)
    if model_type == "functional_unet_resnet18_det1_context_fpn_light":
        return FunctionalSurfaceUNetResNet18Det1ContextFPNLight(pretrained=True)
    if model_type == "functional_unet_resnet18_scratch":
        return FunctionalSurfaceUNetResNet18(pretrained=False)
    raise ValueError(
        "Unknown functional surface model. Use functional_unet_small, functional_unet_resnet18, "
        "functional_unet_resnet18_scratch, functional_unet_resnet18_det1, "
        "functional_unet_resnet18_det1_context2b, functional_unet_resnet18_det1_context2b_recon, "
        "functional_unet_resnet18_det1_context_fpn, "
        "or functional_unet_resnet18_det1_context_fpn_light."
    )


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    prob = prob * valid
    target = target * valid
    intersection = (prob * target).sum(dim=(1, 2, 3))
    denominator = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def weighted_dice_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    pixel_weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pixel_weight is None:
        return dice_loss_from_logits(logits, target, valid, eps=eps)
    prob = torch.sigmoid(logits)
    weight = valid * pixel_weight.to(device=logits.device, dtype=logits.dtype)
    prob = prob * weight
    target = target * weight
    intersection = (prob * target).sum(dim=(1, 2, 3))
    denominator = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def tv_loss_from_logits(logits: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dx = torch.abs(prob[..., :, 1:] - prob[..., :, :-1]).mean()
    dy = torch.abs(prob[..., 1:, :] - prob[..., :-1, :]).mean()
    return dx + dy


def focal_tversky_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    pixel_weight: torch.Tensor | None = None,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    prob_raw = torch.sigmoid(logits)
    target_binary = target * valid
    weight = valid
    if pixel_weight is not None:
        weight = weight * pixel_weight.to(device=logits.device, dtype=logits.dtype)
    prob = prob_raw * weight
    target_weighted = target * weight
    tp = (prob * target_binary).sum(dim=(1, 2, 3))
    fp = (prob_raw * (1.0 - target_binary) * weight).sum(dim=(1, 2, 3))
    fn = ((1.0 - prob_raw) * target_weighted).sum(dim=(1, 2, 3))
    tversky = (tp + eps) / (tp + float(alpha) * fp + float(beta) * fn + eps)
    loss = (1.0 - tversky).clamp_min(0.0)
    if float(gamma) != 1.0:
        loss = loss.pow(float(gamma))
    return loss.mean()


def binary_boundary_from_mask(mask: torch.Tensor, width: int = 3) -> torch.Tensor:
    radius = max(1, int(width))
    kernel = 2 * radius + 1
    binary = (mask > 0.5).float()
    dilated = F.max_pool2d(binary, kernel_size=kernel, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - binary, kernel_size=kernel, stride=1, padding=radius)
    return (dilated - eroded).clamp(0.0, 1.0)


def binary_interior_from_mask(mask: torch.Tensor, width: int = 5) -> torch.Tensor:
    radius = max(1, int(width))
    kernel = 2 * radius + 1
    binary = (mask > 0.5).float()
    return (1.0 - F.max_pool2d(1.0 - binary, kernel_size=kernel, stride=1, padding=radius)).clamp(0.0, 1.0)


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pixel_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    weight = mask.to(device=logits.device, dtype=logits.dtype)
    if pixel_weight is not None:
        weight = weight * pixel_weight.to(device=logits.device, dtype=logits.dtype)
    if float(weight.detach().sum().cpu()) <= 0:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target.to(device=logits.device, dtype=logits.dtype),
        reduction="none",
    )
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def weak_surface_loss(
    logits: torch.Tensor,
    *,
    positive: torch.Tensor,
    negative: torch.Tensor,
    pseudo: torch.Tensor,
    ignore: torch.Tensor,
    weight_map: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    dice_weight: float = 0.5,
    tv_weight: float = 0.01,
    focal_tversky_weight: float = 0.0,
    focal_tversky_alpha: float = 0.3,
    focal_tversky_beta: float = 0.7,
    focal_tversky_gamma: float = 1.0,
    boundary_loss_weight: float = 0.0,
    boundary_width: int = 3,
    interior_loss_weight: float = 0.0,
    interior_width: int = 5,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size = logits.shape[0]
    if sample_weight is None:
        weights = torch.ones(batch_size, device=logits.device, dtype=logits.dtype)
    else:
        weights = sample_weight.to(device=logits.device, dtype=logits.dtype).reshape(batch_size)
    weights = torch.clamp(weights, min=0.0)

    sample_losses = []
    bce_values = []
    dice_values = []
    tv_values = []
    tversky_values = []
    boundary_values = []
    interior_values = []
    for idx in range(batch_size):
        sample_logits = logits[idx : idx + 1]
        sample_positive = positive[idx : idx + 1]
        sample_negative = negative[idx : idx + 1]
        sample_pseudo = pseudo[idx : idx + 1]
        sample_ignore = ignore[idx : idx + 1]
        sample_weight_map = weight_map[idx : idx + 1] if weight_map is not None else None

        pos_valid = sample_positive > 0.5
        neg_valid = sample_negative > 0.5
        bce_terms = []
        if pos_valid.any():
            bce_pos = F.binary_cross_entropy_with_logits(
                sample_logits[pos_valid],
                torch.ones_like(sample_logits[pos_valid]),
                reduction="none",
            )
            if sample_weight_map is not None:
                bce_pos = bce_pos * sample_weight_map[pos_valid].to(device=logits.device, dtype=logits.dtype)
                bce_terms.append(bce_pos.sum() / sample_weight_map[pos_valid].sum().clamp_min(1e-6))
            else:
                bce_terms.append(bce_pos.mean())
        if neg_valid.any():
            bce_neg = F.binary_cross_entropy_with_logits(
                sample_logits[neg_valid],
                torch.zeros_like(sample_logits[neg_valid]),
                reduction="none",
            )
            if sample_weight_map is not None:
                bce_neg = bce_neg * sample_weight_map[neg_valid].to(device=logits.device, dtype=logits.dtype)
                bce_terms.append(bce_neg.sum() / sample_weight_map[neg_valid].sum().clamp_min(1e-6))
            else:
                bce_terms.append(bce_neg.mean())
        bce = torch.stack(bce_terms).mean() if bce_terms else sample_logits.sum() * 0.0
        valid = (~(sample_ignore > 0.5)).float()
        dice = weighted_dice_loss_from_logits(sample_logits, sample_pseudo, valid, sample_weight_map)
        tv = tv_loss_from_logits(sample_logits)
        tversky = focal_tversky_loss_from_logits(
            sample_logits,
            sample_pseudo,
            valid,
            sample_weight_map,
            alpha=focal_tversky_alpha,
            beta=focal_tversky_beta,
            gamma=focal_tversky_gamma,
        )
        boundary_mask = binary_boundary_from_mask(sample_positive, width=boundary_width) * valid
        boundary = masked_bce_with_logits(sample_logits, sample_positive, boundary_mask, sample_weight_map)
        interior_mask = binary_interior_from_mask(sample_positive, width=interior_width) * valid
        interior = masked_bce_with_logits(sample_logits, torch.ones_like(sample_logits), interior_mask, sample_weight_map)
        sample_loss = (
            bce
            + float(dice_weight) * dice
            + float(tv_weight) * tv
            + float(focal_tversky_weight) * tversky
            + float(boundary_loss_weight) * boundary
            + float(interior_loss_weight) * interior
        )

        sample_losses.append(sample_loss)
        bce_values.append(bce.detach())
        dice_values.append(dice.detach())
        tv_values.append(tv.detach())
        tversky_values.append(tversky.detach())
        boundary_values.append(boundary.detach())
        interior_values.append(interior.detach())

    losses = torch.stack(sample_losses)
    weight_sum = weights.sum().clamp_min(1e-6)
    loss = (losses * weights).sum() / weight_sum
    bce = (torch.stack(bce_values) * weights).sum() / weight_sum
    dice = (torch.stack(dice_values) * weights).sum() / weight_sum
    tv = (torch.stack(tv_values) * weights).sum() / weight_sum
    tversky = (torch.stack(tversky_values) * weights).sum() / weight_sum
    boundary = (torch.stack(boundary_values) * weights).sum() / weight_sum
    interior = (torch.stack(interior_values) * weights).sum() / weight_sum
    return loss, {
        "loss": float(loss.detach().cpu()),
        "bce": float(bce.detach().cpu()),
        "dice": float(dice.detach().cpu()),
        "tv": float(tv.detach().cpu()),
        "focal_tversky": float(tversky.detach().cpu()),
        "boundary": float(boundary.detach().cpu()),
        "interior": float(interior.detach().cpu()),
    }






