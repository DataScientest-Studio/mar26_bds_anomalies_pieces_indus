"""Feature reconstruction autoencoders for anomaly detection experiments."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from src.models.pixel_ae import build_resnet, conv_block


RESNET_LAYER_CHANNELS = {
    "resnet18": {
        "layer1": 64,
        "layer2": 128,
        "layer3": 256,
    },
}


class ResNetTeacherFeatures(nn.Module):
    """Frozen ResNet teacher returning selected intermediate feature maps."""

    def __init__(self, backbone: str = "resnet18", layers: Iterable[str] = ("layer2", "layer3")):
        super().__init__()
        if backbone not in RESNET_LAYER_CHANNELS:
            raise ValueError(f"Unsupported Feature-AE teacher backbone: {backbone!r}.")
        self.backbone = backbone
        self.layers = tuple(layers)
        unknown = sorted(set(self.layers) - set(RESNET_LAYER_CHANNELS[backbone]))
        if unknown:
            raise ValueError(f"Unsupported {backbone} teacher layers: {unknown}")

        self.model = build_resnet(backbone).eval()
        self._features: dict[str, torch.Tensor] = {}
        modules = dict(self.model.named_modules())
        for layer_name in self.layers:
            modules[layer_name].register_forward_hook(self._make_hook(layer_name))
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _make_hook(self, layer_name: str):
        def hook(_module, _inputs, output):
            self._features[layer_name] = output.detach()

        return hook

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self._features = {}
        with torch.no_grad():
            _ = self.model(images)
        return {name: self._features[name].detach() for name in self.layers}


class FeatureAEResNet18(nn.Module):
    """Small student reconstructing ResNet teacher feature maps."""

    model_type = "feature_ae_resnet18"
    teacher_backbone = "resnet18"
    stem_channels = 64
    layer1_channels = 64
    layer2_base_channels = 128
    layer3_base_channels = 256

    def __init__(self, layers: Iterable[str] = ("layer2", "layer3")):
        super().__init__()
        self.layers = tuple(layers)
        layer_channels = RESNET_LAYER_CHANNELS[self.teacher_backbone]
        unknown = sorted(set(self.layers) - set(layer_channels))
        if unknown:
            raise ValueError(f"Unsupported Feature-AE layers: {unknown}")

        c0 = int(self.stem_channels)
        c1 = int(self.layer1_channels)
        c2 = int(self.layer2_base_channels)
        c3 = int(self.layer3_base_channels)

        self.stem = nn.Sequential(conv_block(3, c0, stride=2), conv_block(c0, c0))
        self.down1 = nn.Sequential(conv_block(c0, c1, stride=2), conv_block(c1, c1))
        self.layer1_block = nn.Sequential(conv_block(c1, c1), conv_block(c1, c1))
        self.down2 = nn.Sequential(conv_block(c1, c2, stride=2), conv_block(c2, c2))
        self.layer2_head = nn.Sequential(conv_block(c2, c2), nn.Conv2d(c2, layer_channels["layer2"], 1))
        self.down3 = nn.Sequential(conv_block(c2, c3, stride=2), conv_block(c3, c3))
        self.layer3_head = nn.Sequential(conv_block(c3, c3), nn.Conv2d(c3, layer_channels["layer3"], 1))
        self.layer1_head = nn.Sequential(conv_block(c1, c1), nn.Conv2d(c1, layer_channels["layer1"], 1))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(images)
        x = self.down1(x)
        layer1_base = self.layer1_block(x)
        layer2_base = self.down2(layer1_base)
        layer3_base = self.down3(layer2_base)

        outputs = {}
        if "layer1" in self.layers:
            outputs["layer1"] = self.layer1_head(layer1_base)
        if "layer2" in self.layers:
            outputs["layer2"] = self.layer2_head(layer2_base)
        if "layer3" in self.layers:
            outputs["layer3"] = self.layer3_head(layer3_base)
        return outputs


class FeatureAEDualContextResNet18(FeatureAEResNet18):
    """Feature-AE with a local tile path modulated by a larger context tile."""

    model_type = "feature_ae_resnet18_dual_context"

    def __init__(self, layers: Iterable[str] = ("layer2", "layer3")):
        super().__init__(layers=layers)
        c0 = int(self.stem_channels)
        c1 = int(self.layer1_channels)
        c2 = int(self.layer2_base_channels)
        c3 = int(self.layer3_base_channels)
        self.context_stem = nn.Sequential(conv_block(3, c0, stride=2), conv_block(c0, c0))
        self.context_down1 = nn.Sequential(conv_block(c0, c1, stride=2), conv_block(c1, c1))
        self.context_down2 = nn.Sequential(conv_block(c1, c2, stride=2), conv_block(c2, c2))
        self.context_down3 = nn.Sequential(conv_block(c2, c3, stride=2), conv_block(c3, c3))
        self.context_film = nn.Conv2d(c3, 2 * c3, kernel_size=1)

    def encode_context(self, context_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.context_stem(context_images)
        context = self.context_down1(context)
        context = self.context_down2(context)
        context = self.context_down3(context)
        gamma_beta = self.context_film(context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return 0.25 * torch.tanh(gamma), 0.25 * torch.tanh(beta)

    def forward(self, images: torch.Tensor, context_images: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if context_images is None:
            context_images = images

        x = self.stem(images)
        x = self.down1(x)
        layer1_base = self.layer1_block(x)
        layer2_base = self.down2(layer1_base)
        layer3_base = self.down3(layer2_base)

        gamma, beta = self.encode_context(context_images)
        if gamma.shape[-2:] != layer3_base.shape[-2:]:
            gamma = F.interpolate(gamma, size=layer3_base.shape[-2:], mode="bilinear", align_corners=False)
            beta = F.interpolate(beta, size=layer3_base.shape[-2:], mode="bilinear", align_corners=False)
        layer3_base = layer3_base * (1.0 + gamma) + beta

        outputs = {}
        if "layer1" in self.layers:
            outputs["layer1"] = self.layer1_head(layer1_base)
        if "layer2" in self.layers:
            outputs["layer2"] = self.layer2_head(layer2_base)
        if "layer3" in self.layers:
            outputs["layer3"] = self.layer3_head(layer3_base)
        return outputs


class FeatureAEGatedDualContextResNet18(FeatureAEDualContextResNet18):
    """Dual-context Feature-AE with a learned gate controlling context injection."""

    model_type = "feature_ae_resnet18_dual_context_gated"

    def __init__(self, layers: Iterable[str] = ("layer2", "layer3")):
        super().__init__(layers=layers)
        c3 = int(self.layer3_base_channels)
        self.context_gate = nn.Conv2d(c3, c3, kernel_size=1)
        nn.init.zeros_(self.context_film.weight)
        nn.init.zeros_(self.context_film.bias)
        nn.init.zeros_(self.context_gate.weight)
        nn.init.constant_(self.context_gate.bias, -2.0)

    def encode_context(self, context_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.context_stem(context_images)
        context = self.context_down1(context)
        context = self.context_down2(context)
        context = self.context_down3(context)
        gamma_beta = self.context_film(context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gate = torch.sigmoid(self.context_gate(context))
        return 0.25 * torch.tanh(gamma), 0.25 * torch.tanh(beta), gate

    def forward(self, images: torch.Tensor, context_images: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if context_images is None:
            context_images = images

        x = self.stem(images)
        x = self.down1(x)
        layer1_base = self.layer1_block(x)
        layer2_base = self.down2(layer1_base)
        layer3_base = self.down3(layer2_base)

        gamma, beta, gate = self.encode_context(context_images)
        if gamma.shape[-2:] != layer3_base.shape[-2:]:
            gamma = F.interpolate(gamma, size=layer3_base.shape[-2:], mode="bilinear", align_corners=False)
            beta = F.interpolate(beta, size=layer3_base.shape[-2:], mode="bilinear", align_corners=False)
            gate = F.interpolate(gate, size=layer3_base.shape[-2:], mode="bilinear", align_corners=False)
        layer3_base = layer3_base + gate * (layer3_base * gamma + beta)

        outputs = {}
        if "layer1" in self.layers:
            outputs["layer1"] = self.layer1_head(layer1_base)
        if "layer2" in self.layers:
            outputs["layer2"] = self.layer2_head(layer2_base)
        if "layer3" in self.layers:
            outputs["layer3"] = self.layer3_head(layer3_base)
        return outputs


class ReverseDistillationResNet18(FeatureAEResNet18):
    """Multi-scale reverse-distillation student with a compressed latent bottleneck.

    Unlike the plain Feature-AE, the shallow teacher layers are reconstructed from
    the deepest compressed representation, which makes local defects harder to
    copy through the student and closer to a classic reverse-distillation setup.
    """

    model_type = "reverse_distill_resnet18"

    def __init__(self, layers: Iterable[str] = ("layer2", "layer3")):
        super().__init__(layers=layers)
        layer_channels = RESNET_LAYER_CHANNELS[self.teacher_backbone]
        c0 = int(self.stem_channels)
        c1 = int(self.layer1_channels)
        c2 = int(self.layer2_base_channels)
        c3 = int(self.layer3_base_channels)

        self.encoder_stem = nn.Sequential(conv_block(3, c0, stride=2), conv_block(c0, c0))
        self.encoder_down1 = nn.Sequential(conv_block(c0, c1, stride=2), conv_block(c1, c1))
        self.encoder_down2 = nn.Sequential(conv_block(c1, c2, stride=2), conv_block(c2, c2))
        self.encoder_down3 = nn.Sequential(conv_block(c2, c3, stride=2), conv_block(c3, c3))
        self.bottleneck = nn.Sequential(
            conv_block(c3, c3),
            nn.Conv2d(c3, c3, kernel_size=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            conv_block(c3, c3),
        )

        self.decode_layer3 = nn.Sequential(conv_block(c3, c3), nn.Conv2d(c3, layer_channels["layer3"], 1))
        self.up_layer2 = nn.Sequential(conv_block(c3, c2), conv_block(c2, c2))
        self.decode_layer2 = nn.Sequential(conv_block(c2, c2), nn.Conv2d(c2, layer_channels["layer2"], 1))
        self.up_layer1 = nn.Sequential(conv_block(c2, c1), conv_block(c1, c1))
        self.decode_layer1 = nn.Sequential(conv_block(c1, c1), nn.Conv2d(c1, layer_channels["layer1"], 1))

    def encode_local(self, images: torch.Tensor) -> torch.Tensor:
        x = self.encoder_stem(images)
        x = self.encoder_down1(x)
        x = self.encoder_down2(x)
        x = self.encoder_down3(x)
        return self.bottleneck(x)

    def decode(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}
        if "layer3" in self.layers:
            outputs["layer3"] = self.decode_layer3(latent)

        layer2_latent = F.interpolate(latent, scale_factor=2.0, mode="bilinear", align_corners=False)
        layer2_latent = self.up_layer2(layer2_latent)
        if "layer2" in self.layers:
            outputs["layer2"] = self.decode_layer2(layer2_latent)

        layer1_latent = F.interpolate(layer2_latent, scale_factor=2.0, mode="bilinear", align_corners=False)
        layer1_latent = self.up_layer1(layer1_latent)
        if "layer1" in self.layers:
            outputs["layer1"] = self.decode_layer1(layer1_latent)
        return outputs

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.decode(self.encode_local(images))


class ReverseDistillationGatedDualContextResNet18(ReverseDistillationResNet18):
    """Reverse-distillation student with gated large-context modulation."""

    model_type = "reverse_distill_resnet18_dual_context_gated"

    def __init__(self, layers: Iterable[str] = ("layer2", "layer3")):
        super().__init__(layers=layers)
        c0 = int(self.stem_channels)
        c1 = int(self.layer1_channels)
        c2 = int(self.layer2_base_channels)
        c3 = int(self.layer3_base_channels)
        self.context_stem = nn.Sequential(conv_block(3, c0, stride=2), conv_block(c0, c0))
        self.context_down1 = nn.Sequential(conv_block(c0, c1, stride=2), conv_block(c1, c1))
        self.context_down2 = nn.Sequential(conv_block(c1, c2, stride=2), conv_block(c2, c2))
        self.context_down3 = nn.Sequential(conv_block(c2, c3, stride=2), conv_block(c3, c3))
        self.context_film = nn.Conv2d(c3, 2 * c3, kernel_size=1)
        self.context_gate = nn.Conv2d(c3, c3, kernel_size=1)
        nn.init.zeros_(self.context_film.weight)
        nn.init.zeros_(self.context_film.bias)
        nn.init.zeros_(self.context_gate.weight)
        nn.init.constant_(self.context_gate.bias, -2.0)

    def encode_context(self, context_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.context_stem(context_images)
        context = self.context_down1(context)
        context = self.context_down2(context)
        context = self.context_down3(context)
        gamma_beta = self.context_film(context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gate = torch.sigmoid(self.context_gate(context))
        return 0.25 * torch.tanh(gamma), 0.25 * torch.tanh(beta), gate

    def forward(self, images: torch.Tensor, context_images: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if context_images is None:
            context_images = images
        latent = self.encode_local(images)
        gamma, beta, gate = self.encode_context(context_images)
        if gamma.shape[-2:] != latent.shape[-2:]:
            gamma = F.interpolate(gamma, size=latent.shape[-2:], mode="bilinear", align_corners=False)
            beta = F.interpolate(beta, size=latent.shape[-2:], mode="bilinear", align_corners=False)
            gate = F.interpolate(gate, size=latent.shape[-2:], mode="bilinear", align_corners=False)
        latent = latent + gate * (latent * gamma + beta)
        return self.decode(latent)


def build_feature_autoencoder(model_type: str, layers: Iterable[str]) -> FeatureAEResNet18:
    if model_type == "feature_ae_resnet18":
        return FeatureAEResNet18(layers=layers)
    if model_type == "feature_ae_resnet18_dual_context":
        return FeatureAEDualContextResNet18(layers=layers)
    if model_type == "feature_ae_resnet18_dual_context_gated":
        return FeatureAEGatedDualContextResNet18(layers=layers)
    if model_type == "reverse_distill_resnet18":
        return ReverseDistillationResNet18(layers=layers)
    if model_type == "reverse_distill_resnet18_dual_context_gated":
        return ReverseDistillationGatedDualContextResNet18(layers=layers)
    raise ValueError(
        "Unknown feature AE model_type. Use a ResNet18 Feature-AE or Reverse-Distillation model."
    )


def feature_reconstruction_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    cosine_weight: float,
    pixel_weight: torch.Tensor | None = None,
    layer_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weighted_losses = []
    total_weight = 0.0
    metrics: dict[str, float] = {}
    for layer_name, target in targets.items():
        prediction = predictions[layer_name]
        if prediction.shape[-2:] != target.shape[-2:]:
            prediction = F.interpolate(
                prediction,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        l2_map = torch.mean((prediction - target).pow(2), dim=1)
        cosine_map = 1.0 - F.cosine_similarity(prediction, target, dim=1)
        if pixel_weight is not None:
            weight = F.interpolate(
                pixel_weight.float(),
                size=target.shape[-2:],
                mode="nearest",
            )[:, 0].clamp_min(0.0)
            denom = weight.sum().clamp_min(1e-6)
            l2 = (l2_map * weight).sum() / denom
            cosine = (cosine_map * weight).sum() / denom
        else:
            l2 = l2_map.mean()
            cosine = cosine_map.mean()
        layer_loss = l2 + float(cosine_weight) * cosine
        layer_weight = float((layer_weights or {}).get(layer_name, 1.0))
        if layer_weight <= 0:
            continue
        weighted_losses.append(layer_loss * layer_weight)
        total_weight += layer_weight
        metrics[f"{layer_name}_l2"] = float(l2.detach().cpu())
        metrics[f"{layer_name}_cosine"] = float(cosine.detach().cpu())
        metrics[f"{layer_name}_weight"] = layer_weight
    if not weighted_losses:
        raise ValueError("At least one selected layer must have a positive loss weight.")
    total = torch.stack(weighted_losses).sum() / max(total_weight, 1e-6)
    metrics["loss"] = float(total.detach().cpu())
    return total, metrics


def feature_error_map(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_weight: float,
) -> torch.Tensor:
    if prediction.shape[-2:] != target.shape[-2:]:
        prediction = F.interpolate(
            prediction,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    l2 = torch.sqrt(torch.mean((prediction - target).pow(2), dim=1).clamp_min(1e-12))
    cosine = 1.0 - F.cosine_similarity(prediction, target, dim=1)
    return l2 + float(cosine_weight) * cosine






