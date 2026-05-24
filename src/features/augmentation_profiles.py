"""Category-aware augmentation profiles for reconstruction training."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AugmentationProfile:
    name: str
    crop_profile: str
    scale: tuple[float, float]
    ratio: tuple[float, float]
    horizontal_flip_p: float
    vertical_flip_p: float
    rotation_degrees: float
    brightness: float
    contrast: float
    saturation: float
    hue: float
    blur_p: float
    repeat_factor: int
    notes: str
    flat_lighting_p: float = 0.0
    flat_lighting_contrast: tuple[float, float] = (1.0, 1.0)
    flat_lighting_lift: tuple[float, float] = (0.0, 0.0)
    workspace_size: int | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["scale"] = list(self.scale)
        data["ratio"] = list(self.ratio)
        data["flat_lighting_contrast"] = list(self.flat_lighting_contrast)
        data["flat_lighting_lift"] = list(self.flat_lighting_lift)
        return data


NONE_PROFILE = AugmentationProfile(
    name="none",
    crop_profile="crop_none",
    scale=(1.0, 1.0),
    ratio=(1.0, 1.0),
    horizontal_flip_p=0.0,
    vertical_flip_p=0.0,
    rotation_degrees=0.0,
    brightness=0.0,
    contrast=0.0,
    saturation=0.0,
    hue=0.0,
    blur_p=0.0,
    repeat_factor=1,
    notes="No category-specific augmentation.",
)


DEFAULT_PROFILE = AugmentationProfile(
    name="default",
    crop_profile="crop_soft",
    scale=(0.85, 1.0),
    ratio=(0.95, 1.05),
    horizontal_flip_p=0.0,
    vertical_flip_p=0.0,
    rotation_degrees=0.0,
    brightness=0.03,
    contrast=0.03,
    saturation=0.0,
    hue=0.0,
    blur_p=0.0,
    repeat_factor=1,
    notes="Conservative generic profile for categories without a dedicated policy.",
)


TOOTHBRUSH_PROFILE = AugmentationProfile(
    name="toothbrush",
    crop_profile="crop_soft",
    scale=(0.85, 1.0),
    ratio=(0.95, 1.05),
    horizontal_flip_p=0.0,
    vertical_flip_p=0.5,
    rotation_degrees=0.0,
    brightness=0.03,
    contrast=0.03,
    saturation=0.0,
    hue=0.0,
    blur_p=0.0,
    repeat_factor=8,
    notes=(
        "Toothbrush is a tiny MVTec category. Preserve the full object, use soft "
        "random crops and vertical flips only, avoid blur and strong color changes."
    ),
)


TOOTHBRUSH_HEADPRIOR_PROFILE = AugmentationProfile(
    name="toothbrush_headprior",
    crop_profile="crop_headprior_conservative",
    scale=(0.90, 1.0),
    ratio=(0.98, 1.02),
    horizontal_flip_p=0.0,
    vertical_flip_p=0.5,
    rotation_degrees=0.0,
    brightness=0.04,
    contrast=0.04,
    saturation=0.0,
    hue=0.0,
    blur_p=0.0,
    repeat_factor=12,
    notes=(
        "Toothbrush V2 profile. Keep a near-native square view of the object, "
        "use vertical flips only, and pair training with head-prior masking so "
        "the model learns normal bristle structure instead of copying."
    ),
)


CASTING_MICRODEFECT_PROFILE = AugmentationProfile(
    name="casting_microdefect",
    crop_profile="crop_tile_soft",
    scale=(0.90, 1.0),
    ratio=(0.95, 1.05),
    horizontal_flip_p=0.5,
    vertical_flip_p=0.5,
    rotation_degrees=0.0,
    brightness=0.03,
    contrast=0.03,
    saturation=0.0,
    hue=0.0,
    blur_p=0.0,
    repeat_factor=1,
    notes=(
        "Casting micro-defect profile. Use native tiles, conservative random "
        "crops, horizontal/vertical flips, and very light brightness/contrast. "
        "Avoid blur and arbitrary rotations so tiny cavities remain sharp."
    ),
)


FUNCTIONAL_SURFACE_LIGHTING_PROFILE = AugmentationProfile(
    name="functional_surface_lighting",
    crop_profile="crop_lighting_geometry",
    scale=(0.90, 1.0),
    ratio=(0.95, 1.05),
    horizontal_flip_p=0.5,
    vertical_flip_p=0.5,
    rotation_degrees=15.0,
    brightness=0.15,
    contrast=0.12,
    saturation=0.08,
    hue=0.0,
    blur_p=0.0,
    repeat_factor=4,
    notes=(
        "Functional-surface crop profile. Crops are already materialized on disk, "
        "so use synchronized light geometry: small crop jitter, flips, modest rotations, "
        "and lighting/color variation. Image and masks are transformed together."
    ),
)


FUNCTIONAL_SURFACE_LIGHTING_256_PROFILE = AugmentationProfile(
    name="functional_surface_lighting_256",
    crop_profile="crop_lighting_geometry_256",
    scale=(0.92, 1.0),
    ratio=(0.97, 1.03),
    horizontal_flip_p=0.5,
    vertical_flip_p=0.5,
    rotation_degrees=8.0,
    brightness=0.10,
    contrast=0.08,
    saturation=0.04,
    hue=0.0,
    blur_p=0.0,
    repeat_factor=3,
    notes=(
        "Stable 256px functional-surface profile. Use moderate synchronized "
        "geometry and lighting variation so larger context crops keep readable "
        "holes, edges, and machined-surface boundaries."
    ),
)


FUNCTIONAL_SURFACE_FULL_SOURCE_256_PROFILE = AugmentationProfile(
    name="functional_surface_full_source_256",
    crop_profile="full_source_runtime_crops_256",
    scale=(0.16, 0.50),
    ratio=(0.80, 1.25),
    horizontal_flip_p=0.5,
    vertical_flip_p=0.5,
    rotation_degrees=6.0,
    brightness=0.10,
    contrast=0.08,
    saturation=0.04,
    hue=0.0,
    blur_p=0.0,
    repeat_factor=12,
    notes=(
        "Runtime crop profile for full-image functional-surface masks. Keep only "
        "full source images on disk, then sample synchronized crops during "
        "training. Area scale 0.16-0.50 roughly covers medium context crops "
        "without forcing the model to see the whole part every step."
    ),
)


FUNCTIONAL_SURFACE_FULL_SOURCE_256_FLAT_LIGHTING_PROFILE = AugmentationProfile(
    name="functional_surface_full_source_256_flat_lighting",
    crop_profile="full_source_runtime_crops_256",
    scale=(0.16, 0.50),
    ratio=(0.80, 1.25),
    horizontal_flip_p=0.5,
    vertical_flip_p=0.5,
    rotation_degrees=6.0,
    brightness=0.06,
    contrast=0.04,
    saturation=0.02,
    hue=0.0,
    blur_p=0.0,
    repeat_factor=12,
    flat_lighting_p=0.35,
    flat_lighting_contrast=(0.45, 0.72),
    flat_lighting_lift=(0.10, 0.22),
    notes=(
        "Runtime full-source profile plus low-dynamic-range lighting simulation. "
        "This targets problematic Casting_class1 test images where blacks are lifted, "
        "contrast is compressed, and machined surfaces are hard to separate from "
        "painted/rough regions. The flat-lighting transform is image-only and keeps "
        "all masks synchronized."
    ),
)


FUNCTIONAL_SURFACE_FULL_SOURCE_512_TO_256_LARGE_CONTEXT_PROFILE = AugmentationProfile(
    name="functional_surface_full_source_512_to_256_large_context",
    crop_profile="full_source_runtime_crops_512_to_256",
    scale=(0.35, 0.90),
    ratio=(0.75, 1.35),
    horizontal_flip_p=0.5,
    vertical_flip_p=0.5,
    rotation_degrees=6.0,
    brightness=0.08,
    contrast=0.06,
    saturation=0.03,
    hue=0.0,
    blur_p=0.0,
    repeat_factor=8,
    workspace_size=512,
    notes=(
        "Large-context functional-surface profile. Full images and masks are "
        "letterboxed to a 512px workspace, then random crops are resized back to "
        "the model input size. With input_size=256, each crop can cover roughly "
        "2x more source context than the standard 256 workspace while preserving "
        "the existing context2b model size and memory footprint."
    ),
)


FUNCTIONAL_SURFACE_FULL_SOURCE_768_TO_384_DENOISE_PROFILE = AugmentationProfile(
    name="functional_surface_full_source_768_to_384_denoise",
    crop_profile="full_source_runtime_crops_768_to_384_denoise",
    scale=(0.35, 0.92),
    ratio=(0.75, 1.35),
    horizontal_flip_p=0.5,
    vertical_flip_p=0.5,
    rotation_degrees=6.0,
    brightness=0.09,
    contrast=0.07,
    saturation=0.03,
    hue=0.0,
    blur_p=0.03,
    repeat_factor=8,
    workspace_size=768,
    notes=(
        "V23 high-context multiclass profile. Full images and masks are "
        "letterboxed to a 768px workspace, then synchronized crops are resized "
        "to 384px locally. Pair with --context-size 768 and synthetic defect "
        "denoising so the model learns normal surface/landmark geometry instead "
        "of absorbing defect-like local dark blobs into landmarks."
    ),
)


PROFILES = {
    "none": NONE_PROFILE,
    "default": DEFAULT_PROFILE,
    "toothbrush": TOOTHBRUSH_PROFILE,
    "toothbrush_headprior": TOOTHBRUSH_HEADPRIOR_PROFILE,
    "casting_microdefect": CASTING_MICRODEFECT_PROFILE,
    "functional_surface_lighting": FUNCTIONAL_SURFACE_LIGHTING_PROFILE,
    "functional_surface_lighting_256": FUNCTIONAL_SURFACE_LIGHTING_256_PROFILE,
    "functional_surface_full_source_256": FUNCTIONAL_SURFACE_FULL_SOURCE_256_PROFILE,
    "functional_surface_full_source_256_flat_lighting": FUNCTIONAL_SURFACE_FULL_SOURCE_256_FLAT_LIGHTING_PROFILE,
    "functional_surface_full_source_512_to_256_large_context": FUNCTIONAL_SURFACE_FULL_SOURCE_512_TO_256_LARGE_CONTEXT_PROFILE,
    "functional_surface_full_source_768_to_384_denoise": FUNCTIONAL_SURFACE_FULL_SOURCE_768_TO_384_DENOISE_PROFILE,
}


CATEGORY_TO_PROFILE = {
    "toothbrush": "toothbrush",
    "Casting_class1": "casting_microdefect",
    "Casting_class2": "casting_microdefect",
    "Casting_class3": "casting_microdefect",
}


def resolve_augmentation_profile(name: str, category: str | None) -> AugmentationProfile:
    """Resolve a CLI profile name, including the category-aware ``auto`` alias."""

    if name == "auto":
        name = CATEGORY_TO_PROFILE.get(str(category), "default")
    try:
        return PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted([*PROFILES.keys(), "auto"]))
        raise ValueError(f"Unknown augmentation profile {name!r}. Expected one of: {known}") from exc





