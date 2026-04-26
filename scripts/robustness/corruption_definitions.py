"""
Shared corruption parameters and helpers for robustness (step1) and Figure 4.

Single source of truth: CORRUPTIONS matches what step1_create_corrupted_test_sets.py applies.

References:
- Hendrycks, D. and Dietterich, T. (2019) Common Corruptions benchmark
  (motivation for corruption-style robustness evaluation).
- Di Salvo, F., Doerrich, S. and Ledig, C. (2024) MedMNIST-C
  (medical adaptation of controlled corruption robustness testing).
"""

from __future__ import annotations

# In-memory byte buffer for JPEG re-encode simulation.
import io
from typing import Optional

# NumPy is used for pixel-array math and Gaussian noise sampling.
import numpy as np
# PIL provides blur/filter/enhancement and image I/O utilities.
from PIL import Image, ImageEnhance, ImageFilter

# Reference (single source of truth):
# These values define corruption strength for both robustness scripts and Figure 4.
# Keep folder naming and these keys aligned (e.g., blur_mild, jpeg_strong).
CORRUPTIONS = {
    "blur": {
        "mild": {"radius": 1.5},
        "medium": {"radius": 3.0},
        "strong": {"radius": 5.0},
    },
    "brightness": {
        "mild": {"factor": 0.85},
        "medium": {"factor": 0.65},
        "strong": {"factor": 0.45},
    },
    "contrast": {
        "mild": {"factor": 0.85},
        "medium": {"factor": 0.60},
        "strong": {"factor": 0.40},
    },
    "noise": {
        "mild": {"std": 15},
        "medium": {"std": 35},
        "strong": {"std": 60},
    },
    "jpeg": {
        "mild": {"quality": 75},
        "medium": {"quality": 50},
        "strong": {"quality": 25},
    },
}


def corrupt_blur(img: Image.Image, radius: float) -> Image.Image:
    # Reference: Gaussian blur as a standard defocus corruption.
    # (Common Corruptions style perturbation)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def corrupt_brightness(img: Image.Image, factor: float) -> Image.Image:
    # Scale image luminance: <1.0 darker, >1.0 brighter.
    return ImageEnhance.Brightness(img).enhance(factor)


def corrupt_contrast(img: Image.Image, factor: float) -> Image.Image:
    # Scale image contrast: <1.0 flatter contrast, >1.0 stronger contrast.
    return ImageEnhance.Contrast(img).enhance(factor)


def corrupt_noise(
    img: Image.Image, std: float, *, seed: Optional[int] = None
) -> Image.Image:
    """Additive Gaussian noise. If seed is set, draws are reproducible (e.g. Figure 4)."""
    # Convert image to float array so additive noise is applied safely.
    arr = np.array(img, dtype=np.float64)
    # If seed is provided, use a local RNG for deterministic noise fields.
    if seed is not None:
        noise = np.random.default_rng(seed).normal(0.0, std, arr.shape)
    else:
        # Otherwise use global RNG state (suitable for batch corruption generation).
        noise = np.random.normal(0, std, arr.shape)
    # Clip back to valid 8-bit range and convert to uint8 image format.
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def corrupt_jpeg(img: Image.Image, quality: int) -> Image.Image:
    # Reference: JPEG compression artifact simulation by re-encoding at chosen quality.
    # Lower quality introduces stronger blocking/ringing artifacts.
    buf = io.BytesIO()
    # Encode image to JPEG in-memory at requested quality.
    img.save(buf, format="JPEG", quality=quality)
    # Rewind buffer to start so PIL can read the encoded bytes.
    buf.seek(0)
    # Decode back to image and preserve original mode (e.g., RGB).
    return Image.open(buf).convert(img.mode)
