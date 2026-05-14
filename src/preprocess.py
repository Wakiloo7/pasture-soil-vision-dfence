from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_supported_image(file_path: str) -> bool:
    """Check whether an image file has a supported extension."""
    return Path(file_path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def load_image_as_rgb(image_path: str) -> np.ndarray:
    """Load an image from disk and convert it to RGB."""
    if not is_supported_image(image_path):
        raise ValueError(f"Unsupported image format: {image_path}")

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an RGB NumPy array."""
    return np.array(image.convert("RGB"))


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """Resize an image for model inference or preprocessing."""
    return cv2.resize(image, target_size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to the range [0, 1]."""
    return image.astype(np.float32) / 255.0


def check_image_quality(image: np.ndarray) -> dict:
    """
    Run basic image quality checks.

    Returns blur score, brightness, and simple quality flags.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))

    return {
        "blur_score": round(float(blur_score), 2),
        "brightness": round(brightness, 2),
        "is_blurry": bool(blur_score < 50),
        "is_too_dark": bool(brightness < 40),
        "is_too_bright": bool(brightness > 220),
    }


def save_rgb_image(image: np.ndarray, output_path: str) -> None:
    """Save an RGB image to disk."""
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path_obj), image_bgr)