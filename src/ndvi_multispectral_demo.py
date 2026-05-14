import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rgb(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.array(image).astype(np.float32)


def load_nir(nir_path: str, target_shape: tuple) -> np.ndarray:
    """
    Load an optional NIR image as grayscale and resize it to RGB image size.
    """
    nir_image = Image.open(nir_path).convert("L")
    nir_array = np.array(nir_image).astype(np.float32)

    height, width = target_shape[:2]
    nir_resized = cv2.resize(nir_array, (width, height))
    return nir_resized


def estimate_pseudo_nir_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    Estimate pseudo-NIR from RGB when real NIR is unavailable.

    This is not real NDVI. It is a portfolio demo to show the processing logic.
    Real NDVI requires a real NIR band.
    """
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    pseudo_nir = (0.55 * green + 0.25 * red + 0.20 * blue)
    return pseudo_nir


def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    NDVI = (NIR - Red) / (NIR + Red)
    """
    denominator = nir + red
    denominator[denominator == 0] = 1e-6

    ndvi = (nir - red) / denominator
    ndvi = np.clip(ndvi, -1, 1)

    return ndvi


def classify_ndvi(ndvi: np.ndarray) -> dict:
    """
    Basic NDVI class percentages.
    """
    water_or_low = ndvi < 0.0
    bare_soil = (ndvi >= 0.0) & (ndvi < 0.2)
    sparse_vegetation = (ndvi >= 0.2) & (ndvi < 0.4)
    healthy_vegetation = ndvi >= 0.4

    total_pixels = ndvi.size

    return {
        "water_or_very_low_index_percentage": round(float(np.sum(water_or_low) / total_pixels * 100), 2),
        "bare_soil_or_low_vegetation_percentage": round(float(np.sum(bare_soil) / total_pixels * 100), 2),
        "sparse_vegetation_percentage": round(float(np.sum(sparse_vegetation) / total_pixels * 100), 2),
        "healthy_vegetation_percentage": round(float(np.sum(healthy_vegetation) / total_pixels * 100), 2),
        "mean_ndvi": round(float(np.mean(ndvi)), 4),
    }


def save_ndvi_visualization(ndvi: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(label="NDVI")
    plt.title("NDVI / Vegetation Index Map")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="NDVI / multispectral vegetation index demo.")
    parser.add_argument("--image", required=True, help="Path to RGB image.")
    parser.add_argument("--nir", required=False, help="Optional path to NIR band image.")
    args = parser.parse_args()

    rgb = load_rgb(args.image)
    red = rgb[:, :, 0]

    if args.nir:
        nir = load_nir(args.nir, rgb.shape)
        mode = "real_nir_band"
    else:
        nir = estimate_pseudo_nir_from_rgb(rgb)
        mode = "pseudo_nir_from_rgb_demo"

    ndvi = calculate_ndvi(red=red, nir=nir)
    indicators = classify_ndvi(ndvi)

    output_path = OUTPUT_DIR / "ndvi_vegetation_index.png"
    save_ndvi_visualization(ndvi, output_path)

    print("\nNDVI / Multispectral Vegetation Index Demo")
    print("=" * 50)
    print(f"Mode: {mode}")

    for key, value in indicators.items():
        print(f"- {key}: {value}")

    print(f"\nSaved NDVI visualization: {output_path}")

    print("\nImportant clarification:")
    print(
        "If no NIR band is provided, this script uses a pseudo-NIR estimate from RGB. "
        "This demonstrates the pipeline logic, but real NDVI requires multispectral data "
        "with a real NIR channel."
    )


if __name__ == "__main__":
    main()