import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
except Exception:
    DeepLabV3_ResNet50_Weights = None
    deeplabv3_resnet50 = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rgb_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def pasture_soil_heuristic_segmentation(image_rgb: np.ndarray) -> dict:
    """
    Prototype pasture/soil segmentation using simple HSV thresholds.

    Classes:
    0 = background/unknown
    1 = vegetation_cover
    2 = bare_soil
    3 = waterlogged_soil
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    vegetation_mask = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
    bare_soil_mask = cv2.inRange(hsv, (5, 30, 30), (30, 255, 220))
    waterlogged_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 70))

    segmentation = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    segmentation[vegetation_mask > 0] = 1
    segmentation[bare_soil_mask > 0] = 2
    segmentation[waterlogged_mask > 0] = 3

    total_pixels = segmentation.size

    indicators = {
        "vegetation_cover_percentage": round(float(np.sum(segmentation == 1) / total_pixels * 100), 2),
        "bare_soil_percentage": round(float(np.sum(segmentation == 2) / total_pixels * 100), 2),
        "waterlogged_soil_percentage": round(float(np.sum(segmentation == 3) / total_pixels * 100), 2),
        "unknown_percentage": round(float(np.sum(segmentation == 0) / total_pixels * 100), 2),
    }

    return {
        "segmentation": segmentation,
        "indicators": indicators,
    }


def create_mask_overlay(image_rgb: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
    """
    Create colored overlay for segmentation classes.
    """
    color_mask = np.zeros_like(image_rgb)

    # vegetation = green
    color_mask[segmentation == 1] = [0, 180, 0]

    # bare soil = brown/orange
    color_mask[segmentation == 2] = [180, 100, 40]

    # waterlogged = blue
    color_mask[segmentation == 3] = [40, 90, 200]

    overlay = cv2.addWeighted(image_rgb, 0.65, color_mask, 0.35, 0)
    return overlay


def run_deeplabv3_demo(image_rgb: np.ndarray) -> dict:
    """
    CPU-based DeepLabV3 semantic segmentation demo.

    This uses pretrained DeepLabV3 if weights are available.
    It is not trained for pasture/soil classes; it demonstrates semantic
    segmentation capability. The pasture/soil estimate is handled separately
    by the heuristic segmentation.
    """
    if deeplabv3_resnet50 is None or DeepLabV3_ResNet50_Weights is None:
        return {
            "available": False,
            "message": "torchvision DeepLabV3 is not available.",
        }

    try:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights)
        model.eval()
        model.to("cpu")

        preprocess = weights.transforms()
        input_image = Image.fromarray(image_rgb)
        batch = preprocess(input_image).unsqueeze(0).to("cpu")

        with torch.no_grad():
            output = model(batch)["out"][0]

        predicted_mask = output.argmax(0).byte().cpu().numpy()
        categories = weights.meta.get("categories", [])

        unique_classes = sorted(np.unique(predicted_mask).tolist())
        detected_classes = []

        for class_id in unique_classes:
            if class_id < len(categories):
                class_name = categories[class_id]
            else:
                class_name = str(class_id)

            if class_name != "__background__":
                detected_classes.append(class_name)

        return {
            "available": True,
            "detected_classes": detected_classes,
            "mask": predicted_mask,
        }

    except Exception as error:
        return {
            "available": False,
            "message": f"DeepLabV3 could not run: {error}",
        }


def main():
    parser = argparse.ArgumentParser(description="Semantic segmentation demo for pasture/soil monitoring.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    args = parser.parse_args()

    image_rgb = load_rgb_image(args.image)

    heuristic_result = pasture_soil_heuristic_segmentation(image_rgb)
    segmentation = heuristic_result["segmentation"]
    indicators = heuristic_result["indicators"]
    overlay = create_mask_overlay(image_rgb, segmentation)

    output_overlay_path = OUTPUT_DIR / "semantic_segmentation_overlay.png"
    output_mask_path = OUTPUT_DIR / "semantic_segmentation_mask.png"

    cv2.imwrite(str(output_overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_mask_path), segmentation)

    deeplab_result = run_deeplabv3_demo(image_rgb)

    print("\nSemantic Segmentation Demo")
    print("=" * 40)
    print("Pasture/soil heuristic segmentation indicators:")
    for key, value in indicators.items():
        print(f"- {key}: {value}%")

    print("\nDeepLabV3 demo result:")
    if deeplab_result["available"]:
        print(f"- Detected pretrained segmentation classes: {deeplab_result['detected_classes']}")
    else:
        print(f"- {deeplab_result['message']}")

    print("\nSaved outputs:")
    print(f"- {output_overlay_path}")
    print(f"- {output_mask_path}")

    print("\nImportant clarification:")
    print(
        "DeepLabV3 is used here as a semantic segmentation capability demo. "
        "For real DFence pasture/soil segmentation, a custom annotated dataset "
        "and custom model training would be required."
    )


if __name__ == "__main__":
    main()