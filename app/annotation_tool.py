import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = ANNOTATION_DIR / "annotations.csv"
JSON_PATH = ANNOTATION_DIR / "annotations_demo.json"

CLASSES = [
    "vegetation_cover",
    "bare_soil",
    "waterlogged_soil",
    "degraded_area",
    "overgrazed_area",
    "suitable_grazing_area",
]


def ensure_files() -> None:
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "timestamp",
                    "image_name",
                    "source_type",
                    "grazing_area_id",
                    "latitude",
                    "longitude",
                    "annotation_type",
                    "class_label",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "review_status",
                    "notes",
                ]
            )

    if not JSON_PATH.exists():
        with open(JSON_PATH, "w", encoding="utf-8") as file:
            json.dump({"annotations": []}, file, indent=2)


def draw_bbox(image: Image.Image, x1, y1, x2, y2, label: str) -> np.ndarray:
    image_rgb = np.array(image.convert("RGB"))
    annotated = image_rgb.copy()

    try:
        x1 = int(float(x1))
        y1 = int(float(y1))
        x2 = int(float(x2))
        y2 = int(float(y2))
    except Exception:
        return annotated

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        annotated,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    return annotated


def preview_annotation(image, class_label, x1, y1, x2, y2):
    if image is None:
        return None, "Please upload an image first."

    annotated = draw_bbox(image, x1, y1, x2, y2, class_label)
    return annotated, "Preview generated."


def save_annotation(
    image,
    image_name,
    source_type,
    grazing_area_id,
    latitude,
    longitude,
    annotation_type,
    class_label,
    x1,
    y1,
    x2,
    y2,
    review_status,
    notes,
):
    ensure_files()

    if image is None:
        return "Please upload an image first."

    if not image_name:
        image_name = f"annotated_image_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.png"

    timestamp = datetime.now(timezone.utc).isoformat()

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                timestamp,
                image_name,
                source_type,
                grazing_area_id,
                latitude,
                longitude,
                annotation_type,
                class_label,
                x1,
                y1,
                x2,
                y2,
                review_status,
                notes,
            ]
        )

    with open(JSON_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    data["annotations"].append(
        {
            "timestamp": timestamp,
            "image_name": image_name,
            "source_type": source_type,
            "grazing_area_id": grazing_area_id,
            "latitude": latitude,
            "longitude": longitude,
            "annotation_type": annotation_type,
            "class_label": class_label,
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "review_status": review_status,
            "notes": notes,
        }
    )

    with open(JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    return f"Annotation saved to {CSV_PATH} and {JSON_PATH}"


ensure_files()

with gr.Blocks(title="DFence Image Annotation Tool Demo") as demo:
    gr.Markdown(
        """
        # DFence Image Annotation Tool Demo

        This is a simple self-service annotation tool for pasture and soil images.

        It supports:
        - image upload
        - source metadata
        - pasture/soil class selection
        - manual bounding-box coordinate entry
        - annotation preview
        - CSV and JSON export

        This is a portfolio demo, not a full production annotation platform.
        """
    )

    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Upload image")
            image_name = gr.Textbox(label="Image name", value="sample_pasture_image.jpg")

            source_type = gr.Dropdown(
                choices=["drone", "satellite", "rover", "animal_camera", "manual_upload"],
                value="manual_upload",
                label="Source type",
            )

            grazing_area_id = gr.Textbox(label="Grazing area ID", value="area_001")
            latitude = gr.Textbox(label="Latitude", value="41.1579")
            longitude = gr.Textbox(label="Longitude", value="-8.6291")

            annotation_type = gr.Dropdown(
                choices=["image_classification", "object_detection_bbox", "segmentation_mask_future"],
                value="object_detection_bbox",
                label="Annotation type",
            )

            class_label = gr.Dropdown(
                choices=CLASSES,
                value="suitable_grazing_area",
                label="Class label",
            )

            gr.Markdown("### Bounding Box Coordinates")
            x1 = gr.Number(label="x1", value=50)
            y1 = gr.Number(label="y1", value=50)
            x2 = gr.Number(label="x2", value=300)
            y2 = gr.Number(label="y2", value=250)

            review_status = gr.Dropdown(
                choices=["draft", "review_required", "approved"],
                value="draft",
                label="Review status",
            )

            notes = gr.Textbox(label="Notes", lines=3)

            preview_button = gr.Button("Preview Annotation")
            save_button = gr.Button("Save Annotation")
            status = gr.Textbox(label="Status", lines=2)

        with gr.Column():
            output_image = gr.Image(label="Annotation Preview")

    preview_button.click(
        fn=preview_annotation,
        inputs=[image, class_label, x1, y1, x2, y2],
        outputs=[output_image, status],
    )

    save_button.click(
        fn=save_annotation,
        inputs=[
            image,
            image_name,
            source_type,
            grazing_area_id,
            latitude,
            longitude,
            annotation_type,
            class_label,
            x1,
            y1,
            x2,
            y2,
            review_status,
            notes,
        ],
        outputs=[status],
    )


if __name__ == "__main__":
    demo.launch()