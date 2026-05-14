import csv
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import gradio as gr
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

from predict import PASTURE_SOIL_CLASSES, PastureSoilPredictor
from preprocess import check_image_quality, pil_to_rgb_array


FEEDBACK_FILE = PROJECT_ROOT / "data" / "feedback_records.csv"
INDICATOR_FILE = PROJECT_ROOT / "data" / "georeferenced_indicators.csv"

CUSTOM_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
PRETRAINED_MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"

predictor = PastureSoilPredictor(
    custom_model_path=str(CUSTOM_MODEL_PATH),
    pretrained_model_path=str(PRETRAINED_MODEL_PATH),
)


def ensure_output_files() -> None:
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "timestamp",
                    "source_type",
                    "grazing_area_id",
                    "latitude",
                    "longitude",
                    "top_yolo_detection",
                    "top_yolo_confidence",
                    "pasture_soil_condition",
                    "pasture_soil_confidence",
                    "corrected_label",
                    "user_comment",
                    "model_mode",
                    "quality_json",
                    "prediction_json",
                ]
            )

    if not INDICATOR_FILE.exists():
        with open(INDICATOR_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "timestamp",
                    "source_type",
                    "grazing_area_id",
                    "latitude",
                    "longitude",
                    "vegetation_cover_score",
                    "bare_soil_score",
                    "waterlogged_soil_score",
                    "degraded_area_score",
                    "overgrazed_area_score",
                    "suitable_grazing_area_score",
                    "condition_estimate",
                    "condition_confidence",
                    "model_mode",
                ]
            )


def build_indicator_json(
    prediction: dict,
    source_type: str,
    grazing_area_id: str,
    latitude: str,
    longitude: str,
) -> dict:
    scores = prediction["pasture_soil_scores"]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_type": source_type,
        "grazing_area_id": grazing_area_id,
        "latitude": latitude,
        "longitude": longitude,
        "vegetation_cover_score": scores.get("vegetation_cover"),
        "bare_soil_score": scores.get("bare_soil"),
        "waterlogged_soil_score": scores.get("waterlogged_soil"),
        "degraded_area_score": scores.get("degraded_area"),
        "overgrazed_area_score": scores.get("overgrazed_area"),
        "suitable_grazing_area_score": scores.get("suitable_grazing_area"),
        "condition_estimate": prediction["pasture_soil_condition"],
        "condition_confidence": prediction["pasture_soil_confidence"],
        "model_mode": prediction["mode"],
    }


def save_indicator(indicator: dict) -> None:
    with open(INDICATOR_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                indicator["timestamp"],
                indicator["source_type"],
                indicator["grazing_area_id"],
                indicator["latitude"],
                indicator["longitude"],
                indicator["vegetation_cover_score"],
                indicator["bare_soil_score"],
                indicator["waterlogged_soil_score"],
                indicator["degraded_area_score"],
                indicator["overgrazed_area_score"],
                indicator["suitable_grazing_area_score"],
                indicator["condition_estimate"],
                indicator["condition_confidence"],
                indicator["model_mode"],
            ]
        )


def analyze_image(
    image: Image.Image,
    source_type: str,
    grazing_area_id: str,
    latitude: str,
    longitude: str,
):
    if image is None:
        return None, "Please upload an image first.", "", "", "", ""

    image_rgb = pil_to_rgb_array(image)
    quality = check_image_quality(image_rgb)
    prediction = predictor.predict(image)

    indicator = build_indicator_json(
        prediction=prediction,
        source_type=source_type,
        grazing_area_id=grazing_area_id,
        latitude=latitude,
        longitude=longitude,
    )

    save_indicator(indicator)

    summary = (
        f"Top YOLO detection: {prediction['top_yolo_detection']}\n"
        f"YOLO confidence: {prediction['top_yolo_confidence']}\n"
        f"Pasture/soil condition estimate: {prediction['pasture_soil_condition']}\n"
        f"Pasture/soil confidence: {prediction['pasture_soil_confidence']}\n"
        f"Mode: {prediction['mode']}\n\n"
        f"Georeferenced indicator:\n"
        f"- Source type: {source_type}\n"
        f"- Grazing area ID: {grazing_area_id}\n"
        f"- Latitude: {latitude}\n"
        f"- Longitude: {longitude}\n"
        f"- Vegetation score: {indicator['vegetation_cover_score']}\n"
        f"- Bare soil score: {indicator['bare_soil_score']}\n"
        f"- Waterlogged score: {indicator['waterlogged_soil_score']}\n"
        f"- Degraded score: {indicator['degraded_area_score']}\n"
        f"- Overgrazed score: {indicator['overgrazed_area_score']}\n"
        f"- Suitable grazing score: {indicator['suitable_grazing_area_score']}\n\n"
        f"Image quality checks:\n"
        f"- Blur score: {quality['blur_score']}\n"
        f"- Brightness: {quality['brightness']}\n"
        f"- Is blurry: {quality['is_blurry']}\n"
        f"- Is too dark: {quality['is_too_dark']}\n"
        f"- Is too bright: {quality['is_too_bright']}\n\n"
        f"Note: Pretrained YOLO detects COCO objects such as cows or people. "
        f"Pasture/soil classes are estimated separately until a custom dataset is trained."
    )

    prediction_for_json = {
    key: value
    for key, value in prediction.items()
    if key != "annotated_image"
}

    prediction_json = json.dumps(prediction_for_json, indent=2)
    quality_json = json.dumps(quality, indent=2)
    indicator_json = json.dumps(indicator, indent=2)

    return (
        prediction["annotated_image"],
        summary,
        prediction["pasture_soil_condition"],
        quality_json,
        prediction_json,
        indicator_json,
    )


def save_feedback(
    source_type: str,
    grazing_area_id: str,
    latitude: str,
    longitude: str,
    predicted_condition: str,
    corrected_label: str,
    user_comment: str,
    quality_json: str,
    prediction_json: str,
):
    ensure_output_files()

    if not predicted_condition:
        return "No prediction available. Please analyze an image first."

    if not corrected_label:
        return "Please select a corrected label before saving feedback."

    try:
        prediction = json.loads(prediction_json)
    except Exception:
        prediction = {}

    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                source_type,
                grazing_area_id,
                latitude,
                longitude,
                prediction.get("top_yolo_detection", ""),
                prediction.get("top_yolo_confidence", ""),
                prediction.get("pasture_soil_condition", predicted_condition),
                prediction.get("pasture_soil_confidence", ""),
                corrected_label,
                user_comment,
                prediction.get("mode", ""),
                quality_json,
                prediction_json,
            ]
        )

    return f"Feedback saved to {FEEDBACK_FILE}"


ensure_output_files()

with gr.Blocks(title="Pasture and Soil Vision DFence Prototype") as demo:
    gr.Markdown(
        """
        # Pasture and Soil Vision DFence Prototype

        Self-service computer vision workflow for pasture and soil image analysis.

        This prototype demonstrates:
        - image upload from different sources
        - CPU-based YOLO object detection
        - pasture/soil condition estimation
        - georeferenced indicator generation
        - human correction feedback for annotation/retraining
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload pasture/soil image")

            source_type = gr.Dropdown(
                choices=["drone", "satellite", "rover", "animal_camera", "manual_upload"],
                value="manual_upload",
                label="Image source type",
            )

            grazing_area_id = gr.Textbox(
                label="Grazing area ID",
                value="area_001",
            )

            latitude = gr.Textbox(
                label="Latitude",
                value="41.1579",
            )

            longitude = gr.Textbox(
                label="Longitude",
                value="-8.6291",
            )

            analyze_button = gr.Button("Analyze Image")

            corrected_label = gr.Dropdown(
                choices=PASTURE_SOIL_CLASSES,
                label="Corrected pasture/soil label / human feedback",
                value=None,
            )

            user_comment = gr.Textbox(
                label="Optional comment",
                placeholder="Example: This should be overgrazed_area because vegetation is sparse.",
                lines=3,
            )

            feedback_button = gr.Button("Save Feedback")
            feedback_status = gr.Textbox(label="Feedback Status", lines=2)

        with gr.Column():
            output_image = gr.Image(label="Annotated Output")
            prediction_summary = gr.Textbox(label="Prediction Summary", lines=22)

            predicted_condition_state = gr.Textbox(label="Predicted Condition", visible=False)
            quality_json_state = gr.Textbox(label="Quality JSON", visible=False)
            prediction_json_state = gr.Textbox(label="Prediction JSON", visible=False)
            indicator_json_state = gr.Textbox(label="Indicator JSON", visible=False)

    analyze_button.click(
        fn=analyze_image,
        inputs=[
            input_image,
            source_type,
            grazing_area_id,
            latitude,
            longitude,
        ],
        outputs=[
            output_image,
            prediction_summary,
            predicted_condition_state,
            quality_json_state,
            prediction_json_state,
            indicator_json_state,
        ],
    )

    feedback_button.click(
        fn=save_feedback,
        inputs=[
            source_type,
            grazing_area_id,
            latitude,
            longitude,
            predicted_condition_state,
            corrected_label,
            user_comment,
            quality_json_state,
            prediction_json_state,
        ],
        outputs=[feedback_status],
    )


if __name__ == "__main__":
    demo.launch()