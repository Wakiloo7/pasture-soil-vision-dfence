from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


PASTURE_SOIL_CLASSES = [
    "vegetation_cover",
    "bare_soil",
    "waterlogged_soil",
    "degraded_area",
    "overgrazed_area",
    "suitable_grazing_area",
]


class PastureSoilPredictor:
    """
    DFence-style computer vision predictor.

    This prototype separates:
    1. YOLO object detection for farm-context objects such as cows, people, vehicles.
    2. Pasture/soil condition estimation using simple image heuristics.

    For a real DFence deployment, the pasture/soil classes should be trained
    with a custom annotated dataset.
    """

    def __init__(self, custom_model_path: Optional[str] = None, pretrained_model_path: Optional[str] = None):
        self.custom_model_path = custom_model_path
        self.pretrained_model_path = pretrained_model_path
        self.model = None
        self.model_type = "demo_heuristic_only"

        if YOLO is not None:
            if custom_model_path and Path(custom_model_path).exists():
                self.model = YOLO(custom_model_path)
                self.model_type = "custom_pasture_soil_yolo"
            elif pretrained_model_path and Path(pretrained_model_path).exists():
                self.model = YOLO(pretrained_model_path)
                self.model_type = "pretrained_yolov8n_coco"

    def predict(self, image: Image.Image) -> Dict[str, object]:
        image_rgb = np.array(image.convert("RGB"))

        pasture_result = self.estimate_pasture_soil_condition(image_rgb)
        yolo_result = self.run_yolo_detection(image_rgb)

        annotated_image = self.create_combined_annotation(
            image_rgb=image_rgb,
            yolo_result=yolo_result,
            pasture_result=pasture_result,
        )

        return {
            "top_yolo_detection": yolo_result["top_detection"],
            "top_yolo_confidence": yolo_result["top_confidence"],
            "yolo_predictions": yolo_result["predictions"],
            "pasture_soil_condition": pasture_result["condition"],
            "pasture_soil_confidence": pasture_result["confidence"],
            "pasture_soil_scores": pasture_result["scores"],
            "annotated_image": annotated_image,
            "mode": self.model_type,
        }

    def run_yolo_detection(self, image_rgb: np.ndarray) -> Dict[str, object]:
        """
        Run YOLO detection on CPU.

        If no YOLO model exists, return empty detection.
        """
        if self.model is None:
            return {
                "top_detection": "no_yolo_model",
                "top_confidence": 0.0,
                "predictions": [],
            }

        results = self.model.predict(
            source=image_rgb,
            device="cpu",
            imgsz=640,
            conf=0.25,
            verbose=False,
        )

        result = results[0]
        predictions = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = result.names.get(class_id, str(class_id))

                predictions.append(
                    {
                        "label": label,
                        "confidence": round(confidence, 4),
                        "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    }
                )

        if predictions:
            top_detection = predictions[0]["label"]
            top_confidence = predictions[0]["confidence"]
        else:
            top_detection = "no_object_detected"
            top_confidence = 0.0

        return {
            "top_detection": top_detection,
            "top_confidence": top_confidence,
            "predictions": predictions,
        }

    def estimate_pasture_soil_condition(self, image_rgb: np.ndarray) -> Dict[str, object]:
        """
        Estimate pasture/soil condition using simple RGB/HSV heuristics.

        This is not a trained model. It is a prototype-level estimator used
        until a custom pasture/soil dataset is annotated and trained.
        """
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        green_mask = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
        brown_mask = cv2.inRange(hsv, (5, 30, 30), (30, 255, 220))
        dark_wet_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 70))

        total_pixels = image_rgb.shape[0] * image_rgb.shape[1]

        green_ratio = float(np.sum(green_mask > 0) / total_pixels)
        brown_ratio = float(np.sum(brown_mask > 0) / total_pixels)
        wet_dark_ratio = float(np.sum(dark_wet_mask > 0) / total_pixels)

        vegetation_cover_score = green_ratio
        bare_soil_score = brown_ratio
        waterlogged_score = wet_dark_ratio

        degraded_score = max(0.0, 1.0 - green_ratio - brown_ratio)
        overgrazed_score = min(1.0, brown_ratio * 0.7 + max(0.0, 0.25 - green_ratio))
        suitable_grazing_score = max(0.0, green_ratio - wet_dark_ratio * 0.5)

        scores = {
            "vegetation_cover": round(vegetation_cover_score, 4),
            "bare_soil": round(bare_soil_score, 4),
            "waterlogged_soil": round(waterlogged_score, 4),
            "degraded_area": round(degraded_score, 4),
            "overgrazed_area": round(overgrazed_score, 4),
            "suitable_grazing_area": round(suitable_grazing_score, 4),
        }

        condition = max(scores, key=scores.get)
        confidence = scores[condition]

        if green_ratio > 0.40 and wet_dark_ratio < 0.20:
            condition = "suitable_grazing_area"
            confidence = min(0.95, green_ratio + 0.35)
        elif wet_dark_ratio > 0.28:
            condition = "waterlogged_soil"
            confidence = min(0.90, wet_dark_ratio + 0.45)
        elif brown_ratio > 0.35 and green_ratio < 0.20:
            condition = "bare_soil"
            confidence = min(0.90, brown_ratio + 0.40)
        elif brown_ratio > 0.25 and green_ratio < 0.25:
            condition = "overgrazed_area"
            confidence = min(0.85, brown_ratio + 0.35)
        elif green_ratio > 0.25:
            condition = "vegetation_cover"
            confidence = min(0.85, green_ratio + 0.35)
        else:
            condition = "degraded_area"
            confidence = 0.55

        return {
            "condition": condition,
            "confidence": round(float(confidence), 4),
            "scores": scores,
        }

    def create_combined_annotation(
        self,
        image_rgb: np.ndarray,
        yolo_result: Dict[str, object],
        pasture_result: Dict[str, object],
    ) -> np.ndarray:
        annotated = image_rgb.copy()

        # Draw YOLO bounding boxes
        for pred in yolo_result["predictions"]:
            box = pred.get("box")
            if box is None:
                continue

            x1, y1, x2, y2 = box
            label = pred["label"]
            confidence = pred["confidence"]

            cv2.rectangle(
                annotated,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )

            # Put YOLO label close to the box, smaller text
            cv2.putText(
                annotated,
                f"YOLO: {label} {confidence:.2f}",
                (int(x1), max(18, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

        # Add a dark background panel for pasture/soil summary
        panel_x1, panel_y1 = 10, 10
        panel_x2, panel_y2 = 360, 130

        cv2.rectangle(
            annotated,
            (panel_x1, panel_y1),
            (panel_x2, panel_y2),
            (0, 0, 0),
            -1,
        )

        # Slight transparency effect
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (panel_x1, panel_y1),
            (panel_x2, panel_y2),
            (0, 0, 0),
            -1,
        )
        annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)

        overlay_lines = [
            f"Pasture/soil: {pasture_result['condition']}",
            f"Confidence: {pasture_result['confidence']:.2f}",
            f"Vegetation: {pasture_result['scores']['vegetation_cover']:.2f}",
            f"Bare soil: {pasture_result['scores']['bare_soil']:.2f}",
            f"Waterlogged: {pasture_result['scores']['waterlogged_soil']:.2f}",
        ]

        y_position = 32
        for line in overlay_lines:
            cv2.putText(
                annotated,
                line,
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            y_position += 22

        return annotated