# Pasture and Soil Vision DFence Prototype

This is a computer vision portfolio prototype inspired by the **DFence** research grant use case at INESC TEC.

The project demonstrates a self-service image analysis workflow for pasture and soil condition monitoring using **Python**, **YOLOv8**, **OpenCV**, and **Gradio**.

The prototype supports:

- Image upload from pasture/soil sources
- CPU-based YOLO object detection
- Pasture and soil condition estimation
- Vegetation, bare soil, and waterlogged area scoring
- Georeferenced indicator generation
- Human correction feedback
- Evaluation metric examples
- Documentation for taxonomy, annotation, evaluation, and self-service workflow

This is a **portfolio prototype only**. It is not a production agricultural monitoring system and it is not a field-validated model.

---

## Project Objective

The objective of this project is to demonstrate a practical computer vision workflow aligned with pasture and soil monitoring use cases.

The system is designed around the following workflow:

```text
Image Source
Satellite / Drone / Rover / Animal-mounted Camera / Manual Upload
        ↓
Image Upload
        ↓
Image Quality Check
        ↓
YOLO Object Detection
        ↓
Pasture/Soil Condition Estimation
        ↓
Georeferenced Indicator Generation
        ↓
Human Feedback / Correction
        ↓
Future Annotation Review or Model Retraining
```

---

## Grant Alignment

This project is aligned with the DFence grant requirements in the following ways:

| Grant Requirement | Project Implementation |
|---|---|
| Define visual taxonomy of grazing and soil conditions | `docs/visual_taxonomy.md` |
| Build image collection, preprocessing, annotation, and management pipeline | `src/preprocess.py`, `app/gradio_app.py`, `data/labels_sample/annotation_schema.json` |
| Develop classification, detection, or segmentation models | YOLOv8 object detection + pasture/soil condition estimator |
| Evaluate deep learning models such as YOLO, CNNs, segmentation, detection | `src/metrics.py`, `src/evaluate_demo.py`, `docs/evaluation_plan.md` |
| Extract georeferenced indicators | Georeferenced indicator output and CSV export |
| Integrate outputs with decision-support logic | Indicator JSON/CSV designed for downstream decision-support systems |
| Support self-service approaches for non-technical users | Gradio upload, prediction, correction, and feedback workflow |
| Allow user participation in image annotation | Human correction feedback stored for future annotation/retraining |

---

## Current Prototype Capabilities

The current prototype includes:

- Gradio web interface
- Upload of pasture/soil images
- Image source selection:
  - drone
  - satellite
  - rover
  - animal camera
  - manual upload
- Grazing area ID input
- Latitude and longitude input
- YOLOv8n CPU inference
- Farm-context object detection, for example cows
- Pasture/soil condition estimation
- Image quality checks:
  - blur score
  - brightness
  - too dark flag
  - too bright flag
- Georeferenced indicator output
- Human correction feedback
- Feedback saved to CSV
- Indicator records saved to CSV
- Demo evaluation script for accuracy, F1-score, and IoU

---

## Important Technical Clarification

The project currently uses **pretrained YOLOv8n**.

The pretrained YOLOv8n model is trained on the COCO dataset, so it can detect general objects such as:

```text
cow
person
horse
sheep
dog
car
truck
bird
```

It is **not trained** to directly detect DFence-specific pasture and soil classes such as:

```text
vegetation_cover
bare_soil
waterlogged_soil
degraded_area
overgrazed_area
suitable_grazing_area
```

For this reason, the prototype separates the output into two parts:

```text
1. YOLO object detection:
   Detects farm-context objects such as cows.

2. Pasture/soil condition estimation:
   Uses image-based vegetation, bare soil, and wet/dark ratios to estimate field condition.
```

In a real research implementation, the next step would be to annotate a custom pasture/soil dataset and train a custom YOLO or segmentation model.

---

## Project Structure

```text
pasture-soil-vision-dfence/
├── README.md
├── app/
│   └── gradio_app.py
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_yolo_training.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── preprocess.py
│   ├── predict.py
│   ├── metrics.py
│   └── evaluate_demo.py
├── data/
│   ├── sample_images/
│   ├── labels_sample/
│   │   └── annotation_schema.json
│   ├── feedback_records.csv
│   └── georeferenced_indicators.csv
├── docs/
│   ├── visual_taxonomy.md
│   ├── annotation_guidelines.md
│   ├── evaluation_plan.md
│   ├── georeferenced_indicators.md
│   └── self_service_workflow.md
├── models/
│   └── yolov8n.pt
├── requirements.txt
└── screenshots/
```

---

## Visual Taxonomy

The project uses the following pasture and soil condition classes:

```text
vegetation_cover
bare_soil
waterlogged_soil
degraded_area
overgrazed_area
suitable_grazing_area
```

### Class Descriptions

| Class | Description |
|---|---|
| vegetation_cover | Visible grass, pasture, or healthy vegetation |
| bare_soil | Exposed soil with little or no vegetation |
| waterlogged_soil | Wet, muddy, saturated, or water-covered soil |
| degraded_area | Damaged or unhealthy pasture/soil area |
| overgrazed_area | Area with very sparse vegetation due to grazing pressure |
| suitable_grazing_area | Area with healthy vegetation and acceptable grazing condition |

---

## Application Workflow

The Gradio app follows this workflow:

```text
User uploads pasture/soil image
        ↓
User selects image source type
        ↓
User enters grazing area ID and coordinates
        ↓
System runs image quality checks
        ↓
System runs YOLOv8n object detection on CPU
        ↓
System estimates pasture/soil condition
        ↓
System creates georeferenced indicator output
        ↓
User reviews prediction
        ↓
User optionally corrects the pasture/soil label
        ↓
Feedback is saved for future annotation or retraining
```

---

## Example Output

For an image of cows grazing in a field, the system may produce:

```text
Top YOLO detection: cow
YOLO confidence: 0.8741

Pasture/soil condition estimate: suitable_grazing_area
Pasture/soil confidence: 0.8573

Vegetation score: 0.5073
Bare soil score: 0.0327
Waterlogged score: 0.0930
Degraded score: 0.4600
Overgrazed score: 0.0229
Suitable grazing score: 0.4608
```

Example georeferenced metadata:

```text
Source type: manual_upload
Grazing area ID: area_001
Latitude: 41.1579
Longitude: -8.6291
```

---

## Georeferenced Indicators

The prototype generates indicators that could be used by a decision-support system.

Example indicator fields:

```text
timestamp
source_type
grazing_area_id
latitude
longitude
vegetation_cover_score
bare_soil_score
waterlogged_soil_score
degraded_area_score
overgrazed_area_score
suitable_grazing_area_score
condition_estimate
condition_confidence
model_mode
```

These indicators can support:

- Grazing area monitoring
- Dynamic virtual fence decision support
- Identification of suitable grazing zones
- Detection of overgrazed or degraded areas
- Tracking pasture condition over time
- Future integration with environmental sensor data

---

## Human Feedback and Annotation Support

The prototype supports human feedback through the Gradio interface.

The user can:

- Review the predicted pasture/soil condition
- Select a corrected label
- Add a comment
- Save feedback

Feedback is stored in:

```text
data/feedback_records.csv
```

This supports a Human-in-the-Loop annotation workflow where user corrections can later be reviewed and used for:

- Annotation improvement
- Dataset refinement
- Model retraining
- Error analysis
- Field expert validation

---

## Image Quality Checks

The system calculates basic image quality indicators:

```text
blur_score
brightness
is_blurry
is_too_dark
is_too_bright
```

These checks help identify low-quality images that may affect prediction reliability.

In a production implementation, low-quality images could be:

- rejected
- flagged for review
- sent to manual annotation
- excluded from training

---

## Evaluation Metrics

The project includes evaluation utilities for:

```text
accuracy
precision
recall
F1-score
confusion matrix
IoU
mean IoU
```

The evaluation demo is located at:

```text
src/evaluate_demo.py
```

Run it with:

```bash
python src/evaluate_demo.py
```

Example evaluation output:

```text
Classification metrics:
accuracy
precision_macro
recall_macro
f1_macro
confusion_matrix

Detection / segmentation-style metric:
IoU example
Mean IoU example
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pasture-soil-vision-dfence.git
cd pasture-soil-vision-dfence
```

### 2. Create a virtual environment

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

On Linux/Mac:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## YOLO Model Setup

The project can run in two modes.

### Mode 1: Pretrained YOLOv8n

This mode uses `yolov8n.pt` for general object detection.

Download YOLOv8n:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

Create the `models` folder:

```bash
mkdir models
```

Move the downloaded model:

```bash
move yolov8n.pt models\yolov8n.pt
```

On Linux/Mac:

```bash
mv yolov8n.pt models/yolov8n.pt
```

### Mode 2: Custom Pasture/Soil YOLO Model

For a future custom-trained model, place the trained weights here:

```text
models/best.pt
```

The app checks for models in this order:

```text
1. models/best.pt
2. models/yolov8n.pt
3. demo heuristic only
```

---

## Running the Gradio App

From the project root, run:

```bash
python app/gradio_app.py
```

The app will start locally:

```text
http://127.0.0.1:7860
```

Open the link in a browser and upload a pasture or farm-context image.

---

## Recommended Test Images

For the current pretrained YOLO model, use images containing COCO-recognizable objects in a pasture context:

```text
cow in pasture
sheep in grass field
horse in field
person in grassland
tractor in field
dog in field
bird in grassland
```

Best demo image:

```text
cows grazing in a pasture field
```

This allows the app to show:

```text
YOLO object detection: cow
Pasture/soil estimate: suitable_grazing_area
Georeferenced indicators: vegetation/bare soil/waterlogged scores
```

---

## Main Files

### `app/gradio_app.py`

Main Gradio application.

Responsible for:

- image upload
- source type input
- geolocation metadata input
- prediction display
- georeferenced indicator generation
- human feedback saving

### `src/predict.py`

Main prediction logic.

Responsible for:

- YOLO inference
- pasture/soil condition estimation
- vegetation/bare soil/waterlogged scoring
- annotated image generation

### `src/preprocess.py`

Image preprocessing utilities.

Responsible for:

- loading images
- converting to RGB
- resizing
- normalization
- quality checks

### `src/metrics.py`

Evaluation metrics.

Responsible for:

- classification metrics
- IoU calculation
- mean IoU calculation

### `src/evaluate_demo.py`

Runnable demonstration of evaluation metrics.

### `data/labels_sample/annotation_schema.json`

Sample annotation schema for the DFence-style pasture/soil classes.

---

## Annotation Schema

The project includes an annotation schema with the following classes:

```json
{
  "classes": [
    "vegetation_cover",
    "bare_soil",
    "waterlogged_soil",
    "degraded_area",
    "overgrazed_area",
    "suitable_grazing_area"
  ]
}
```

This schema supports future annotation workflows for object detection or segmentation.

---

## Future Improvements

The current project is a portfolio prototype. Future improvements could include:

1. **Custom YOLO Training**
   - Annotate pasture/soil images.
   - Train a YOLO model on DFence-specific classes.
   - Replace `models/yolov8n.pt` with `models/best.pt`.

2. **Semantic Segmentation**
   - Use segmentation models such as U-Net, DeepLabV3, or YOLO segmentation.
   - Estimate exact area percentages for vegetation, bare soil, and degraded zones.

3. **Multispectral or Hyperspectral Analysis**
   - Use vegetation indices such as NDVI.
   - Combine RGB images with NIR channels where available.

4. **Field Data Validation**
   - Compare model predictions with field measurements.
   - Validate using sensor data and agro-environmental indicators.

5. **Robustness Testing**
   - Test under different lighting conditions.
   - Test across seasons.
   - Test across terrain types.
   - Test across image sources.

6. **Decision Support Integration**
   - Export indicators to an API or database.
   - Integrate with virtual fence decision logic.
   - Track pasture condition over time.

7. **Improved Self-Service Workflow**
   - Allow batch uploads.
   - Allow producers to configure classes.
   - Add annotation review dashboard.
   - Add user role management.

---

## Limitations

This prototype has the following limitations:

- It does not use a custom-trained pasture/soil model yet.
- Pretrained YOLOv8n detects general COCO objects, not DFence-specific pasture classes.
- Pasture/soil condition estimation currently uses simple image heuristics.
- It has not been validated with field data.
- It does not yet integrate environmental sensor measurements.
- It does not yet implement semantic segmentation.
- It is not a production monitoring system.

These limitations are intentional and clearly defined because the project is a portfolio prototype.

---

## Interview Explanation

A concise way to explain this project:

```text
I built a portfolio prototype aligned with the DFence computer vision call. It provides a self-service Gradio interface where users can upload pasture or soil images, select the image source, provide grazing area metadata, and run CPU-based YOLO inference. The system detects farm-context objects such as cows using pretrained YOLOv8n and separately estimates pasture/soil condition using vegetation, bare soil, and waterlogged scores. It also generates georeferenced indicators and allows users to correct the output, saving feedback for future annotation or model retraining.

Because I do not yet have a custom annotated pasture/soil dataset, I do not claim this as a production model. The next step would be to annotate DFence-specific classes and train a custom YOLO or segmentation model, then evaluate it using F1-score, IoU, robustness testing, and field-data validation.
```

---

## Technologies Used

```text
Python
YOLOv8
Ultralytics
OpenCV
Gradio
Pillow
NumPy
Pandas
scikit-learn
PyTorch
CSV-based feedback storage
Computer vision
Object detection
Image preprocessing
Human-in-the-Loop feedback
Georeferenced indicators
```

---

## Author

**Md Wakil Ahmad**

GitHub: `https://github.com/Wakiloo7`

LinkedIn: `https://www.linkedin.com/in/md-wakil-ahmad`