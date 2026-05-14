from metrics import classification_metrics, calculate_iou, mean_iou


def main():
    print("DFence-style evaluation demo")
    print("=" * 40)

    y_true = [
        "suitable_grazing_area",
        "bare_soil",
        "waterlogged_soil",
        "vegetation_cover",
        "overgrazed_area",
        "degraded_area",
    ]

    y_pred = [
        "suitable_grazing_area",
        "bare_soil",
        "degraded_area",
        "vegetation_cover",
        "bare_soil",
        "degraded_area",
    ]

    metrics = classification_metrics(y_true, y_pred)

    print("\nClassification metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    true_box = [100, 100, 300, 300]
    pred_box = [120, 120, 280, 280]

    print("\nDetection / segmentation-style metric:")
    print(f"IoU example: {calculate_iou(true_box, pred_box)}")

    true_boxes = [
        [100, 100, 300, 300],
        [50, 50, 150, 150],
    ]

    pred_boxes = [
        [120, 120, 280, 280],
        [40, 40, 140, 140],
    ]

    print(f"Mean IoU example: {mean_iou(true_boxes, pred_boxes)}")


if __name__ == "__main__":
    main()