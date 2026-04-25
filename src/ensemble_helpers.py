from ensemble_boxes import weighted_boxes_fusion
import numpy as np
from PIL import Image


def ensemble_predict(models, image_path, weights, iou_thr=0.5, skip_box_thr=0.3):
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    all_boxes, all_scores, all_labels = [], [], []

    for model in models:
        prediction = model(image_path, verbose=False)[0]
        boxes = prediction.boxes.xyxy.cpu().numpy()
        scores = prediction.boxes.conf.cpu().numpy()
        labels = prediction.boxes.cls.cpu().numpy()

        if len(boxes) == 0:
            all_boxes.append([])
            all_scores.append([])
            all_labels.append([])
            continue

        norm = boxes.copy().astype(float)
        norm[:, [0, 2]] /= img_w
        norm[:, [1, 3]] /= img_h
        norm = np.clip(norm, 0, 1)

        all_boxes.append(norm.tolist())
        all_scores.append(scores.tolist())
        all_labels.append(labels.tolist())

    # combine boxes
    if any(len(b) > 0 for b in all_boxes):
        boxes, scores, labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )

        boxes[:, [0, 2]] *= img_w
        boxes[:, [1, 3]] *= img_h
    else:
        boxes = np.zeros((0, 4))
        scores = np.array([])
        labels = np.array([])

    return boxes, scores, labels
