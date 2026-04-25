from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import numpy as np
from PIL import Image

# config
data_path = "../data/combined/combined_data.yaml"

# set up 3 models more ensemble
model_nano = YOLO("../models/yolo26n.pt")
model_medium = YOLO("../models/yolo26m.pt")
model_large = YOLO("../models/yolo26x.pt")


# training / tuning of the 3 models

# nano
model_nano.tune(
    data=data_path,
    epochs=30,
    iterations=100,
    optimizer="AdamW",
    plots=True,
    save=True,
)

model_nano.train(data=data_path,
            epochs=100
            )

# medium

model_medium.tune(
    data=data_path,
    epochs=30,
    iterations=100,
    optimizer="AdamW",
    plots=True,
    save=True,
)

model_medium.train(data=data_path,
            epochs=100
            )

# large

model_large.tune(
    data=data_path,
    epochs=30,
    iterations=100,
    optimizer="AdamW",
    plots=True,
    save=True,
)

model_large.train(data=data_path,
            epochs=100
            )

# validation
model_nano.val(data=data_path)
model_medium.val(data=data_path)
model_large.val(data=data_path)

# ensemble

models = [model_nano, model_medium, model_large]

# weight larger models
weights = [model_nano, model_medium, model_large]
