from ultralytics import YOLO
import optuna

model = YOLO("../models/yolo26n.pt")

model.train(data="../data/combined/combined_data.yaml",
            epochs=1,
            imgsz=512,
            batch=16,
            device="cpu"
            )

model.val(data="../data/combined/combined_data.yaml")
