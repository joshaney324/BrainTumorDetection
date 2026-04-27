from ultralytics import YOLO
from ensemble_helpers import ensemble_predict, visualize

# config
data_path = "../data/combined/combined_data.yaml"
tune = False

# set up 3 models more ensemble
model_nano = YOLO("../models/yolo26n.pt")
model_medium = YOLO("../models/yolo26m.pt")
model_large = YOLO("../models/yolo26x.pt")

# training / tuning of the 3 models

# nano
if tune:
    model_nano.tune(
        data=data_path,
        epochs=30,
        iterations=10,
        optimizer="AdamW",
        plots=True,
        save=True,
    )

model_nano.train(data=data_path,
                 epochs=1
                 )

# medium

if tune:
    model_medium.tune(
        data=data_path,
        epochs=30,
        iterations=10,
        optimizer="AdamW",
        plots=True,
        save=True,
    )

model_medium.train(data=data_path,
                   epochs=1
                   )

# large

if tune:
    model_large.tune(
        data=data_path,
        epochs=30,
        iterations=10,
        optimizer="AdamW",
        plots=True,
        save=True,
    )

model_large.train(data=data_path,
                  epochs=1
                  )

# validation
model_nano.val(data=data_path)
model_medium.val(data=data_path)
model_large.val(data=data_path)

# nano
results = model_nano.predict(
    "C:\\ComputerScience\\BrainTumorDetection\\data\\combined\\images\\test\\axial_00018_101.jpg")
results[0].show()
results[0].save("../example/nano/axial.png")

results = model_nano.predict("C:\\ComputerScience\\BrainTumorDetection\\data\\combined\\images\\test\\coronal_33.jpg")
results[0].show()
results[0].save("../example/nano/coronal.png")

results = model_nano.predict(
    "C:\\ComputerScience\\BrainTumorDetection\\data\\combined\\images\\test\\sagittal_00000_121.jpg")
results[0].show()
results[0].save("../example/nano/sagittal.png")

# medium
results = model_medium.predict(
    "C:\\ComputerScience\\BrainTumorDetection\\data\\combined\\images\\test\\axial_00018_101.jpg")
results[0].show()
results[0].save("../example/medium/axial.png")

results = model_medium.predict("C:\\ComputerScience\\BrainTumorDetection\\data\\combined\\images\\test\\coronal_33.jpg")
results[0].show()
results[0].save("../example/medium/coronal.png")

results = model_medium.predict(
    "C:\\ComputerScience\\BrainTumorDetection\\data\\combined\\images\\test\\sagittal_00000_121.jpg")
results[0].show()
results[0].save("../example/medium/sagittal.png")

# large
results = model_large.predict(
    "C:\\ComputerScience\\BrainTumorDetection\\data\\combined\\images\\test\\axial_00018_101.jpg")
results[0].show()
results[0].save("../example/large/axial.png")

results = model_large.predict("C:\\ComputerScience\\BrainTumorDetection\\data\\combined\\images\\test\\coronal_33.jpg")
results[0].show()
results[0].save("../example/large/coronal.png")

results = model_large.predict(
    "C:\\ComputerScience\\BrainTumorDetection\\data\\combined\\images\\test\\sagittal_00000_121.jpg")
results[0].show()
results[0].save("../example/large/sagittal.png")

# ensemble
models = [model_nano, model_medium, model_large]

# weight larger models
weights = [1, 2, 3]
boxes, scores, labels = ensemble_predict(models,
                                         "C:\\ComputerScience\\BrainTumorDetection\\data\\combined\\images\\test\\axial_00018_101.jpg",
                                         weights)
visualize("C:\\ComputerScience\\BrainTumorDetection\\data\\combined\\images\\test\\axial_00018_101.jpg",
          "../example/ensemble/axial.png", boxes, scores, labels)

boxes, scores, labels = ensemble_predict(models,
                                         "C:\\ComputerScience\\BrainTumorDetection\\data\\combined\\images\\test\\coronal_33.jpg",
                                         weights)
visualize("C:\\ComputerScience\\BrainTumorDetection\\data\\combined\\images\\test\\coronal_33.jpg",
          "../example/ensemble/coronal.png", boxes, scores, labels)

boxes, scores, labels = ensemble_predict(models,
                                         "C:\\ComputerScience\\BrainTumorDetection\\data\\combined\\images\\test\\sagittal_00000_121.jpg",
                                         weights)
visualize("C:\\ComputerScience\\BrainTumorDetection\\data\\combined\\images\\test\\sagittal_00000_121.jpg",
          "../example/ensemble/sagittal.png", boxes, scores, labels)
