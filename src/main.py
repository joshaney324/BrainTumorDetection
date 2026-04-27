from ultralytics import YOLO
from ensemble_helpers import ensemble_predict, visualize

# config
data_path = "/home/x89h835/BrainTumorDetection/data/combined/combined_data.yaml"
tune = True

# set up 3 models more ensemble
model_nano = YOLO("/home/x89h835/BrainTumorDetection/models/yolo26n.pt")
model_medium = YOLO("/home/x89h835/BrainTumorDetection/models/yolo26m.pt")
model_large = YOLO("/home/x89h835/BrainTumorDetection/models/yolo26x.pt")

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
                 epochs=100
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
                   epochs=100
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
                  epochs=100
                  )

# validation
model_nano.val(data=data_path)
model_medium.val(data=data_path)
model_large.val(data=data_path)

# nano
results = model_nano.predict(
    "/home/x89h835/BrainTumorDetection/data/combined/images/test/axial_00018_101.jpg")
results[0].show()
results[0].save("/home/x89h835/BrainTumorDetection/example/nano/axial.png")

results = model_nano.predict("/home/x89h835/BrainTumorDetection/data/combined/images/test/coronal_33.jpg")
results[0].show()
results[0].save("/home/x89h835/BrainTumorDetection/example/nano/coronal.png")

results = model_nano.predict(
    "/home/x89h835/BrainTumorDetection/data/combined/images/test/sagittal_00000_121.jpg")
results[0].show()
results[0].save("/home/x89h835/BrainTumorDetection/example/nano/sagittal.png")

# medium
results = model_medium.predict(
    "/home/x89h835/BrainTumorDetection/data/combined/images/test/axial_00018_101.jpg")
results[0].show()
results[0].save("/home/x89h835/BrainTumorDetection/example/medium/axial.png")

results = model_medium.predict("/home/x89h835/BrainTumorDetection/data/combined/images/test/coronal_33.jpg")
results[0].show()
results[0].save("/home/x89h835/BrainTumorDetection/example/medium/coronal.png")

results = model_medium.predict(
    "/home/x89h835/BrainTumorDetection/data/combined/images/test/sagittal_00000_121.jpg")
results[0].show()
results[0].save("/home/x89h835/BrainTumorDetection/example/medium/sagittal.png")

# large
results = model_large.predict(
    "/home/x89h835/BrainTumorDetection/data/combined/images/test/axial_00018_101.jpg")
results[0].show()
results[0].save("/home/x89h835/BrainTumorDetection/example/large/axial.png")

results = model_large.predict("/home/x89h835/BrainTumorDetection/data/combined/images/test/coronal_33.jpg")
results[0].show()
results[0].save("/home/x89h835/BrainTumorDetection/example/large/coronal.png")

results = model_large.predict(
    "/home/x89h835/BrainTumorDetection/data/combined/images/test/sagittal_00000_121.jpg")
results[0].show()
results[0].save("/home/x89h835/BrainTumorDetection/example/large/sagittal.png")

# ensemble
models = [model_nano, model_medium, model_large]

# weight larger models
weights = [1, 2, 3]
boxes, scores, labels = ensemble_predict(models,
                                         "/home/x89h835/BrainTumorDetection/data/combined/images/test/axial_00018_101.jpg",
                                         weights)
visualize("/home/x89h835/BrainTumorDetection/data/combined/images/test/axial_00018_101.jpg",
          "/home/x89h835/BrainTumorDetection/example/ensemble/axial.png", boxes, scores, labels)

boxes, scores, labels = ensemble_predict(models,
                                         "/home/x89h835/BrainTumorDetection/data/combined/images/test/coronal_33.jpg",
                                         weights)
visualize("/home/x89h835/BrainTumorDetection/data/combined/images/test/coronal_33.jpg",
          "/home/x89h835/BrainTumorDetection/example/ensemble/coronal.png", boxes, scores, labels)

boxes, scores, labels = ensemble_predict(models,
                                         "/home/x89h835/BrainTumorDetection/data/combined/images/test/sagittal_00000_121.jpg",
                                         weights)
visualize("/home/x89h835/BrainTumorDetection/data/combined/images/test/sagittal_00000_121.jpg",
          "/home/x89h835/BrainTumorDetection/example/ensemble/sagittal.png", boxes, scores, labels)
