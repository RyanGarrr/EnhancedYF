from ultralytics import YOLO

#Load a model
model = YOLO('ultralytics/cfg/models/v8/yolov8bifpn.yaml')

#Train the model
model.train(data = 'yolo-car.yaml',workers = 0,epochs = 150,batch = 16)

# # Load a model
# model = YOLO('runs/detect/train5/weights/last.pt')  # load a partially trained model
#
# # Resume training
# results = model.train(resume=True)

# http://localhost:6006/