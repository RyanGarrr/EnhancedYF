from ultralytics import YOLO

yolo = YOLO("./runs/detect/train5/weights/best.pt",task="detect")

result = yolo(source="./ultralytics/assets/cut_400",save = True,conf = 0.2)