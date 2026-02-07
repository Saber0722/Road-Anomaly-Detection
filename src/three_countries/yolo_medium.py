from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.train(
    data="../../data/yolo_binary/data.yaml",
    epochs=70,
    imgsz=416,
    batch=4,
    device=0,
    project="../../runs/three_country_training/road_defect_binary",
    name="yolov8m",
    patience=15,               # early stopping (stop if no val improvement)
    cache=False,               # safer for long runs
    workers=4,                 # stable default
    amp=True                   # mixed precision (faster, less VRAM)
)
