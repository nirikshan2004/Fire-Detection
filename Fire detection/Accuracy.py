import metrics
from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")

results = model.val()

fire_map50= results.box.map50 * 100

print(f"🔥 Fire Detection Accuracy (mAP@50): {metrics.fire_map50():.2f}%")