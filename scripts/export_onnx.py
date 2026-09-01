from ultralytics import YOLO
import sys

name = sys.argv[1] if len(sys.argv) > 1 else "yolo11n"
m = YOLO(f"{name}.pt")
# Hailo는 NMS 없는 raw head 3개를 원함. end2end/nms를 반드시 끌 것.
p = m.export(format="onnx", imgsz=640, opset=11, simplify=True, nms=False, dynamic=False)
print("EXPORTED:", p)
