"""KITTI fine-tuning (직사각형 입력).

주의: Ultralytics train 은 imgsz 에 [h, w] 리스트를 받지 않는다.
rect=True + imgsz=<장변> 으로 학습하고, export 시점에 고정 직사각형으로 내보낸다.

  python scripts/train_kitti.py --imgsz 1280 --epochs 100 --batch 16
"""
from __future__ import annotations
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--data", default="data/kitti.yaml")
    ap.add_argument("--imgsz", type=int, default=1280, help="장변. rect=True 로 단변은 자동")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="kitti_1280")
    a = ap.parse_args()

    from ultralytics import YOLO
    m = YOLO(a.weights)
    m.train(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch,
        device=a.device, rect=True, project="runs", name=a.name,
        # KITTI 는 도로 주행 영상이라 상하 반전/과한 회전은 해가 된다
        fliplr=0.5, flipud=0.0, degrees=0.0, mosaic=1.0, close_mosaic=10,
        scale=0.5, translate=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        patience=30, seed=0, deterministic=True,
    )


if __name__ == "__main__":
    main()
