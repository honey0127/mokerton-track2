"""YOLO -> ONNX export (직사각형 입력 / opset 13 고정).

opset 13 이상이 필수다. opset 11로 내보낸 그래프에 per-channel PTQ를 적용하면
DequantizeLinear 에 opset13 문법인 axis 속성이 생겨 INVALID_GRAPH 로 로드가 실패한다.
Hailo DFC v5 는 opset 13 을 지원하므로 이식성 손실은 없다.

  python scripts/export_onnx.py --weights yolo11n.pt --imgsz 1280x384
  python scripts/export_onnx.py --weights runs/kitti/weights/best.pt --imgsz 640x192
"""
from __future__ import annotations
import argparse, shutil
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import parse_imgsz  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", default="1280x384", help="WxH, 32의 배수, W<=1280")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--outdir", default="models")
    a = ap.parse_args()

    if a.opset < 13:
        raise SystemExit("opset 13 미만은 per-channel PTQ에서 INVALID_GRAPH를 유발한다.")
    w, h = parse_imgsz(a.imgsz)

    from ultralytics import YOLO
    m = YOLO(a.weights)
    # Hailo는 NMS 없는 raw head를 요구한다. end2end/nms 를 반드시 끈다.
    # Ultralytics 의 imgsz 는 [h, w] 순서다.
    p = m.export(format="onnx", imgsz=[h, w], opset=a.opset,
                 simplify=True, nms=False, dynamic=False)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.weights).stem
    dst = out / f"{stem}_{w}x{h}_op{a.opset}_fp32.onnx"
    shutil.move(str(p), dst)
    print(f"EXPORTED: {dst}")
    return dst


if __name__ == "__main__":
    main()
