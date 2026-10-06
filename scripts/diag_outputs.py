"""FP32 / INT8 출력 텐서 분포 비교 진단.

전체 양자화가 mAP 0 을 내는 이유를 숫자로 보여주는 스크립트.
출력 [1, 4+nc, A] 에서 박스 행(0~3, 0~입력폭 픽셀)과 클래스 행(4~, 0~1)의
스케일 차이가 단일 양자화 스케일에 뭉개지는지 확인한다.

  python scripts/diag_outputs.py --models models/a_fp32.onnx models/a_int8.onnx \
      --image data/kitti_yolo/images/eval_val/000000.png --imgsz 1280x384
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import parse_imgsz, preprocess, model_input_wh  # noqa: E402

import numpy as np
import onnxruntime as ort


def stats(y, conf_thr=0.25):
    p = y[0] if y.ndim == 3 else y
    if p.shape[0] > p.shape[1]:
        p = p.T
    box, cls = p[:4], p[4:]
    conf = cls.max(0)
    return {
        "shape": list(y.shape),
        "box_min": round(float(box.min()), 4), "box_max": round(float(box.max()), 4),
        "cls_min": round(float(cls.min()), 5), "cls_max": round(float(cls.max()), 5),
        "cls_mean": round(float(cls.mean()), 6),
        "n_unique_cls": int(np.unique(cls).size),
        f"anchors_conf>{conf_thr}": int((conf > conf_thr).sum()),
        "implied_scale_if_single": round(float(box.max() - box.min()) / 255.0, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--imgsz", default=None, help="미지정 시 첫 모델의 입력 크기")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    if a.imgsz:
        w, h = parse_imgsz(a.imgsz)
    else:
        w, h = model_input_wh(ort.InferenceSession(a.models[0], providers=["CPUExecutionProvider"]))
    x, *_ = preprocess(a.image, w, h)

    rows = {}
    for m in a.models:
        s = ort.InferenceSession(m, providers=["CPUExecutionProvider"])
        y = np.asarray(s.run(None, {s.get_inputs()[0].name: x})[0])
        rows[Path(m).name] = stats(y, a.conf)

    print(json.dumps(rows, indent=2, ensure_ascii=False))

    # 판정은 절대 임계값이 아니라 첫 모델(FP32 기준) 대비 상대 비교로 한다.
    ref_k = list(rows)[0]
    ref = rows[ref_k]
    print(f"\n[판정] 기준: {ref_k}")
    print(f"  {'model':48s} {'cls_max':>10} {'cls_max 유지율':>14} {'고유값수':>9}  판정")
    for k, v in rows.items():
        ratio = v["cls_max"] / ref["cls_max"] if ref["cls_max"] > 0 else float("nan")
        # 클래스 행이 사실상 상수로 뭉개졌는지: 고유값 개수가 결정적 신호
        collapsed = v["n_unique_cls"] <= 2 or (ref["cls_max"] > 0 and ratio < 0.05)
        verdict = "클래스 점수 소실 -> mAP 0" if collapsed else "보존"
        if k == ref_k:
            verdict = "기준"
        print(f"  {k:48s} {v['cls_max']:>10.5f} {ratio:>13.2%} "
              f"{v['n_unique_cls']:>9d}  {verdict}")
    print("\n  고유값수가 1~2 이면 클래스 행 전체가 단일 스케일에 뭉개진 것이다 (트러블슈팅 4-2).")


if __name__ == "__main__":
    main()
