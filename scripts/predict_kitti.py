"""ONNX 모델 -> KITTI 포맷 예측 덤프 (kitti_eval.py 입력).

KITTI 예측 포맷 16필드: type trunc occ alpha x1 y1 x2 y2 h w l x y z ry score
2D 검출만 하므로 3D 필드는 더미(-1, -1000)로 채운다. 공식 평가는 2D bbox 와 score 만 본다.

  python scripts/predict_kitti.py --model models/m_int8-mixed.onnx \
      --images data/kitti_yolo/images/eval_val --split splits/eval_val.txt \
      --imgsz 1280x384 --out runs/pred_int8mixed
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import CLASSES, parse_imgsz, preprocess, decode_yolo  # noqa: E402

import numpy as np
import onnxruntime as ort



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--split", default="splits/eval_val.txt")
    ap.add_argument("--imgsz", default="1280x384")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="AP 계산용이므로 낮게 둔다. 0.25 같은 값을 쓰면 recall이 잘려 AP가 낮게 나온다.")
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    w, h = parse_imgsz(a.imgsz)

    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    ids = Path(a.split).read_text().split()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_det = 0
    for k, i in enumerate(ids):
        img = Path(a.images) / f"{i}.png"
        x, r, px, py, (oh, ow) = preprocess(img, w, h)
        y = sess.run(None, {iname: x})[0]
        dets = decode_yolo(np.asarray(y), r, px, py, oh, ow, a.conf, a.iou)
        n_det += len(dets)
        # KITTI 16필드: type trunc occ alpha bbox(4) hwl(3) xyz(3) ry score
        fixed = []
        skipped = 0
        for c, s, x1, y1, x2, y2 in dets:
            if c >= len(CLASSES):
                skipped += 1   # 3클래스 모델이 아니다 (COCO 가중치를 그대로 쓴 경우)
                continue
            fixed.append(f"{CLASSES[c]} -1 -1 -10 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                         f"-1000 -1000 -1000 -1000 -1000 -1000 -10 {s:.6f}")
        (out / f"{i}.txt").write_text("\n".join(fixed) + ("\n" if fixed else ""))
        if skipped and k == 0:
            print(f"  [경고] 클래스 인덱스가 {len(CLASSES)} 이상인 예측 {skipped}개를 버렸다. "
                  f"KITTI 3클래스로 학습한 가중치가 맞는지 확인하라.")
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(ids)}")
    print(f"[완료] {len(ids)}장, 예측 {n_det}개, {time.time()-t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
