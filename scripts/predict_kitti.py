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
from common import CLASSES, parse_imgsz, preprocess, decode_yolo, model_input_wh, jsonl_append  # noqa: E402

import numpy as np
import onnxruntime as ort



def stride_slice(w, h, stride, n_anchors, strides=(8, 16, 32)):
    """YOLOv8/11 Detect 출력의 앵커 순서는 P3(stride 8) -> P4(16) -> P5(32) 를 이어 붙인 것이다.
    지정 stride 에 해당하는 앵커 구간(slice)을 돌려준다."""
    counts = [(h // s) * (w // s) for s in strides]
    if sum(counts) != n_anchors:
        raise SystemExit(f"앵커 수 불일치: 계산 {counts} 합 {sum(counts)} != 모델 출력 {n_anchors} "
                         f"(P5 를 이미 뗀 모델이거나 stride 구성이 다르다)")
    i = strides.index(stride)
    start = sum(counts[:i])
    return slice(start, start + counts[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--split", default="splits/eval_val.txt")
    ap.add_argument("--imgsz", default=None, help="미지정 시 모델 입력 크기를 그대로 쓴다")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jsonl", default="benchmarks/results.jsonl",
                    help="단계별(전처리/추론/후처리) 평균 시간을 기록한다. 지연 공식 수치는 bench_onnx.py 기준")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="AP 계산용이므로 낮게 둔다. 0.25 같은 값을 쓰면 recall이 잘려 AP가 낮게 나온다.")
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--drop-stride", type=int, default=0, choices=[0, 8, 16, 32],
                    help=("이 stride 검출 헤드의 출력을 버리고 채점한다 (재학습 없는 ablation 근사). "
                          "예: 32 = P5 헤드를 뗀 것처럼. 재학습하면 남은 헤드가 일부 메워 주므로 '최대 피해'에 가깝다"))
    a = ap.parse_args()

    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    w, h = parse_imgsz(a.imgsz) if a.imgsz else model_input_wh(sess)
    drop = stride_slice(w, h, a.drop_stride, sess.get_outputs()[0].shape[-1]) if a.drop_stride else None
    if drop:
        print(f"  [ablation] stride {a.drop_stride} 출력 {drop.stop - drop.start}칸을 버리고 채점한다 "
              f"(앵커 {drop.start}~{drop.stop - 1})")

    ids = Path(a.split).read_text().split()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_det = 0
    tm = {"pre": 0.0, "infer": 0.0, "post": 0.0}
    for k, i in enumerate(ids):
        img = Path(a.images) / f"{i}.png"
        t1 = time.perf_counter()
        x, r, px, py, (oh, ow) = preprocess(img, w, h)
        t2 = time.perf_counter()
        y = sess.run(None, {iname: x})[0]
        t3 = time.perf_counter()
        if drop:
            y = np.array(y, copy=True)
            y[0, 4:, drop] = 0.0           # 그 헤드의 클래스 점수를 0 -> conf 문턱에서 모두 탈락
        dets = decode_yolo(np.asarray(y), r, px, py, oh, ow, a.conf, a.iou)
        t4 = time.perf_counter()
        tm["pre"] += t2 - t1; tm["infer"] += t3 - t2; tm["post"] += t4 - t3
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
    n = max(1, len(ids))
    stage = {f"{k}_ms": round(1000 * v / n, 2) for k, v in tm.items()}
    print(f"[완료] {len(ids)}장, 예측 {n_det}개, {time.time()-t0:.1f}s -> {out}")
    print(f"       장당 평균: 전처리 {stage['pre_ms']}ms / 추론 {stage['infer_ms']}ms / "
          f"후처리(conf={a.conf}) {stage['post_ms']}ms  (threads={a.threads})")
    if not a.drop_stride:   # ablation 근사는 실제 배포 경로가 아니므로 시간 기록을 남기지 않는다
        jsonl_append(a.jsonl, {"kind": "e2e", "model": Path(a.model).stem, "imgsz": f"{w}x{h}",
                               "threads": a.threads, "conf": a.conf, "n_images": len(ids), **stage,
                               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})


if __name__ == "__main__":
    main()
