#!/usr/bin/env python3
"""Hailo 에뮬레이터(SDK)로 HAR 을 돌려 KITTI 예측 덤프 + Moderate AP40 채점.

03_hailo_compile.sh 가 만든 *_optimized.har 를 입력으로 받는다. 실제 NPU 없이
'양자화된 모델이 정확도를 얼마나 잃었는지'를 본선 전에 확인하는 용도.

  context
    quantized : SDK_QUANTIZED   — NPU 와 같은 비트 정확도 에뮬레이션 (본 측정)
    fp        : SDK_FP_OPTIMIZED — 양자화 전 FP, 모델 스크립트(normalization 등) 반영
    native    : SDK_NATIVE       — parse 직후 원본 FP. 입력을 /255 해서 넣는다

  quantized vs fp 차이 = 순수 양자화 손실. fp 가 ONNX FP32 와 다르면 디코드/전처리 버그.

그래프는 Detect 디코드 직전 raw Conv 6개(또는 P5 제거 시 4개)에서 잘려 있으므로
DFL 디코드 + sigmoid 를 여기서 numpy 로 재현한 뒤, 저장소 scripts/common.py 의
decode_yolo(letterbox 역변환 + 클래스별 NMS)를 그대로 쓴다 — ORT 경로와 같은 후처리.

  python 04_emulate_eval.py --har hailo_out/A_1280x384/A_1280x384_optimized.har \
      --repo /mnt/d/ibzago/submit \
      --images /mnt/d/ibzago/mokerton-train/data/kitti_yolo/images/holdout \
      --gt /mnt/d/ibzago/mokerton-train/data/kitti_raw/training/label_2 \
      --split /mnt/d/ibzago/submit/splits/holdout.txt --context quantized --tag A_hailo_q
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path

import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis, keepdims=True)


def decode_heads(outs, in_h, in_w, nc, reg_max):
    """한 이미지의 head 출력 리스트 [(h,w,c), ...] -> (4+nc, A) cx,cy,w,h + 클래스 확률.

    Ultralytics Detect 와 같은 앵커 순서: stride 오름차순(P3->P4->P5), 각 grid 는 행 우선(y, x).
    """
    by_stride = {}
    for o in outs:
        h, w, c = o.shape
        s = in_h // h
        if in_w // w != s:
            raise SystemExit(f"출력 {o.shape} 의 stride 를 판정할 수 없다 (입력 {in_h}x{in_w})")
        slot = by_stride.setdefault(s, {})
        if c == 4 * reg_max:
            slot["box"] = o
        elif c == nc:
            slot["cls"] = o
        else:
            raise SystemExit(f"채널 수 {c} 는 box({4*reg_max}) 도 cls({nc}) 도 아니다 — --nc/--reg-max 확인")
    cols = []
    proj = np.arange(reg_max, dtype=np.float32)
    for s in sorted(by_stride):
        slot = by_stride[s]
        if set(slot) != {"box", "cls"}:
            raise SystemExit(f"stride {s} 에 box/cls 짝이 없다: {list(slot)}")
        h, w, _ = slot["box"].shape
        box = slot["box"].reshape(h * w, 4, reg_max).astype(np.float32)
        ltrb = softmax(box, -1) @ proj                                   # (hw, 4)
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        ax, ay = xs.reshape(-1) + 0.5, ys.reshape(-1) + 0.5
        x1, y1 = ax - ltrb[:, 0], ay - ltrb[:, 1]
        x2, y2 = ax + ltrb[:, 2], ay + ltrb[:, 3]
        xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 0) * s   # (4, hw)
        cls = 1.0 / (1.0 + np.exp(-slot["cls"].reshape(h * w, nc).astype(np.float32)))
        cols.append(np.concatenate([xywh, cls.T], 0))
    return np.concatenate(cols, 1)                                       # (4+nc, A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--har", required=True)
    ap.add_argument("--repo", required=True, help="submit 저장소 루트 (scripts/common.py, kitti_eval.py)")
    ap.add_argument("--images", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--gt", default=None, help="KITTI label_2 폴더. 주면 덤프 후 바로 채점")
    ap.add_argument("--out", default=None, help="기본: pred_<tag>")
    ap.add_argument("--tag", default="hailo_emu")
    ap.add_argument("--context", choices=["quantized", "fp", "native"], default="quantized")
    ap.add_argument("--imgsz", default="1280x384", help="WxH")
    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--reg-max", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N장만 (스모크 테스트용)")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--check-onnx", default=None,
                    help="원본 FP32 ONNX. 첫 배치에서 디코드 결과를 ORT 출력과 비교한다 (fp/native 에서 의미 있음)")
    ap.add_argument("--allow-eval", action="store_true",
                    help="eval_val 1000장 채점 허용. 모델 선택은 holdout 으로만 한다 (DECISION.md 규칙)")
    a = ap.parse_args()

    repo = Path(a.repo)
    sys.path.insert(0, str(repo / "scripts"))
    from common import CLASSES, letterbox, decode_yolo, parse_imgsz  # noqa: E402

    if "eval_val" in Path(a.split).name and not a.allow_eval:
        sys.exit("[중단] eval_val 은 최종 수치 산출용이다. 정말 채점하려면 --allow-eval")

    import cv2
    from hailo_sdk_client import ClientRunner, InferenceContext

    W, H = parse_imgsz(a.imgsz)
    ids = Path(a.split).read_text().split()
    if a.limit:
        ids = ids[:a.limit]
    out = Path(a.out or f"pred_{a.tag}")
    out.mkdir(parents=True, exist_ok=True)

    ctx_type = {"quantized": InferenceContext.SDK_QUANTIZED,
                "fp": InferenceContext.SDK_FP_OPTIMIZED,
                "native": InferenceContext.SDK_NATIVE}[a.context]
    scale = 255.0 if a.context == "native" else 1.0   # native 에는 normalization 레이어가 없다

    ort_sess = None
    if a.check_onnx:
        import onnxruntime as ort
        ort_sess = ort.InferenceSession(a.check_onnx, providers=["CPUExecutionProvider"])

    runner = ClientRunner(har=a.har)
    print(f"HAR {a.har}  context={a.context}  {len(ids)}장  {W}x{H}  -> {out}")

    t0 = time.time()
    n_det = 0
    with runner.infer_context(ctx_type) as ctx:
        for b0 in range(0, len(ids), a.batch):
            bid = ids[b0:b0 + a.batch]
            metas, xs = [], []
            for i in bid:
                im = cv2.imread(str(Path(a.images) / f"{i}.png"))
                if im is None:
                    sys.exit(f"이미지 없음: {Path(a.images) / f'{i}.png'}")
                oh, ow = im.shape[:2]
                lb, r, px, py = letterbox(im, W, H)
                xs.append(lb[:, :, ::-1])                       # BGR -> RGB, NHWC uint8
                metas.append((i, r, px, py, oh, ow))
            x = np.stack(xs).astype(np.float32) / scale
            res = runner.infer(ctx, x)
            if isinstance(res, np.ndarray):
                res = [res]
            if b0 == 0:
                print("  출력 텐서:", [tuple(o.shape) for o in res])

            for k, (i, r, px, py, oh, ow) in enumerate(metas):
                p = decode_heads([np.asarray(o[k]).reshape(o.shape[1:]) for o in res], H, W, a.nc, a.reg_max)
                if ort_sess is not None and b0 == 0 and k == 0:
                    xo = np.ascontiguousarray(xs[0].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
                    ref = ort_sess.run(None, {ort_sess.get_inputs()[0].name: xo})[0][0]
                    if ref.shape != p.shape:
                        print(f"  [check-onnx] 형상 불일치 ONNX {ref.shape} vs 에뮬 {p.shape}")
                    else:
                        d = np.abs(ref - p)
                        print(f"  [check-onnx] box max|Δ|={d[:4].max():.3f}px  "
                              f"cls max|Δ|={d[4:].max():.4f}  cls mean|Δ|={d[4:].mean():.5f}")
                dets = decode_yolo(p[None], r, px, py, oh, ow, a.conf, a.iou)
                n_det += len(dets)
                lines = [f"{CLASSES[c]} -1 -1 -10 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                         f"-1000 -1000 -1000 -1000 -1000 -1000 -10 {s:.6f}"
                         for c, s, x1, y1, x2, y2 in dets if c < len(CLASSES)]
                (out / f"{i}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

            done = b0 + len(bid)
            if done % (a.batch * 10) == 0 or done == len(ids):
                el = time.time() - t0
                print(f"  {done}/{len(ids)}  {el:.0f}s  ({el / done:.2f}s/장)", flush=True)

    print(f"[완료] {len(ids)}장, 예측 {n_det}개, {time.time() - t0:.0f}s -> {out}")

    if a.gt:
        cmd = [sys.executable, str(repo / "scripts" / "kitti_eval.py"), "--gt", a.gt, "--pred", str(out),
               "--split", a.split, "--tag", a.tag,
               "--out", str(out.parent / f"ap40_{a.tag}.json"),
               "--jsonl", str(out.parent / "emu_results.jsonl")]
        if a.limit:   # 일부만 돌렸으면 그 id 만 채점
            sub = out.parent / f"split_{a.tag}.txt"
            sub.write_text("\n".join(ids) + "\n")
            cmd[cmd.index("--split") + 1] = str(sub)
        print("채점:", " ".join(cmd))
        return subprocess.call(cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
