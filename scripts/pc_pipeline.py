"""make 없는 PC(윈도우 공용 PC 등)용 실행기: ONNX 내보내기 -> INT8 양자화 -> holdout 공식 AP40 -> 표.

Makefile 의 export / quantize / quantize-extra / sensitivity / eval-all(SPLIT=holdout) / summary 와
같은 스크립트를 같은 인자로 부른다. 다른 점은 두 가지뿐이다.
  - 채점은 holdout 으로만 한다 (모델 고르기용). eval_val 은 이 실행기로 돌릴 수 없다.
  - 지연(ms)은 재지 않는다. 공식 지연은 노트북 한 대에서 bench_suite.py 로만 잰다.
    여기 기록은 benchmarks/pc5070/ 에 따로 쌓아 노트북 기록과 섞이지 않게 한다.

  python scripts/pc_pipeline.py core     # A(1280x384) + C(640x192): FP32 / INT8 / INT8-conv / INT8-mixed + P5 근사
  python scripts/pc_pipeline.py p34      # D(640x192, P5 제거) — runs/kitti_640_p34 학습이 끝난 뒤 (B 가 있으면 B 도)
  python scripts/pc_pipeline.py extra    # 추가 실험(A, C): 박스 FP32 / percentile / entropy / 민감도 top-k
  python scripts/pc_pipeline.py summary  # 표만 다시
반드시 저장소 루트에서, 학습용 가상환경(.venv-train)을 켠 상태로 실행한다.
"""
from __future__ import annotations
import argparse, glob, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = "scripts/"

# 구성 이름: (학습 가중치, ONNX 이름 앞부분, 추론 해상도, P5 근사 채점 여부)
CONFIGS = {
    "A": ("runs/kitti_1280/weights/best.pt", "best", "1280x384", True),
    "C": ("runs/kitti_640/weights/best.pt", "best", "640x192", True),
    "D": ("runs/kitti_640_p34/weights/best.pt", "best-p34", "640x192", False),
    "B": ("runs/kitti_1280_p34/weights/best.pt", "best-p34", "1280x384", False),
}


def run(args, out_file=None):
    """하위 스크립트 실행. 실패하면 거기서 멈춘다 (뒤 단계가 틀린 입력으로 돌지 않게)."""
    cmd = [sys.executable] + [str(a) for a in args]
    print("\n>>> python " + " ".join(str(a) for a in args), flush=True)
    t = time.time()
    if out_file:
        r = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        Path(ROOT / out_file).write_text(r.stdout, encoding="utf-8")
    else:
        r = subprocess.run(cmd, cwd=ROOT, env=ENV)
    if r.returncode != 0:
        sys.exit(f"\n[중단] 위 명령이 실패했다 (종료코드 {r.returncode}). 오류 메시지를 그대로 보내 주면 된다.")
    print(f"    ({time.time() - t:.0f}s)", flush=True)


def onnx_name(stem, res, tag="fp32"):
    return f"models/{stem}_{res}_op13_{tag}.onnx"


def finished(w):
    """학습이 끝까지 돈 가중치인가. Ultralytics 는 학습을 마칠 때 best.pt 의 epoch 를 -1 로 바꾼다.
    학습 도중(또는 중간에 죽은) best.pt 는 epoch 가 0 이상이다 -> 덜 된 모델을 실수로 채점하지 않게 막는다."""
    import torch
    ck = torch.load(ROOT / w, map_location="cpu", weights_only=False)
    return ck.get("epoch", -1) == -1


def prepare(cfg, a):
    """내보내기 + 그래프 검증 + 연산량 + 제외목록 + 양자화 3종 (Makefile export/quantize 와 같음)."""
    w, stem, res, _ = CONFIGS[cfg]
    if not (ROOT / w).exists():
        sys.exit(f"[중단] {cfg} 가중치가 없다: {w}")
    if not finished(w):
        sys.exit(f"[중단] {cfg} 학습이 아직 안 끝났다 (또는 중간에 멈췄다): {w}")
    if stem != "best":                      # D/B 는 best.pt 이름이 C/A 와 겹치므로 이름을 바꾼 사본으로 내보낸다
        w2 = str(Path(w).with_name(f"{stem}.pt").as_posix())
        shutil.copy2(ROOT / w, ROOT / w2)
        w = w2
    fp32 = onnx_name(stem, res)
    excl = f"models/exclude_{stem}_{res}.txt"
    print(f"\n========== [{cfg}] {w} @ {res}: 내보내기 + 양자화 ==========")
    run([S + "export_onnx.py", "--weights", w, "--imgsz", res, "--opset", "13"])
    run([S + "check_onnx.py", fp32])
    run([S + "model_stats.py", "--weights", w, "--imgsz", res, "--jsonl", a.jsonl])
    run([S + "head_nodes.py", fp32], out_file=excl)
    q = [S + "quantize_ort.py", "--model", fp32, "--calib", a.calib, "--n-calib", a.n_calib, "--imgsz", res]
    run(q)                                  # 전체 INT8 (실패 재현: 클래스 점수가 뭉개진다)
    run(q + ["--op-types", "Conv"])
    run(q + ["--exclude", excl])            # 채택 후보: Detect 디코드 경로 FP32


def extra(cfg, a):
    """quantize-extra + sensitivity (Makefile 과 같음). prepare 가 먼저 돌아 있어야 한다."""
    _, stem, res, _ = CONFIGS[cfg]
    fp32, mixed = onnx_name(stem, res), onnx_name(stem, res, "int8-mixed")
    excl, box = f"models/exclude_{stem}_{res}.txt", f"models/exclude_{stem}_{res}_box.txt"
    for f in (fp32, mixed, excl):
        if not (ROOT / f).exists():
            sys.exit(f"[중단] {f} 가 없다 — core 를 먼저 돌린다")
    print(f"\n========== [{cfg}] 추가 실험 @ {res} ==========")
    q = [S + "quantize_ort.py", "--model", fp32, "--calib", a.calib, "--n-calib", a.n_calib, "--imgsz", res]
    run([S + "head_nodes.py", fp32, "--box-fp32"], out_file=box)
    run(q + ["--exclude", box, "--suffix", "boxfp32"])
    run(q + ["--exclude", excl, "--calib-method", "percentile"])
    run(q + ["--exclude", excl, "--calib-method", "entropy"])
    run([S + "sensitivity.py", "--fp32", fp32, "--qdq", mixed, "--base-exclude", excl,
         "--images", f"{a.images}/holdout", "--n", "16", "--topk", "2,4,8"])
    for k in (2, 4, 8):
        run(q + ["--exclude", f"models/exclude_{stem}_{res}_op13_int8-mixed_sens_top{k}.txt", "--suffix", f"sens{k}"])


def evaluate(pattern, a, drop_stride=0, suffix=""):
    """holdout 예측 -> 공식 AP40. pattern 에 맞는 ONNX 를 모두 채점한다."""
    files = sorted(glob.glob(str(ROOT / pattern)))
    if not files:
        sys.exit(f"[중단] 채점할 모델이 없다: {pattern}")
    for f in files:
        name = Path(f).stem + suffix
        pred = f"runs/pred_holdout_{name}"
        p = [S + "predict_kitti.py", "--model", Path(f).relative_to(ROOT).as_posix(),
             "--images", f"{a.images}/holdout", "--split", a.split, "--out", pred,
             "--threads", a.threads, "--jsonl", a.jsonl]
        if drop_stride:
            p += ["--drop-stride", drop_stride]
        run(p)
        run([S + "kitti_eval.py", "--gt", a.gt, "--pred", pred, "--split", a.split, "--tag", name,
             "--out", f"benchmarks/pc5070/ap40_holdout_{name}.json", "--jsonl", a.jsonl])


def summary(a):
    out = "benchmarks/pc5070/summary_holdout.md"
    run([S + "summarize.py", "--jsonl", a.jsonl, "--split", a.split, "--threads", "4", "--out", out])
    print(f"\n[표] {out}  <- 이 내용을 보내 주면 된다. 여기 ms 값(전처리/추론/후처리)은 공용 PC 값이라 보고서에 쓰지 않는다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["core", "p34", "extra", "summary"])
    ap.add_argument("--jsonl", default="benchmarks/pc5070/results.jsonl")
    ap.add_argument("--split", default="splits/holdout.txt")
    ap.add_argument("--images", default="data/kitti_yolo/images", help="그 아래 holdout/ 과 train/ 이 있어야 한다")
    ap.add_argument("--gt", default="data/kitti_raw/training/label_2")
    ap.add_argument("--calib", default="data/kitti_yolo/images/train", help="캘리브레이션은 train 에서만")
    ap.add_argument("--n-calib", default="256")
    ap.add_argument("--threads", default="8", help="채점용 추론 스레드 (정확도에는 영향 없음)")
    a = ap.parse_args()
    if "eval_val" in a.split or "eval_val" in a.calib:
        sys.exit("[중단] eval_val 은 최종 1회 채점에만 쓴다. 이 실행기는 holdout 전용이다.")
    try:
        import numba, onnx, onnxruntime, ultralytics  # noqa: F401
    except ImportError as e:
        sys.exit(f"[중단] {e.name} 가 없다 — 학습용 가상환경(.venv-train)을 켰는지 확인 ({sys.executable})")
    if Path.cwd().resolve() != ROOT:
        print(f"[알림] 저장소 루트({ROOT})를 기준으로 실행한다")

    global ENV
    tmp = ROOT / "runs" / "tmp"             # C: 가 꽉 찬 PC 대비: 임시 파일을 저장소(D:) 안에 만든다
    tmp.mkdir(parents=True, exist_ok=True)
    ENV = dict(os.environ, PYTHONUTF8="1", TEMP=str(tmp), TMP=str(tmp))
    (ROOT / "benchmarks" / "pc5070").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if a.mode == "core":
        run([S + "make_split.py", "--verify", "splits/eval_val.txt"])
        run([S + "kitti_eval.py", "--selftest"])
        for cfg in ("A", "C"):
            prepare(cfg, a)
        for cfg in ("A", "C"):
            _, stem, res, p5_proxy = CONFIGS[cfg]
            evaluate(f"models/{stem}_{res}_op13_*.onnx", a)
            if p5_proxy:   # P5 헤드를 뗀 것처럼 채점 (재학습 없는 근사 = 피해의 상한)
                evaluate(onnx_name(stem, res), a, drop_stride=32, suffix="_noP5")
    elif a.mode == "p34":
        done = []
        for c in ("D", "B"):
            w = CONFIGS[c][0]
            if not (ROOT / w).exists():
                continue
            if finished(w):
                done.append(c)
            else:
                print(f"[건너뜀] {c} 학습이 아직 안 끝났다 (또는 중간에 멈췄다): {w}")
        if not done:
            sys.exit("[중단] 끝난 P5 제거 모델이 없다 — D 학습(runs/kitti_640_p34)이 끝난 뒤 실행한다")
        for cfg in done:
            prepare(cfg, a)
            _, stem, res, _ = CONFIGS[cfg]
            evaluate(f"models/{stem}_{res}_op13_*.onnx", a)
    elif a.mode == "extra":
        for cfg in ("A", "C"):
            extra(cfg, a)
            _, stem, res, _ = CONFIGS[cfg]
            evaluate(f"models/{stem}_{res}_op13_int8-mixed-*.onnx", a)

    summary(a)
    print(f"\n[완료] {a.mode}: {(time.time() - t0) / 60:.1f}분")


ENV = None

if __name__ == "__main__":
    main()
