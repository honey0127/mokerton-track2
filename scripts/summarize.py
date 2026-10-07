"""benchmarks/results.jsonl -> 보고서용 표 (Markdown).

같은 모델·해상도의 FP32 를 기준으로 Drop Rate(정확도 하락률)와 속도 배율을 계산한다.
  Drop Rate(%) = (FP32 Moderate mAP - 양자화 Moderate mAP) / FP32 Moderate mAP x 100

모델 이름 규칙(export_onnx.py / quantize_ort.py 가 만드는 이름)에 기대어 묶는다:
  <가중치>_<W>x<H>_op13_<정밀도>   예) best_1280x384_op13_fp32, best_1280x384_op13_int8-mixed

  python scripts/summarize.py                       # 화면 출력
  python scripts/summarize.py --out benchmarks/summary.md
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path


def split_name(stem):
    if "_op" not in stem:
        return stem, "?"
    base, _, prec = stem.rpartition("_op")
    prec = prec.split("_", 1)[1] if "_" in prec else prec
    return base, prec


def load(path):
    recs = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            try:
                recs.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="benchmarks/results.jsonl")
    ap.add_argument("--split", default="splits/eval_val.txt",
                    help="이 split 의 AP40 기록만 쓴다 (holdout 튜닝 기록과 섞이지 않게)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", default=None, help="Pareto 그림 저장 경로 (예: benchmarks/pareto.png)")
    ap.add_argument("--threads", default="4,12", help="표에 넣을 스레드 조건. 첫 값이 주 조건(속도배율 기준)")
    a = ap.parse_args()
    threads = [int(t) for t in a.threads.split(",")]

    ap40, bench, stats, e2e = {}, defaultdict(dict), {}, {}
    for r in load(a.jsonl):                     # 같은 키는 나중 기록이 이긴다
        k = r.get("kind")
        if k == "ap40" and r.get("model") and r.get("split", a.split) == a.split:
            ap40[r["model"]] = r
        elif k == "bench":
            bench[Path(r["model"]).stem][int(r["threads"])] = r
        elif k == "model_stats":
            stats[f"{Path(r['weights']).stem}_{r['imgsz']}"] = r
        elif k == "e2e":
            e2e[r["model"]] = r

    models = sorted(set(ap40) | set(bench), key=lambda s: (split_name(s)[0], split_name(s)[1] != "fp32", s))
    hdr = ["모델(해상도)", "정밀도", "Moderate mAP", "Car", "Ped", "Cyc", "Drop(점)", "Drop Rate"]
    for t in threads:
        hdr += [f"p50 ms ({t}T)", f"p99 ms ({t}T)"]
    hdr += [f"속도배율({threads[0]}T)", "peak RSS MB", "크기 MB", "Params M", "GFLOPs"]
    rows = []
    for m in models:
        base, prec = split_name(m)
        ref = f"{base}_op13_fp32"
        A, R = ap40.get(m), ap40.get(ref)
        f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"          # noqa: E731
        drop = pts = None
        if A and R and R.get("mAP_moderate"):
            pts = R["mAP_moderate"] - A["mAP_moderate"]
            drop = pts / R["mAP_moderate"] * 100
        is_ref = prec == "fp32"
        row = [base, prec, f(A and A["mAP_moderate"]), f(A and A.get("Car_moderate")),
               f(A and A.get("Pedestrian_moderate")), f(A and A.get("Cyclist_moderate")),
               "기준" if is_ref else f(pts),
               "기준" if is_ref else (f(drop) + "%" if drop is not None else "-")]
        for t in threads:
            b = bench.get(m, {}).get(t)
            row += [f(b and b["p50_ms"]), f(b and b["p99_ms"])]
        b0, r0 = bench.get(m, {}).get(threads[0]), bench.get(ref, {}).get(threads[0])
        row += [f(r0["p50_ms"] / b0["p50_ms"]) + "x" if (b0 and r0) else "-"]
        bb = b0 or next(iter(bench.get(m, {}).values()), None)
        s = stats.get(base)
        row += [f(bb and bb["peak_rss_MB"], 1), f(bb and bb["size_MB"]),
                f(s and s["params_M"], 3), f(s and s["GFLOPs"] if isinstance(s and s["GFLOPs"], (int, float)) else None)]
        rows.append(row)

    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    notes = ["", f"- 정확도 기록: {a.split} 기준.",
             "- Drop(점) = FP32 mAP - 해당 모델 mAP. Drop Rate = Drop(점) / FP32 mAP x 100. 같은 해상도의 FP32 가 기준.",
             "- GFLOPs 는 연산 횟수라 정밀도와 무관하게 같다 (INT8 은 같은 연산을 더 싼 정수 연산으로 한다).",
             "- 지연은 bench_suite.py 기준: batch 1, 예열 120초 후 모델·스레드 순서를 섞은 5라운드(각 warm-up 30 + 300회)의 중앙값. 한 PC 에서만 측정."]
    if e2e:
        notes.append("- 전처리/추론/후처리 평균(ms, predict_kitti.py): " + "; ".join(
            f"{k}: {v['pre_ms']}/{v['infer_ms']}/{v['post_ms']}" for k, v in sorted(e2e.items())))
    text = "\n".join(md + notes)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
        print(f"\n-> {a.out}")
    if a.plot:
        pareto_plot(models, ap40, bench, threads[0], a.plot)


def pareto_plot(models, ap40, bench, t, path):
    """x = p50 지연(ms), y = Moderate mAP. 해상도별 색, 정밀도는 점 옆 글자. 왼쪽 위로 갈수록 좋다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = defaultdict(list)
    for m in models:
        b, A = bench.get(m, {}).get(t), ap40.get(m)
        if b and A:
            base, prec = split_name(m)
            pts[base].append((b["p50_ms"], A["mAP_moderate"], prec))
    if not pts:
        print("[plot] 지연과 AP40 이 둘 다 있는 모델이 없다 — bench-all 과 eval-all 을 먼저 돌린다")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for base, P in sorted(pts.items()):
        P.sort()
        ax.plot([p[0] for p in P], [p[1] for p in P], "o", label=base)
        for x, y, prec in P:
            ax.annotate(prec, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel(f"p50 latency (ms, ORT CPU, {t} threads)")
    ax.set_ylabel("KITTI Moderate AP40 (3-class mean)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"[plot] -> {path}")


if __name__ == "__main__":
    main()
