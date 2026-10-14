"""bench_suite.py 기록에서 '두 모델의 속도 차이를 말할 수 있는가'를 본다 — 속도 비와 95% 구간.

왜 필요한가 (2026-10-14, i7-10510U 노트북): 5라운드 중앙값은 같은 파일을 다시 재도 최대 8% 달랐다
(B INT8: 56.27 -> 51.88 ms). 모델 간 차이가 10% 안쪽이면 중앙값 두 개를 비교하는 것만으로는 결론을 낼 수 없다.
여기서는 같은 라운드끼리 묶어 라운드를 다시 뽑는 부트스트랩으로 '앞 모델 p50 / 뒤 모델 p50' 의 구간을 구한다.
구간에 1.00 이 들어가면 차이를 말할 수 없다.

  python scripts/bench_suite.py <주력 관련 8개> --threads 4 --rounds 15 --soak 120 --tag key15 \
      --jsonl benchmarks/bench_key15.jsonl --raw benchmarks/bench_key15_rounds.jsonl
  python scripts/bench_compare.py --jsonl benchmarks/bench_key15.jsonl --out benchmarks/bench_key15.md
"""
from __future__ import annotations
import argparse, json, random, statistics as st
from pathlib import Path

A, B, C, D = "best_1280x384", "best-p34_1280x384", "best_640x192", "best-p34_640x192"
INT8 = "int8-mixed-percentile"
# (이름, 앞 모델, 뒤 모델): 비가 1 보다 크면 뒤 모델이 빠르다
PAIRS = [
    ("INT8 속도배율 A", f"{A}_op13_fp32", f"{A}_op13_{INT8}"),
    ("INT8 속도배율 B", f"{B}_op13_fp32", f"{B}_op13_{INT8}"),
    ("INT8 속도배율 C", f"{C}_op13_fp32", f"{C}_op13_{INT8}"),
    ("INT8 속도배율 D", f"{D}_op13_fp32", f"{D}_op13_{INT8}"),
    ("B vs A (INT8)", f"{A}_op13_{INT8}", f"{B}_op13_{INT8}"),
    ("D vs C (INT8)", f"{C}_op13_{INT8}", f"{D}_op13_{INT8}"),
    ("B vs A (FP32)", f"{A}_op13_fp32", f"{B}_op13_fp32"),
    ("D vs C (FP32)", f"{C}_op13_fp32", f"{D}_op13_fp32"),
]


def ratio_ci(a, b, n_boot, seed):
    """a/b 의 중앙값 비와 95% 구간, 그리고 b 가 빨랐던 라운드 수. a, b 는 라운드 순서가 같은 p50 목록."""
    k = min(len(a), len(b))
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(k) for _ in range(k)]
        boots.append(st.median(a[i] for i in idx) / st.median(b[i] for i in idx))
    boots.sort()
    wins = sum(b[i] < a[i] for i in range(k))
    return st.median(a) / st.median(b), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)], wins, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="benchmarks/bench_key15.jsonl")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    recs = {}
    for ln in Path(a.jsonl).read_text(encoding="utf-8").splitlines():
        r = json.loads(ln) if ln.strip() else {}
        if r.get("kind") == "bench" and int(r["threads"]) == a.threads:
            recs[Path(r["model"]).stem] = r                      # 같은 모델은 나중 기록이 이긴다

    md = [f"| 모델 ({a.threads}스레드) | p50 중앙값 ms | 최소 | 최대 | 흔들림 | p99 중앙값 ms | 라운드 |", "|---|---|---|---|---|---|---|"]
    for n, r in sorted(recs.items()):
        p = r["p50_all"]
        md.append(f"| {n.replace('_op13', '')} | {st.median(p):.2f} | {min(p):.2f} | {max(p):.2f} | "
                  f"{r['p50_spread_pct']:.1f}% | {r['p99_ms']:.2f} | {len(p)} |")
    md += ["", "| 비교 | 속도 비 | 95% 구간 | 뒤 모델이 빠른 라운드 | 판정 |", "|---|---|---|---|---|"]
    for label, slow, fast in PAIRS:
        if slow not in recs or fast not in recs:
            continue
        r, lo, hi, wins, k = ratio_ci(recs[slow]["p50_all"], recs[fast]["p50_all"], a.boot, a.seed)
        verdict = "차이 있음" if (lo > 1.0 or hi < 1.0) else "구분 안 됨"
        md.append(f"| {label} | {r:.3f}x | {lo:.3f} ~ {hi:.3f} | {wins}/{k} | {verdict} |")
    md += ["", "- 속도 비 = 앞 모델 p50 / 뒤 모델 p50. 1 보다 크면 뒤 모델(INT8 또는 P5 제거)이 빠르다.",
           f"- 95% 구간: 같은 라운드끼리 묶어 라운드를 {a.boot}번 다시 뽑은 부트스트랩 (seed {a.seed}). 구간에 1.00 이 들어가면 '구분 안 됨'.",
           f"- 기록: {a.jsonl}"]
    text = "\n".join(md)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
