"""여러 모델 x 스레드 조건을 '번갈아' 재는 벤치마크 — 순서·발열 효과 제거.

왜 필요한가 (2026-10-07 리허설, Core Ultra 7 155H 노트북):
  bench_onnx.py 를 모델마다 3회 '연속'으로 돌리면, 같은 모델의 3회 p50 이 최대 17~34% 달랐다
  (예: int8 4스레드 27.54 / 22.31 / 20.51 ms). 노트북 CPU 는 터보·전력 한도·발열 때문에
  시간에 따라 속도가 변하므로, 모델을 차례로 재면 '언제 쟀는지'가 결과에 섞인다.
  여기서는 라운드마다 (모델, 스레드) 순서를 섞어 1회씩 재고, R 라운드의 중앙값을 보고한다.
  그러면 시간에 따른 변화가 모든 모델에 고르게 퍼진다.

측정 1회 = bench_onnx.py 를 '새 프로세스'로 --repeat 1 실행 (warm-up 30, 300회, 시드 고정, ORT 설정 동일).
새 프로세스로 띄우는 이유: 한 프로세스에서 모델을 여러 개 돌리면 peak RSS 에 앞 모델 메모리가 섞인다.

예열(--soak): 2026-10-07 스레드 스윕에서 첫 라운드(약 1분)만 4스레드 이상이 7~19% 빨랐고(터보),
2스레드는 ±1% 로 그대로였다. 둘째·셋째 라운드끼리는 대부분 5% 안쪽으로 안정. 즉 흔들림의 대부분은
'처음 1분의 터보'다. 자율주행 인지는 계속 도는 작업이므로 예열 뒤의 '지속 성능'을 보고한다.

  python scripts/bench_suite.py models/*.onnx --threads 4,12 --rounds 5 --soak 120      # = make bench-all
  python scripts/bench_suite.py models/a_fp32.onnx models/a_int8-mixed.onnx \
      --threads 2,4,6,8,12,16 --rounds 3 --runs 150 --soak 120 --tag thread-sweep      # = make thread-sweep
"""
from __future__ import annotations
import argparse, json, os, random, statistics as st, subprocess, sys, tempfile, time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import jsonl_append  # noqa: E402


def run_once(model, threads, runs, tag):
    """bench_onnx.py 를 새 프로세스로 1회 실행하고 그 기록(dict)을 돌려준다."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp = f.name
    try:
        cmd = [sys.executable, str(HERE / "bench_onnx.py"), model, "--threads", str(threads),
               "--repeat", "1", "--runs", str(runs), "--tag", tag, "--jsonl", tmp]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"bench_onnx 실패: {model} t={threads}\n{r.stderr[-800:]}")
        lines = [l for l in Path(tmp).read_text(encoding="utf-8").splitlines() if l.strip()]
        return json.loads(lines[-1])
    finally:
        os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--threads", default="4,12", help="쉼표 구분. 첫 값이 주 조건 (예: 4,12 또는 2,4,6,8,12,16)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--runs", type=int, default=300, help="측정 1회의 반복 횟수 (warm-up 30 은 별도)")
    ap.add_argument("--seed", type=int, default=0, help="라운드별 순서 섞기 시드 (재현용)")
    ap.add_argument("--cooldown", type=float, default=0.0, help="측정 사이 쉬는 시간(초)")
    ap.add_argument("--soak", type=float, default=0.0,
                    help=("측정 전에 이 시간(초) 동안 첫 모델을 최대 스레드로 계속 돌려 CPU 를 '지속 상태'로 만든다. "
                          "2026-10-07 스윕: 첫 1분 남짓은 터보 때문에 4스레드 이상이 7~19%% 빨랐고 그 뒤로는 안정"))
    ap.add_argument("--tag", default="interleaved")
    ap.add_argument("--jsonl", default="benchmarks/results.jsonl", help="모델 x 스레드별 집계 기록")
    ap.add_argument("--raw", default="benchmarks/bench_rounds.jsonl", help="라운드별 원본 기록")
    a = ap.parse_args()

    threads = list(dict.fromkeys(int(t) for t in a.threads.split(",") if t.strip()))
    models = [m for m in a.models if m.endswith(".onnx")]
    jobs = [(m, t) for m in models for t in threads]
    rng = random.Random(a.seed)
    est = len(jobs) * a.rounds
    print(f"[bench_suite] 모델 {len(models)}개 x 스레드 {threads} x {a.rounds}라운드 = 측정 {est}회 (순서 무작위, seed={a.seed})")

    if a.soak > 0:
        ts, n = time.time(), 0
        print(f"[예열] {Path(models[0]).stem} t={max(threads)} 로 {a.soak:.0f}초 동안 계속 돌린다 (결과는 버림)")
        while time.time() - ts < a.soak:
            run_once(models[0], max(threads), a.runs, "soak")
            n += 1
        print(f"[예열] 끝 — {n}회, {time.time() - ts:.0f}초")

    res = defaultdict(list)
    t0, k = time.time(), 0
    for r in range(a.rounds):
        order = jobs[:]
        rng.shuffle(order)
        for m, t in order:
            rec = run_once(m, t, a.runs, f"{a.tag}/round{r + 1}")
            rec.update(round=r + 1, order="interleaved", seed=a.seed, soak_s=a.soak)
            res[(m, t)].append(rec)
            jsonl_append(a.raw, rec)
            k += 1
            eta = (time.time() - t0) / k * (est - k)
            print(f"  [{k:3d}/{est}] R{r + 1} {Path(m).stem:48s} t={t:<2d} "
                  f"p50={rec['p50_ms']:7.2f} p99={rec['p99_ms']:7.2f} rss={rec['peak_rss_MB']:6.1f}MB  (남은 약 {eta / 60:.0f}분)")
            if a.cooldown:
                time.sleep(a.cooldown)

    print(f"\n[집계] 라운드 중앙값 (흔들림 = (최대-최소)/중앙값, 10% 넘으면 '!')")
    print(f"  {'model':48s} " + "".join(f"{'t=' + str(t):>18s}" for t in threads))
    noisy = 0
    for m in models:
        cells = []
        for t in threads:
            reps = res[(m, t)]
            p50s, p99s = [x["p50_ms"] for x in reps], [x["p99_ms"] for x in reps]
            p50 = st.median(p50s)
            spread = (max(p50s) - min(p50s)) / p50 * 100
            noisy += spread > 10
            agg = {
                "kind": "bench", "model": Path(m).name, "path": m, "imgsz": reps[0]["imgsz"],
                "threads": t, "repeat": len(reps), "runs": a.runs, "warmup": reps[0]["warmup"],
                "tag": a.tag, "order": "interleaved", "seed": a.seed, "soak_s": a.soak,
                "size_MB": reps[0]["size_MB"],
                "p50_ms": round(p50, 2), "p99_ms": round(st.median(p99s), 2),
                "p50_all": p50s, "p99_all": p99s, "p50_spread_pct": round(spread, 1),
                "peak_rss_MB": max(x["peak_rss_MB"] for x in reps),
                "fps_p50": round(1000.0 / p50, 1), "ort": reps[0]["ort"],
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            jsonl_append(a.jsonl, agg)
            cells.append(f"{p50:8.2f} ({spread:4.1f}%){'!' if spread > 10 else ' '}")
        print(f"  {Path(m).stem:48s} " + "".join(f"{c:>18s}" for c in cells))
    print(f"\n  기록: {a.jsonl} (집계), {a.raw} (라운드별 원본). 총 {(time.time() - t0) / 60:.1f}분")
    if noisy:
        print(f"  [주의] 흔들림 10% 초과 {noisy}칸 — --soak 120 으로 예열했는지, 전원 연결·최고 성능 모드·다른 앱 종료 확인, "
              f"--rounds 를 늘리거나 --cooldown 5 를 준다. 10% 미만 차이는 결론으로 쓰지 않는다.")


if __name__ == "__main__":
    main()
