"""결정론적 지연/메모리 측정 — 문서화된 방법론을 코드가 그대로 강제한다.

방법론 (보고서 '벤치마크 및 측정 방법' 절과 1:1 대응):
  - batch=1 고정. 자율주행 실시간 인지는 처리량이 아니라 지연이 지표다.
  - warm-up 30회 제외 -> 300회 반복 -> 정렬 후 p50/p95/p99
  - 입력 시드 고정: np.random.default_rng(0)
  - ORT: intra_op 명시 고정, inter_op=1, ORT_SEQUENTIAL, 최적화 레벨 ALL
  - 메모리: psutil peak RSS
  - 보고 지표는 평균이 아니라 p50 + p99. 최악 프레임 지연이 안전과 직결된다.

  python scripts/bench_onnx.py models/x.onnx --imgsz 1280x384 --threads 12 --repeat 3
"""
from __future__ import annotations
import argparse, gc, json, os, statistics as st, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import parse_imgsz, jsonl_append  # noqa: E402

import numpy as np
import onnxruntime as ort
import psutil

WARMUP, RUNS = 30, 300


def make_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def bench_once(path, w, h, threads, runs=RUNS, warmup=WARMUP):
    sess = make_session(path, threads)
    inp = sess.get_inputs()[0]
    rng = np.random.default_rng(0)                      # 시드 고정
    x = rng.random((1, 3, h, w), dtype=np.float32)
    feed = {inp.name: x}

    for _ in range(warmup):
        sess.run(None, feed)

    proc = psutil.Process(os.getpid())
    peak = proc.memory_info().rss
    lat = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, feed)
        lat.append((time.perf_counter() - t) * 1000.0)
        peak = max(peak, proc.memory_info().rss)

    lat.sort()
    def q(p):
        return round(lat[min(len(lat) - 1, int(len(lat) * p))], 2)
    del sess
    gc.collect()
    return {
        "p50_ms": round(st.median(lat), 2), "p95_ms": q(0.95), "p99_ms": q(0.99),
        "mean_ms": round(st.fmean(lat), 2), "std_ms": round(st.pstdev(lat), 2),
        "fps_p50": round(1000.0 / st.median(lat), 1),
        "peak_rss_MB": round(peak / 1e6, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--imgsz", default=None,
                    help="미지정 시 모델 입력 크기를 그대로 쓴다 (1280x384 / 640x192 모델이 섞여 있어도 안전)")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--repeat", type=int, default=3,
                    help="전체 측정 반복 횟수. 단발 측정의 10%% 미만 차이는 결론으로 삼지 않는다.")
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--jsonl", default="benchmarks/results.jsonl")
    a = ap.parse_args()
    if a.imgsz:
        w, h = parse_imgsz(a.imgsz)
    else:
        # ORT 세션을 미리 만들면 그 메모리가 peak RSS 에 섞이므로 그래프 메타데이터만 읽는다
        import onnx
        dims = onnx.load(a.model, load_external_data=False).graph.input[0].type.tensor_type.shape.dim
        h, w = dims[2].dim_value, dims[3].dim_value
        if not (h and w):
            sys.exit("동적 입력 모델이다 — --imgsz 를 지정하라")

    reps = [bench_once(a.model, w, h, a.threads, a.runs) for _ in range(a.repeat)]
    agg = {
        "kind": "bench", "model": os.path.basename(a.model), "path": a.model,
        "imgsz": f"{w}x{h}", "threads": a.threads, "repeat": a.repeat, "runs": a.runs,
        "warmup": WARMUP, "tag": a.tag,
        "size_MB": round(os.path.getsize(a.model) / 1e6, 2),
        "p50_ms": round(st.median([r["p50_ms"] for r in reps]), 2),
        "p99_ms": round(st.median([r["p99_ms"] for r in reps]), 2),
        "p50_all": [r["p50_ms"] for r in reps],
        "p99_all": [r["p99_ms"] for r in reps],
        "peak_rss_MB": max(r["peak_rss_MB"] for r in reps),
        "fps_p50": round(1000.0 / st.median([r["p50_ms"] for r in reps]), 1),
        "ort": ort.__version__,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    jsonl_append(a.jsonl, agg)


if __name__ == "__main__":
    main()
