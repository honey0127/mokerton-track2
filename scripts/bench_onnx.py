import time, sys, json, statistics as st
import numpy as np, onnxruntime as ort, psutil, os

def bench(path, imgsz=640, warmup=20, runs=300, threads=None):
    so = ort.SessionOptions()
    if threads: so.intra_op_num_threads = threads
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    for _ in range(warmup): sess.run(None, {inp.name: x})

    lat = []
    proc = psutil.Process(os.getpid())
    peak = 0
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {inp.name: x})
        lat.append((time.perf_counter() - t) * 1000)
        peak = max(peak, proc.memory_info().rss)

    lat.sort()
    return {
        "model": os.path.basename(path),
        "size_MB": round(os.path.getsize(path)/1e6, 2),
        "p50_ms": round(st.median(lat), 2),
        "p95_ms": round(lat[int(len(lat)*0.95)], 2),
        "p99_ms": round(lat[int(len(lat)*0.99)], 2),
        "mean_ms": round(st.mean(lat), 2),
        "std_ms": round(st.pstdev(lat), 2),
        "fps_p50": round(1000/st.median(lat), 1),
        "peak_rss_MB": round(peak/1e6, 1),
        "threads": threads or "default",
    }

if __name__ == "__main__":
    r = bench(sys.argv[1], threads=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    print(json.dumps(r, indent=2))
