#!/usr/bin/env python3
"""보드 처리량·지연·자원 측정 (비동기 파이프라인). 06_board_e2e.py 의 후속.

06 은 한 장씩 [읽기 -> 전처리 -> run() -> 디코드 -> NMS] 를 직렬로 해서, NPU 단독 200 FPS(B) 인데
전체는 18 FPS 였다. 07 은 전처리·추론·후처리를 겹쳐 돌린다:

  전처리 스레드(--pre-workers) ─▶ NPU run_async (동시 진행 --inflight 묶음, 묶음당 --batch 장) ─▶ 후처리 스레드(--post-workers)

--batch, --inflight 에 쉼표 목록을 주면 조합을 모두 돌려 한 줄씩 출력한다.
--mode sync 는 06 과 같은 직렬 경로 (비교 기준).

입력 (--source)
  ram  : 이미지를 미리 디코드해 메모리에 둔다. 카메라 시연과 같은 조건(파일 읽기·PNG 디코드 없음). 기본값
  disk : 매 장 cv2.imread. 06 과 같은 조건. --dump(정확도 덤프)는 이걸로

자원: 프로세스 CPU%(100% = 코어 1개), 시스템 CPU%(100% = 4코어 전부), RSS(최대, 그리고
모델 올리기 전 기준값 rss_base), SoC 온도 최대, vcgencmd get_throttled 전후.

예 (보드)
  # 1) 디코드가 06 과 같은지 (출력 형식 둘 다)
  python3 hailo/07_board_fast.py --backend hailort --model hailo/hef/B_1280x384.hef --imgsz 1280x384 \
      --images $IMG --split splits/holdout.txt -n 50 --check-decode
  # 2) 조합 훑기
  python3 hailo/07_board_fast.py --backend hailort --model hailo/hef/B_1280x384.hef --imgsz 1280x384 \
      --images $IMG --split splits/holdout.txt -n 64 --frames 600 --batch 1,2,4,8 --inflight 1,2,4 \
      --json hailo/results/board/fast_B.jsonl
  # 3) 정확도 덤프 (conf 0.001, 결과는 06 덤프와 같아야 한다)
  python3 hailo/07_board_fast.py ... --source disk -n 500 --frames 500 --conf 0.001 --dump runs/fast_pred_B

예 (PC 검증, 시간은 의미 없음)
  python hailo/make_head_onnx.py models/best-p34_640x192_op13_fp32.onnx runs/head_D.onnx
  python hailo/07_board_fast.py --backend onnx --model runs/head_D.onnx --imgsz 640x192 --images ... --check-decode
"""
from __future__ import annotations
import argparse, gc, importlib.util, json, sys, threading, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hailo_rt import (FastDecoder, ResourceMonitor, dets_to_kitti, make_backend,  # noqa: E402
                      preprocess, decode_yolo, REPO)
sys.path.insert(0, str(REPO / "scripts"))
from common import parse_imgsz  # noqa: E402


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def load_ref_decoder():
    spec = importlib.util.spec_from_file_location("emu04", HERE / "04_emulate_eval.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.decode_heads


def find_image(folder, i):
    for ext in (".png", ".jpg", ".jpeg"):
        p = Path(folder) / f"{i}{ext}"
        if p.exists():
            return p
    sys.exit(f"이미지 없음: {folder}/{i}.png")


# --------------------------------------------------------------------------- 디코드 일치 확인
def check_decode(be, ids, images, W, H, conf, iou, nc, reg_max):
    import cv2
    ref_heads = load_ref_decoder()
    fast = FastDecoder(W, H, nc, reg_max, conf, iou)
    worst, n_diff, n_det, t_ref, t_fast = 0.0, 0, 0, [], []
    for i in ids:
        img = cv2.imread(str(find_image(images, i)))
        x, meta = preprocess(img, W, H)
        outs = be.run_sync([x])[0]
        fl = [o if not isinstance(o, tuple) else (o[0].astype(np.float32) - o[2]) * o[1] for o in outs]
        t0 = time.perf_counter()
        p = ref_heads([o.reshape(o.shape[-3:]) for o in fl], H, W, nc, reg_max)
        ref = decode_yolo(p[None], *meta, conf, iou)
        t1 = time.perf_counter()
        got = fast(outs, *meta)
        t2 = time.perf_counter()
        t_ref.append((t1 - t0) * 1000); t_fast.append((t2 - t1) * 1000)
        n_det += len(ref)
        if len(ref) != len(got) or any(a[0] != b[0] for a, b in zip(ref, got)):
            n_diff += 1
            continue
        if ref:
            d = np.abs(np.array([a[1:] for a in ref]) - np.array([b[1:] for b in got])).max()
            worst = max(worst, float(d))
    print(f"[check-decode] {len(ids)}장, 검출 {n_det}개 (conf {conf})")
    print(f"  검출 목록(개수·클래스·순서)이 다른 이미지: {n_diff}")
    print(f"  같은 이미지들에서 점수·좌표 최대 차이: {worst:.2e}")
    print(f"  디코드+NMS 평균: 기존 {np.mean(t_ref):.2f} ms -> 빠른 디코드 {np.mean(t_fast):.2f} ms")
    return n_diff == 0 and worst < 1e-3


# --------------------------------------------------------------------------- 실행 한 번
def run_config(a, W, H, ids, imgs, batch, inflight, out_dir=None):
    import cv2
    be = make_backend(a.backend, a.model, batch=batch, out_format=a.out_format)
    dec = FastDecoder(W, H, a.nc, a.reg_max, a.conf, a.iou)
    frames = [(ids[k % len(ids)], k) for k in range(a.frames)]
    rec = {"t_end": [], "lat": [], "npu": [], "pre": [], "post": [], "n_det": 0}
    lock = threading.Lock()
    errors = []

    def pre_job(item):
        i, k = item
        t0 = time.perf_counter()
        img = imgs[i] if imgs is not None else cv2.imread(str(find_image(a.images, i)))
        x, meta = preprocess(img, W, H)
        return i, k, x, meta, t0, time.perf_counter()

    def post_job(i, k, outs, meta, t0, t_pre, t_sub, t_cb):
        tp = time.perf_counter()
        dets = dec(outs, *meta)
        t_end = time.perf_counter()
        if out_dir is not None and k < len(ids):
            (out_dir / f"{i}.txt").write_text(dets_to_kitti(dets))
        with lock:
            rec["t_end"].append(t_end); rec["lat"].append((t_pre - t0 + t_end - t_sub) * 1000)   # 대기열에서 기다린 시간은 뺀다
            rec["npu"].append((t_cb - t_sub) * 1000); rec["pre"].append((t_pre - t0) * 1000)
            rec["post"].append((t_end - tp) * 1000); rec["n_det"] += len(dets)

    post_pool = ThreadPoolExecutor(a.post_workers)
    pre_pool = ThreadPoolExecutor(a.pre_workers)
    sem = threading.Semaphore(inflight)

    def on_done(outs_list, err, batch_items, t_sub):
        t_cb = time.perf_counter()
        sem.release()
        if err is not None:
            errors.append(err); return
        for (i, k, _, meta, t0, t_pre), outs in zip(batch_items, outs_list):
            if k >= 0:
                post_pool.submit(post_job, i, k, outs, meta, t0, t_pre, t_sub, t_cb)

    with ResourceMonitor() as mon:
        rss_base = mon._rss_mb() if mon.ok else None
        t_start = time.perf_counter()
        if a.mode == "sync":
            for item in frames:
                i, k, x, meta, t0, t_pre = pre_job(item)
                t_sub = time.perf_counter()
                outs = be.run_sync([x])[0]
                t_cb = time.perf_counter()
                post_job(i, k, outs, meta, t0, t_pre, t_sub, t_cb)
        else:
            it = iter(frames)
            look = deque()

            def fill():
                while len(look) < max(8, 2 * batch * inflight):
                    try:
                        look.append(pre_pool.submit(pre_job, next(it)))
                    except StopIteration:
                        break
            fill()
            while look and not errors:
                items = []
                while look and len(items) < batch:
                    items.append(look.popleft().result()); fill()
                while len(items) < batch:                         # 마지막 묶음은 앞 장으로 채우고 결과는 버린다
                    i, k, x, meta, t0, t_pre = items[0]
                    items.append((i, -1, x, meta, t0, t_pre))
                sem.acquire()
                t_sub = time.perf_counter()
                be.submit([x for _, _, x, _, _, _ in items], partial(on_done, batch_items=items, t_sub=t_sub))
            be.drain()
            for _ in range(inflight):                              # 남은 콜백까지 기다린다
                sem.acquire()
        post_pool.shutdown(wait=True)
        pre_pool.shutdown(wait=True)
        t_stop = time.perf_counter()
    desc = be.describe()
    be.close()
    del be
    gc.collect()
    if errors:
        raise RuntimeError(f"추론 오류: {errors[0]}")

    te = np.sort(np.array(rec["t_end"]))
    w = min(a.warmup, len(te) - 2)
    fps = (len(te) - 1 - w) / (te[-1] - te[w]) if len(te) > w + 1 else float("nan")
    lat, npu = np.array(rec["lat"][w:]), np.array(rec["npu"][w:])
    res = {"backend": a.backend, "model": Path(a.model).name, "imgsz": f"{W}x{H}", "mode": a.mode,
           "batch": batch, "inflight": inflight, "out_format": a.out_format, "source": a.source,
           "pre_workers": a.pre_workers, "post_workers": a.post_workers, "conf": a.conf,
           "frames": len(te), "warmup": w, "fps": round(fps, 2),
           "lat_p50_ms": round(pct(lat, 50), 2), "lat_p99_ms": round(pct(lat, 99), 2),
           "npu_p50_ms": round(pct(npu, 50), 2),
           "pre_mean_ms": round(float(np.mean(rec["pre"][w:])), 2),
           "post_mean_ms": round(float(np.mean(rec["post"][w:])), 2),
           "det_per_img": round(rec["n_det"] / max(len(te), 1), 2),
           "rss_base_mb": round(rss_base, 1) if rss_base else None,
           **mon.summary(te[w] if len(te) > w else None, t_stop),
           "wall_s": round(t_stop - t_start, 2), "device": desc, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["hailort", "onnx"], default="hailort")
    ap.add_argument("--model", required=True, help=".hef (hailort) 또는 head ONNX (onnx)")
    ap.add_argument("--imgsz", required=True, help="WxH, 예 1280x384")
    ap.add_argument("--images", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("-n", type=int, default=64, help="split 앞에서부터 쓸 이미지 수")
    ap.add_argument("--frames", type=int, default=None, help="처리할 총 장수 (n 장을 반복). 기본 n")
    ap.add_argument("--warmup", type=int, default=20, help="통계에서 뺄 앞쪽 완료 장수")
    ap.add_argument("--source", choices=["ram", "disk"], default="ram")
    ap.add_argument("--mode", choices=["async", "sync"], default="async")
    ap.add_argument("--batch", default="1", help="쉼표 목록, 예 1,2,4,8")
    ap.add_argument("--inflight", default="4", help="동시에 NPU 에 걸어둘 묶음 수, 쉼표 목록")
    ap.add_argument("--pre-workers", type=int, default=2)
    ap.add_argument("--post-workers", type=int, default=2)
    ap.add_argument("--out-format", choices=["float32", "raw"], default="float32",
                    help="raw = NPU 정수 출력 그대로 받고 필요한 값만 역양자화")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--reg-max", type=int, default=16)
    ap.add_argument("--check-decode", action="store_true", help="빠른 디코드가 06 디코드와 같은지 확인하고 끝낸다")
    ap.add_argument("--dump", default=None, help="KITTI 예측 txt 폴더 (조합 1개일 때만)")
    ap.add_argument("--json", default=None, help="결과를 jsonl 로 덧붙일 경로")
    a = ap.parse_args()

    import cv2
    W, H = parse_imgsz(a.imgsz)
    ids = Path(a.split).read_text().split()[:a.n]
    a.frames = a.frames or len(ids)
    batches = [int(v) for v in a.batch.split(",")]
    infl = [int(v) for v in a.inflight.split(",")]
    if a.mode == "sync":
        batches, infl = [1], [1]

    if a.check_decode:
        ok = True
        for fmt in (["float32", "raw"] if a.backend == "hailort" else ["float32"]):
            print(f"\n== out-format {fmt}")
            be = make_backend(a.backend, a.model, batch=1, out_format=fmt)
            print("  ", be.describe())
            ok &= check_decode(be, ids, a.images, W, H, a.conf, a.iou, a.nc, a.reg_max)
            be.close()
        print("\n결과:", "일치" if ok else "불일치 — 로그를 그대로 보내주세요")
        sys.exit(0 if ok else 1)

    imgs = None
    if a.source == "ram":
        imgs = {i: cv2.imread(str(find_image(a.images, i))) for i in ids}
    out_dir = None
    if a.dump:
        if len(batches) * len(infl) != 1:
            sys.exit("--dump 는 batch·inflight 조합 1개일 때만")
        out_dir = Path(a.dump); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{Path(a.model).name} {W}x{H} | {a.backend} {a.mode} | source={a.source} out={a.out_format} "
          f"conf={a.conf} | {a.frames}장 (warmup {a.warmup})")
    print(f"{'batch':>5}{'infl':>5}{'FPS':>8}{'lat50':>8}{'lat99':>8}{'npu50':>8}{'pre':>7}{'post':>7}"
          f"{'pCPU%':>7}{'sCPU%':>7}{'RSSmax':>8}{'base':>7}{'temp':>6}")
    for b in batches:
        for k in infl:
            r = run_config(a, W, H, ids, imgs, b, k, out_dir)
            print(f"{b:5d}{k:5d}{r['fps']:8.1f}{r['lat_p50_ms']:8.1f}{r['lat_p99_ms']:8.1f}{r['npu_p50_ms']:8.1f}"
                  f"{r['pre_mean_ms']:7.1f}{r['post_mean_ms']:7.1f}{r.get('proc_cpu_pct_mean', float('nan')):7.0f}"
                  f"{r.get('sys_cpu_pct_mean', float('nan')):7.0f}{r.get('rss_mb_max', float('nan')):8.0f}"
                  f"{(r['rss_base_mb'] or float('nan')):7.0f}{(r.get('temp_c_max') or float('nan')):6.1f}", flush=True)
            if a.json:
                Path(a.json).parent.mkdir(parents=True, exist_ok=True)
                with open(a.json, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print("throttled:", r.get("throttled_start"), "->", r.get("throttled_end"))
    print("  lat = 장당 지연 (전처리 + 제출~후처리 끝, 대기열 대기 제외), npu = 묶음 제출~완료, pre/post = 장당 CPU 시간 평균")
    if a.backend == "onnx":
        print("  * onnx backend 의 시간은 PC CPU 값이라 의미 없음 — 코드 경로 검증용")


if __name__ == "__main__":
    main()
