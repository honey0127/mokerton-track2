#!/usr/bin/env python3
"""본선 라이브 시연: 카메라·영상·KITTI 이미지 → Hailo-10H 추론 → 박스와 성능 수치를 화면에 표시.

  # 카메라 (Raspberry Pi 카메라 모듈, picamera2)
  python3 hailo/08_demo.py --models hailo/hef/B_1280x384.hef,hailo/hef/D_640x192.hef --source picam --crop kitti
  # USB 웹캠 0번 / 영상 파일
  python3 hailo/08_demo.py --models hailo/hef/D_640x192.hef --source 0
  python3 hailo/08_demo.py --models hailo/hef/D_640x192.hef --source drive.mp4
  # 카메라가 없을 때: KITTI 이미지를 초당 10장으로 넘기며 (KITTI 원래 촬영 속도)
  python3 hailo/08_demo.py --models hailo/hef/B_1280x384.hef,hailo/hef/D_640x192.hef \
      --source kitti --images $IMG --split splits/holdout.txt --rate 10

키: q 종료 | m 모델 전환 (여러 개 줬을 때. 예: B 정확도 모드 <-> D 속도 모드) | s 화면 저장 | space 일시정지

화면 위 수치
  FPS  : 최근 1초 동안 결과가 나온 장수 (입력 속도보다 빠를 수 없다. 카메라 30fps 면 최대 30)
  NPU  : 제출~완료 ms,  E2E : 프레임 받은 순간~박스 그리기 직전 ms
  CPU  : 이 프로세스 CPU% (100% = 코어 1개),  RAM : RSS MB,  SoC : 온도
--headless 면 창 없이 --seconds 동안 돌고 요약만 출력한다 (보드 SSH 에서 확인용). --save 로 mp4 녹화.

모델 해상도(WxH)는 파일 이름의 1280x384 / 640x192 에서 읽는다. 다르면 --imgsz 로 순서대로 준다.
PC 검증: --backend onnx --models runs/head_D.onnx (make_head_onnx.py) --source kitti ... --headless
"""
from __future__ import annotations
import argparse, os, re, sys, threading, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hailo_rt import CLASSES, FastDecoder, ResourceMonitor, make_backend, preprocess  # noqa: E402

COLORS = {0: (0, 200, 255), 1: (80, 255, 80), 2: (255, 120, 40)}   # BGR: Car 주황, Pedestrian 초록, Cyclist 파랑
KITTI_ASPECT = 1242 / 375


# --------------------------------------------------------------------------- 입력
class Source:
    """가장 최근 프레임만 들고 있는 캡처 스레드. 추론이 느리면 오래된 프레임은 버린다 (실시간 시연 방식)."""

    def __init__(self, a):
        self.a, self.frame, self.seq, self.done = a, None, 0, False
        self.cv = threading.Condition()
        self.paused = False
        src = a.source
        if src == "picam":
            from picamera2 import Picamera2
            self.cam = Picamera2()
            w, h = (int(v) for v in a.cam_size.split("x"))
            self.cam.configure(self.cam.create_video_configuration(main={"size": (w, h), "format": "RGB888"}))
            self.cam.start()
            self.read = lambda: self.cam.capture_array()        # RGB888 = 메모리상 B,G,R 순서 (OpenCV 와 같다)
        elif src == "kitti":
            import cv2
            ids = Path(a.split).read_text().split()
            self.paths = [p for i in ids for p in [Path(a.images) / f"{i}.png"] if p.exists()]
            if not self.paths:
                sys.exit("이미지를 찾지 못했다 (--images, --split 확인)")
            self.k = 0

            def rd():
                im = cv2.imread(str(self.paths[self.k % len(self.paths)])); self.k += 1
                return im
            self.read = rd
        else:
            import cv2
            self.is_file = not src.isdigit()
            self.cap = cv2.VideoCapture(int(src) if not self.is_file else src)
            if not self.cap.isOpened():
                sys.exit(f"입력을 열 수 없다: {src}")
            if self.is_file and a.rate == 0:                 # 영상은 원래 속도로 재생
                a.rate = self.cap.get(cv2.CAP_PROP_FPS) or 10

            def rd():
                ok, im = self.cap.read()
                if not ok and self.is_file:                  # 끝나면 다시 연다 (seek 은 ffmpeg 스레드 디코더에서 죽는다)
                    self.cap.release(); self.cap = cv2.VideoCapture(src); ok, im = self.cap.read()
                return im if ok else None
            self.read = rd
        self.th = threading.Thread(target=self._loop, daemon=True); self.th.start()

    def crop(self, im):
        if self.a.crop != "kitti":
            return im
        h, w = im.shape[:2]
        if w / h >= KITTI_ASPECT:
            return im
        ch = int(round(w / KITTI_ASPECT))
        y0 = int((h - ch) * self.a.crop_y)
        return im[y0:y0 + ch]

    def _loop(self):
        period = 1.0 / self.a.rate if self.a.rate > 0 else 0
        nxt = time.perf_counter()
        while not self.done:
            if self.paused:
                time.sleep(0.05); continue
            im = self.read()
            if im is None:
                time.sleep(0.01); continue
            im = np.ascontiguousarray(self.crop(im))
            with self.cv:
                self.frame, self.t_cap = im, time.perf_counter()
                self.seq += 1
                self.cv.notify_all()
            if period:
                nxt += period
                time.sleep(max(0.0, nxt - time.perf_counter()))

    def get(self, last_seq, timeout=1.0):
        with self.cv:
            self.cv.wait_for(lambda: self.seq != last_seq or self.done, timeout)
            return self.seq, self.frame, getattr(self, "t_cap", 0.0)

    def close(self):
        self.done = True
        if hasattr(self, "cam"):
            self.cam.stop()
        if hasattr(self, "cap"):
            self.cap.release()


# --------------------------------------------------------------------------- 모델
def imgsz_of(path, override):
    if override:
        return override
    m = re.search(r"(\d+)x(\d+)", Path(path).name)
    if not m:
        sys.exit(f"{path}: 파일 이름에서 해상도를 읽을 수 없다 — --imgsz 로 주세요")
    return int(m.group(1)), int(m.group(2))


class Runner:
    """모델 하나 = backend + 디코더. 동시에 inflight 장까지 NPU 에 걸어둔다."""

    def __init__(self, a, path, wh):
        self.path, (self.W, self.H) = path, wh
        self.be = make_backend(a.backend, path, batch=1, out_format=a.out_format)
        self.dec = FastDecoder(self.W, self.H, conf=a.conf, iou=a.iou)
        self.sem = threading.Semaphore(a.inflight)
        self.post = ThreadPoolExecutor(2)
        self.label = Path(path).stem

    def submit(self, frame, t_cap, on_result):
        x, meta = preprocess(frame, self.W, self.H)
        self.sem.acquire()
        t_sub = time.perf_counter()

        def done(outs_list, err):
            t_cb = time.perf_counter()
            self.sem.release()
            if err is not None:
                print("추론 오류:", err, file=sys.stderr); return
            self.post.submit(lambda: on_result(frame, self.dec(outs_list[0], *meta), t_cap, t_sub, t_cb, self.label))
        self.be.submit([x], done)

    def close(self):
        self.be.close(); self.post.shutdown(wait=True)


# --------------------------------------------------------------------------- 그리기
def draw(im, dets, hud):
    import cv2
    out = im.copy()
    for c, s, x1, y1, x2, y2 in dets:
        col = COLORS.get(c, (255, 255, 255))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, col, 2)
        txt = f"{CLASSES[c] if c < len(CLASSES) else c} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), col, -1)
        cv2.putText(out, txt, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    y = 22
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 560), 8 + 22 * len(hud)), (0, 0, 0), -1)
    for line in hud:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="쉼표로 여러 개 (m 키로 전환)")
    ap.add_argument("--imgsz", default=None, help="모델별 WxH 쉼표 목록 (파일 이름에서 못 읽을 때)")
    ap.add_argument("--backend", choices=["hailort", "onnx"], default="hailort")
    ap.add_argument("--source", default="picam", help="picam | kitti | 웹캠 번호 | 영상 파일")
    ap.add_argument("--images"); ap.add_argument("--split")
    ap.add_argument("--rate", type=float, default=0, help="입력 프레임 속도 상한 (kitti 는 10 권장, 0 = 제한 없음)")
    ap.add_argument("--cam-size", default="1536x864", help="picam 해상도")
    ap.add_argument("--crop", choices=["none", "kitti"], default="none",
                    help="kitti = 가운데 띠를 KITTI 비율(1242:375)로 잘라 letterbox 낭비를 없앤다")
    ap.add_argument("--crop-y", type=float, default=0.5, help="자를 띠의 세로 위치 (0 위 ~ 1 아래)")
    ap.add_argument("--inflight", type=int, default=3)
    ap.add_argument("--out-format", choices=["float32", "raw"], default="float32")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=20, help="--headless 일 때 실행 시간")
    ap.add_argument("--save", default=None, help="mp4 녹화 경로")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--stats-csv", default=None, help="0.5초마다 CPU·RSS·온도·누적 장수 기록 (발열 장시간 시험용)")
    a = ap.parse_args()

    import cv2
    paths = a.models.split(",")
    sizes = [tuple(int(v) for v in s.split("x")) for s in a.imgsz.split(",")] if a.imgsz else [None] * len(paths)
    if a.headless is False and not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        print("DISPLAY 가 없어 --headless 로 실행한다 (보드 화면에서 돌리거나 ssh -X)")
        a.headless = True

    src = Source(a)
    cur = 0
    runner = Runner(a, paths[cur], imgsz_of(paths[cur], sizes[cur]))
    latest = {"img": None, "dets": [], "t": 0.0}
    done_t, npu_ms, e2e_ms = deque(maxlen=300), deque(maxlen=60), deque(maxlen=60)
    lock = threading.Lock()
    stats = {"n": 0, "det": 0}

    def on_result(frame, dets, t_cap, t_sub, t_cb, label):
        t = time.perf_counter()
        with lock:
            if t_cap >= latest["t"]:                       # 순서가 뒤바뀐 결과는 버린다
                latest.update(img=frame, dets=dets, t=t_cap, label=label)
            done_t.append(t); npu_ms.append((t_cb - t_sub) * 1000); e2e_ms.append((t - t_cap) * 1000)
            stats["n"] += 1; stats["det"] += len(dets)

    writer, win = None, "mokerton-track2 demo"
    if not a.headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        if a.fullscreen:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    feeder_stop = threading.Event()
    state = {"runner": runner}

    def feeder():                                          # 새 프레임이 오면 바로 NPU 로 (inflight 만큼 겹침)
        seq = 0
        while not feeder_stop.is_set():
            seq, fr, t_cap = src.get(seq, 0.5)
            if fr is None or feeder_stop.is_set():
                continue
            state["runner"].submit(fr, t_cap, on_result)
    fth = threading.Thread(target=feeder, daemon=True)

    with ResourceMonitor(period=0.5, counter=lambda: stats["n"]) as mon:
        t_start = time.perf_counter()
        fth.start()
        while True:
            now = time.perf_counter()
            with lock:
                img, dets, label = latest["img"], latest["dets"], latest.get("label", "")
                fps = sum(1 for t in done_t if now - t <= 1.0)
                npu = np.median(npu_ms) if npu_ms else float("nan")
                e2e = np.median(e2e_ms) if e2e_ms else float("nan")
            r = mon.samples[-1] if mon.samples else {}
            hud = [f"{label}  {state['runner'].W}x{state['runner'].H}  INT8 Hailo-10H" if a.backend == "hailort"
                   else f"{label}  {state['runner'].W}x{state['runner'].H}  (ONNX CPU test)",
                   f"FPS {fps:3d}   NPU {npu:5.1f} ms   E2E {e2e:5.1f} ms   det {len(dets)}",
                   f"CPU {r.get('proc_cpu_pct', float('nan')):4.0f}%   RAM {r.get('rss_mb', float('nan')):4.0f} MB   "
                   f"SoC {r.get('temp_c', float('nan')):4.1f} C"]
            if img is not None:
                vis = draw(img, dets, hud)
                if a.save:
                    if writer is None:
                        writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                                 (vis.shape[1], vis.shape[0]))
                    writer.write(vis)
                if not a.headless:
                    cv2.imshow(win, vis)
            if a.headless:
                time.sleep(1 / 15)
                if now - t_start > a.seconds:
                    break
                continue
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                src.paused = not src.paused
            if k == ord("s") and img is not None:
                fn = f"demo_{time.strftime('%H%M%S')}.png"; cv2.imwrite(fn, vis); print("저장:", fn)
            if k == ord("m") and len(paths) > 1:
                cur = (cur + 1) % len(paths)
                old = state["runner"]
                state["runner"] = Runner(a, paths[cur], imgsz_of(paths[cur], sizes[cur]))
                old.close()
                with lock:
                    done_t.clear(); npu_ms.clear(); e2e_ms.clear()
                print("모델 전환:", paths[cur])
        feeder_stop.set(); src.close(); fth.join(timeout=2)
        state["runner"].close()
        t_end = time.perf_counter()
    if writer is not None:
        writer.release()
    if not a.headless:
        cv2.destroyAllWindows()
    if a.stats_csv:
        import csv
        with open(a.stats_csv, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["sec", "fps_0.5s", "proc_cpu_pct", "sys_cpu_pct", "rss_mb", "temp_c"])
            prev_t, prev_n = t_start, 0
            for r in mon.samples:
                n = r.get("n_done", 0)
                w.writerow([round(r["t"] - t_start, 1), round((n - prev_n) / max(r["t"] - prev_t, 1e-6), 1),
                            round(r["proc_cpu_pct"], 1), round(r["sys_cpu_pct"], 1), round(r["rss_mb"], 1), r["temp_c"]])
                prev_t, prev_n = r["t"], n
        print("기록:", a.stats_csv)
    s = mon.summary()
    print(f"[demo] {Path(paths[cur]).name} | {stats['n']}장 / {t_end - t_start:.1f}s = {stats['n'] / (t_end - t_start):.1f} FPS "
          f"| NPU p50 {np.median(npu_ms) if npu_ms else float('nan'):.1f} ms | E2E p50 "
          f"{np.median(e2e_ms) if e2e_ms else float('nan'):.1f} ms | 장당 검출 {stats['det'] / max(stats['n'], 1):.1f}")
    print(f"       CPU {s.get('proc_cpu_pct_mean')}% (코어 1개=100) | 시스템 {s.get('sys_cpu_pct_mean')}% | "
          f"RSS max {s.get('rss_mb_max')} MB | SoC max {s.get('temp_c_max')} C | throttled {s.get('throttled_end')}")


if __name__ == "__main__":
    main()
