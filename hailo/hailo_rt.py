"""본선 보드 실행용 공용 모듈 — 07_board_fast.py (측정) 와 08_demo.py (시연) 가 같이 쓴다.

06_board_e2e.py 와 달라진 점
  1. 비동기 추론: 여러 장을 NPU 에 겹쳐 넣는다 (run_async + 동시 진행 수 제한). 06 은 한 장씩 run() 이라
     NPU 가 CPU 전·후처리를 기다리며 논다. batch 도 바꿀 수 있다 (multi-context 모델은 batch 가 클수록
     context 전환 비용이 여러 장에 나뉜다).
  2. 빠른 디코드: 클래스 점수(logit)로 먼저 거르고 남은 앵커만 DFL 을 푼다. 결과는 04/06 의
     decode_heads + common.decode_yolo 와 같다 (PC 에서 확인: make_head_onnx.py + 07 --check-decode).
  3. raw 출력: --out-format raw 면 NPU 출력을 양자화된 정수 그대로 받아, 거른 뒤 남은 값만 역양자화한다.
  4. 자원 측정: 프로세스 RSS·CPU%, 시스템 CPU%, SoC 온도, throttled 를 0.25초마다 기록한다.

backend
  hailort : 보드 (hailo_platform, HailoRT 5.x InferModel API)
  onnx    : PC 검증용. head 출력까지 자른 ONNX (make_head_onnx.py) 를 ONNX Runtime 으로 돌린다.
            출력 형식이 HEF 와 같아서 디코드·파이프라인 코드를 보드 없이 확인할 수 있다. 시간은 의미 없음.
"""
from __future__ import annotations
import os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "scripts"))
from common import CLASSES, decode_yolo, letterbox, nms  # noqa: E402,F401


# --------------------------------------------------------------------------- 디코드
class FastDecoder:
    """head 출력 [(h,w,c) ...] -> [(cls, score, x1,y1,x2,y2)] (원본 좌표).

    outs 의 각 원소는 float 배열이거나 (정수 배열, scale, zero_point) 튜플(raw 출력)이다.
    """

    def __init__(self, in_w, in_h, nc=3, reg_max=16, conf=0.25, iou=0.65, max_det=300):
        self.W, self.H, self.nc, self.R = in_w, in_h, nc, reg_max
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.logit_thr = float(np.log(conf / (1.0 - conf)))
        self.proj = np.arange(reg_max, dtype=np.float32)
        self._grid = {}

    def grid(self, h, w):
        g = self._grid.get((h, w))
        if g is None:
            ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            g = (xs.reshape(-1).astype(np.float32) + 0.5, ys.reshape(-1).astype(np.float32) + 0.5)
            self._grid[(h, w)] = g
        return g

    @staticmethod
    def _split(o):
        if isinstance(o, tuple):
            return o
        return o, None, None

    def heads(self, outs):
        """stride 별 (box, cls) 짝으로 정리. 04_emulate_eval.decode_heads 와 같은 판정 규칙."""
        by = {}
        for o in outs:
            arr, sc, zp = self._split(o)
            h, w, c = arr.shape[-3:]
            s = self.H // h
            if self.W // w != s:
                raise ValueError(f"출력 {arr.shape} 의 stride 판정 불가 (입력 {self.H}x{self.W})")
            key = "box" if c == 4 * self.R else "cls" if c == self.nc else None
            if key is None:
                raise ValueError(f"채널 {c} 는 box({4*self.R})/cls({self.nc}) 가 아니다")
            by.setdefault(s, {})[key] = (arr.reshape(h * w, c), sc, zp, h, w)
        return [(s, by[s]) for s in sorted(by)]

    def candidates(self, outs):
        """conf 를 넘는 앵커만 (4+nc, K) 로. 앵커 순서는 decode_heads 와 같다."""
        cols = []
        for s, slot in self.heads(outs):
            cls, csc, czp, h, w = slot["cls"]
            box, bsc, bzp, _, _ = slot["box"]
            if csc is None:
                m = cls.max(1) > self.logit_thr
            else:                                   # 정수 영역에서 비교: (q - zp) * scale > thr
                m = (cls.max(1).astype(np.float32) - czp) * csc > self.logit_thr
            idx = np.flatnonzero(m)
            if idx.size == 0:
                continue
            c = cls[idx].astype(np.float32)
            if csc is not None:
                c = (c - czp) * csc
            b = box[idx].astype(np.float32)
            if bsc is not None:
                b = (b - bzp) * bsc
            b = b.reshape(-1, 4, self.R)
            b = np.exp(b - b.max(-1, keepdims=True))
            ltrb = (b / b.sum(-1, keepdims=True)) @ self.proj           # (K, 4)
            ax, ay = self.grid(h, w)
            ax, ay = ax[idx], ay[idx]
            x1, y1 = ax - ltrb[:, 0], ay - ltrb[:, 1]
            x2, y2 = ax + ltrb[:, 2], ay + ltrb[:, 3]
            xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 0) * s
            cols.append(np.concatenate([xywh, (1.0 / (1.0 + np.exp(-c))).T], 0))
        if not cols:
            return np.zeros((4 + self.nc, 0), np.float32)
        return np.concatenate(cols, 1)

    def __call__(self, outs, r, px, py, oh, ow):
        """common.decode_yolo 의 뒷부분과 같은 계산. decode_yolo 를 직접 부르지 않는 이유:
        후보가 7개 미만이면 (4+nc, K) 를 (A, 4+nc) 로 오인해 전치한다."""
        p = self.candidates(outs)
        if p.shape[1] == 0:
            return []
        conf, cid = p[4:].max(0), p[4:].argmax(0)
        m = conf > self.conf
        if not m.any():
            return []
        box, conf, cid = p[:4, m], conf[m], cid[m]
        cx, cy, w, h = box
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - px) / r).clip(0, ow)
        xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - py) / r).clip(0, oh)
        res = []
        for c in range(self.nc):
            s = cid == c
            if not s.any():
                continue
            b, sc = xyxy[s], conf[s]
            for i in nms(b, sc, self.iou):
                res.append((c, float(sc[i]), *[float(v) for v in b[i]]))
        res.sort(key=lambda t: -t[1])
        return res[:self.max_det]


# --------------------------------------------------------------------------- backend
_NP = {"UINT8": np.uint8, "UINT16": np.uint16, "FLOAT32": np.float32}


class HailoRTBackend:
    """HailoRT 5.x InferModel. submit(frames, cb) 는 비동기, cb(list_of_outs) 는 HailoRT 스레드에서 불린다."""

    name = "hailort"

    def __init__(self, hef, batch=1, out_format="float32", timeout_ms=10000):
        from hailo_platform import HEF, VDevice, FormatType
        self.timeout_ms, self.batch = timeout_ms, batch
        self.vdevice = VDevice(VDevice.create_params())
        self.model = self.vdevice.create_infer_model(str(hef))
        self.model.set_batch_size(batch)
        self.model.input().set_format_type(FormatType.UINT8)
        infos = {i.name: i for i in HEF(str(hef)).get_output_vstream_infos()}
        self.out_names, self.out_spec = [], {}
        for o in self.model.outputs:
            nm = o.name
            if out_format == "raw":
                ft = infos[nm].format.type
                o.set_format_type(ft)
                qi = infos[nm].quant_info
                spec = (_NP[ft.name], float(qi.qp_scale), float(qi.qp_zp))
            else:
                o.set_format_type(FormatType.FLOAT32)
                spec = (np.float32, None, None)
            self.out_names.append(nm)
            self.out_spec[nm] = (tuple(self.model.output(nm).shape),) + spec
        self.in_shape = tuple(self.model.input().shape)
        self._cm = self.model.configure()
        self.cfg = self._cm.__enter__()
        try:
            self.queue_size = int(self.cfg.get_async_queue_size())
        except Exception:
            self.queue_size = None
        self.last_job = None

    def describe(self):
        return {"input": self.in_shape, "outputs": {n: (s[0], s[1].__name__, s[2], s[3]) for n, s in self.out_spec.items()},
                "async_queue_size": self.queue_size, "batch": self.batch}

    def _bind(self, frame):
        bufs = {n: np.empty(sh, dt) for n, (sh, dt, _, _) in self.out_spec.items()}
        b = self.cfg.create_bindings(output_buffers=bufs)
        b.input().set_buffer(np.ascontiguousarray(frame))
        return b, bufs

    def _wrap(self, bufs):
        outs = []
        for n in self.out_names:
            _, _, sc, zp = self.out_spec[n]
            outs.append(bufs[n] if sc is None else (bufs[n], sc, zp))
        return outs

    def submit(self, frames, cb):
        pairs = [self._bind(f) for f in frames]
        self.cfg.wait_for_async_ready(timeout_ms=self.timeout_ms, frames_count=len(pairs))

        def done(completion_info, pairs=pairs):
            if completion_info.exception:
                cb(None, completion_info.exception)
                return
            cb([self._wrap(bufs) for _, bufs in pairs], None)

        self.last_job = self.cfg.run_async([b for b, _ in pairs], done)

    def run_sync(self, frames):
        pairs = [self._bind(f) for f in frames]
        self.cfg.run([b for b, _ in pairs], self.timeout_ms)
        return [self._wrap(bufs) for _, bufs in pairs]

    def drain(self):
        if self.last_job is not None:
            self.last_job.wait(self.timeout_ms)

    def close(self):
        self.drain()
        self._cm.__exit__(None, None, None)
        self.vdevice.release()


class OnnxHeadsBackend:
    """PC 검증용. head ONNX (NCHW 출력) -> HWC 로 바꿔 HEF 와 같은 모양으로 준다."""

    name = "onnx"

    def __init__(self, onnx_path, batch=1, workers=2, threads=2):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.batch, self.queue_size = batch, None
        self.pool = ThreadPoolExecutor(workers)
        self.pending = []
        _, _, h, w = self.sess.get_inputs()[0].shape
        self.in_shape = (h, w, 3)

    def describe(self):
        return {"input": self.in_shape, "outputs": self.out_names, "batch": self.batch}

    def _run(self, frames):
        x = np.stack([f[:, :, :].transpose(2, 0, 1) for f in frames]).astype(np.float32) / 255.0
        res = []
        for i in range(len(frames)):           # head ONNX 는 batch 1 고정으로 export 되어 있다
            o = self.sess.run(None, {self.inp: x[i:i + 1]})
            res.append([np.ascontiguousarray(a[0].transpose(1, 2, 0)) for a in o])
        return res

    def submit(self, frames, cb):
        def job():
            try:
                cb(self._run(frames), None)
            except Exception as e:  # noqa: BLE001
                cb(None, e)
        self.pending.append(self.pool.submit(job))

    def run_sync(self, frames):
        return self._run(frames)

    def drain(self):
        for f in self.pending:
            f.result()
        self.pending.clear()

    def close(self):
        self.drain()
        self.pool.shutdown()


def make_backend(kind, path, batch=1, out_format="float32"):
    if kind == "hailort":
        return HailoRTBackend(path, batch=batch, out_format=out_format)
    return OnnxHeadsBackend(path, batch=batch)


def preprocess(img_bgr, W, H):
    """letterbox + BGR->RGB. 정규화(/255)는 HEF 안에서 한다 (model_script.alls 의 normalization)."""
    lb, r, px, py = letterbox(img_bgr, W, H)
    return np.ascontiguousarray(lb[:, :, ::-1]), (r, px, py, img_bgr.shape[0], img_bgr.shape[1])


def dets_to_kitti(dets):
    lines = [f"{CLASSES[c]} -1 -1 -10 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
             f"-1000 -1000 -1000 -1000 -1000 -1000 -10 {s:.6f}"
             for c, s, x1, y1, x2, y2 in dets if c < len(CLASSES)]
    return "\n".join(lines) + ("\n" if lines else "")


# --------------------------------------------------------------------------- 자원 측정
class ResourceMonitor:
    """0.25초마다 /proc·sysfs 에서 읽는다 (Linux 전용, 다른 OS 면 비어 있는 결과)."""

    def __init__(self, period=0.25, counter=None):
        self.period, self.samples, self.counter = period, [], counter   # counter(): 지금까지 처리한 장수 (선택)
        self._stop = threading.Event()
        self.ok = Path("/proc/self/stat").exists()
        self.clk = os.sysconf("SC_CLK_TCK") if self.ok else 100

    @staticmethod
    def _sys_cpu():
        v = [int(x) for x in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
        idle = v[3] + v[4]
        return sum(v), idle

    def _proc_cpu(self):
        f = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        return (int(f[11]) + int(f[12])) / self.clk           # utime + stime (초)

    @staticmethod
    def _rss_mb():
        for ln in Path("/proc/self/status").read_text().splitlines():
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) / 1024
        return float("nan")

    @staticmethod
    def _temp():
        p = Path("/sys/class/thermal/thermal_zone0/temp")
        return int(p.read_text()) / 1000 if p.exists() else float("nan")

    @staticmethod
    def throttled():
        try:
            return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True,
                                  timeout=2).stdout.strip()
        except Exception:
            return "n/a"

    def _loop(self):
        tot0, idle0 = self._sys_cpu(); pc0 = self._proc_cpu(); t0 = time.perf_counter()
        while not self._stop.wait(self.period):
            tot1, idle1 = self._sys_cpu(); pc1 = self._proc_cpu(); t1 = time.perf_counter()
            dt = max(t1 - t0, 1e-6)
            self.samples.append({
                "t": t1,
                "proc_cpu_pct": 100 * (pc1 - pc0) / dt,                    # 100% = 코어 1개
                "sys_cpu_pct": 100 * (1 - (idle1 - idle0) / max(tot1 - tot0, 1)),   # 100% = 전체 코어
                "rss_mb": self._rss_mb(), "temp_c": self._temp(),
                **({"n_done": self.counter()} if self.counter else {})})
            tot0, idle0, pc0, t0 = tot1, idle1, pc1, t1

    def __enter__(self):
        self.throttled_start = self.throttled()
        if self.ok:
            self._th = threading.Thread(target=self._loop, daemon=True); self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self.ok:
            self._th.join()
        self.throttled_end = self.throttled()

    def summary(self, t_from=None, t_to=None):
        s = [x for x in self.samples if (t_from is None or x["t"] >= t_from) and (t_to is None or x["t"] <= t_to)]
        if not s:
            return {"n_samples": 0, "throttled_start": getattr(self, "throttled_start", "n/a"),
                    "throttled_end": getattr(self, "throttled_end", "n/a")}
        a = lambda k: np.array([x[k] for x in s], float)   # noqa: E731
        return {"n_samples": len(s),
                "proc_cpu_pct_mean": round(a("proc_cpu_pct").mean(), 1),
                "sys_cpu_pct_mean": round(a("sys_cpu_pct").mean(), 1),
                "rss_mb_max": round(np.nanmax(a("rss_mb")), 1),
                "temp_c_max": round(np.nanmax(a("temp_c")), 1) if np.isfinite(a("temp_c")).any() else None,
                "throttled_start": getattr(self, "throttled_start", "n/a"),
                "throttled_end": getattr(self, "throttled_end", "n/a")}
