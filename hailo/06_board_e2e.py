#!/usr/bin/env python3
"""Pi 5 + Hailo-10H end-to-end 구간별 지연 측정 (읽기 / 전처리 / NPU 추론 / 디코드 / NMS).

`hailortcli run` 은 NPU 단독 FPS 만 잰다. 실제 시연 속도는 Pi CPU 에서 도는 전처리(letterbox)와
후처리(DFL 디코드 + NMS)까지 합친 값이라, 이 스크립트로 구간을 나눠 어디가 병목인지 본다.

디코드는 04_emulate_eval.py 의 decode_heads, NMS 는 scripts/common.py 의 decode_yolo 를 그대로 쓴다
(에뮬레이터 채점과 같은 코드 → --dump 결과를 PC 에서 kitti_eval 로 채점하면 보드 정확도도 확인된다).

backend
  hailort  : 보드. hailo_platform (HailoRT Python) 의 InferModel API
  emulator : PC 에서 코드 경로 검증용. Hailo SDK 에뮬레이터 (--har 필요). 시간 수치는 의미 없음

예 (보드):
  python3 hailo/06_board_e2e.py --backend hailort --hef hailo/hef/A_1280x384.hef --imgsz 1280x384 \
      --images <holdout 이미지 폴더> --split splits/holdout.txt -n 200 --conf 0.25
  # 정확도 교차 확인용 덤프 (AP 용이라 conf 낮게):
  python3 hailo/06_board_e2e.py ... --conf 0.001 -n 500 --dump runs/board_pred_A
  # PC 에서: python scripts/kitti_eval.py --gt <label_2> --pred runs/board_pred_A --split splits/holdout.txt

예 (PC 검증):
  python 06_board_e2e.py --backend emulator --har D_640x192_optimized.har --imgsz 640x192 ... -n 16
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "scripts"))
from common import CLASSES, letterbox, decode_yolo, parse_imgsz  # noqa: E402

_spec = importlib.util.spec_from_file_location("emu04", HERE / "04_emulate_eval.py")
_emu = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_emu)
decode_heads = _emu.decode_heads


class HailoRTBackend:
    """HailoRT InferModel API (HailoRT 4.18+/5.x). 입력 uint8 NHWC, 출력은 float32 로 역양자화해서 받는다."""

    def __init__(self, hef, timeout_ms=10000):
        from hailo_platform import VDevice, FormatType
        self.timeout_ms = timeout_ms
        self.vdevice = VDevice(VDevice.create_params())
        self.model = self.vdevice.create_infer_model(str(hef))
        self.model.set_batch_size(1)
        self.model.input().set_format_type(FormatType.UINT8)
        for o in self.model.outputs:
            o.set_format_type(FormatType.FLOAT32)
        self.cfg = self.model.configure().__enter__()
        self.bind = self.cfg.create_bindings()
        self.outs = {}
        for o in self.model.outputs:
            buf = np.empty(self.model.output(o.name).shape, dtype=np.float32)
            self.bind.output(o.name).set_buffer(buf)
            self.outs[o.name] = buf
        print("  입력:", self.model.input().shape, " 출력:", {k: v.shape for k, v in self.outs.items()})

    def infer(self, x_hwc_uint8):
        self.bind.input().set_buffer(np.ascontiguousarray(x_hwc_uint8))
        self.cfg.run([self.bind], self.timeout_ms)
        return list(self.outs.values())          # [(h, w, c), ...]

    def close(self):
        self.cfg.__exit__(None, None, None)
        self.vdevice.release()


class EmulatorBackend:
    def __init__(self, har):
        from hailo_sdk_client import ClientRunner, InferenceContext
        self.runner = ClientRunner(har=str(har))
        self._cm = self.runner.infer_context(InferenceContext.SDK_QUANTIZED)
        self.ctx = self._cm.__enter__()

    def infer(self, x_hwc_uint8):
        res = self.runner.infer(self.ctx, x_hwc_uint8[None].astype(np.float32))
        res = [res] if isinstance(res, np.ndarray) else res
        return [np.asarray(o[0]) for o in res]

    def close(self):
        self._cm.__exit__(None, None, None)


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hailort", "emulator"], default="hailort")
    ap.add_argument("--hef"); ap.add_argument("--har")
    ap.add_argument("--images", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--imgsz", required=True, help="WxH, 예 1280x384")
    ap.add_argument("-n", type=int, default=200, help="측정 장수 (split 앞에서부터)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="시연 속도는 0.25 근처, AP 채점용 덤프는 0.001 (후처리 시간이 크게 달라진다)")
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--reg-max", type=int, default=16)
    ap.add_argument("--dump", default=None, help="KITTI 예측 txt 저장 폴더 (PC 에서 kitti_eval 로 채점)")
    ap.add_argument("--json", default=None, help="구간별 통계 저장 경로")
    a = ap.parse_args()

    import cv2
    W, H = parse_imgsz(a.imgsz)
    ids = Path(a.split).read_text().split()[:a.n]
    if a.backend == "hailort":
        if not a.hef: sys.exit("--hef 필요")
        be = HailoRTBackend(a.hef)
    else:
        if not a.har: sys.exit("--har 필요")
        be = EmulatorBackend(a.har)
    out = Path(a.dump) if a.dump else None
    if out: out.mkdir(parents=True, exist_ok=True)

    stages = ("read", "pre", "infer", "decode", "nms")
    t = {k: [] for k in stages}
    n_det = 0
    seq = ids[:min(a.warmup, len(ids))] + ids          # 앞 warmup 장은 측정에서 뺀다
    t_all0 = None
    for k, i in enumerate(seq):
        measure = k >= a.warmup or a.warmup == 0
        if measure and t_all0 is None:
            t_all0 = time.perf_counter()
        t0 = time.perf_counter()
        im = cv2.imread(str(Path(a.images) / f"{i}.png"))
        if im is None: sys.exit(f"이미지 없음: {i}")
        oh, ow = im.shape[:2]
        t1 = time.perf_counter()
        lb, r, px, py = letterbox(im, W, H)
        x = np.ascontiguousarray(lb[:, :, ::-1])            # RGB, uint8 NHWC — 정규화는 HEF 안에서
        t2 = time.perf_counter()
        outs = be.infer(x)
        t3 = time.perf_counter()
        p = decode_heads([o.reshape(o.shape[-3:]) for o in outs], H, W, a.nc, a.reg_max)
        t4 = time.perf_counter()
        dets = decode_yolo(p[None], r, px, py, oh, ow, a.conf, a.iou)
        t5 = time.perf_counter()
        if not measure:
            continue
        for s, dt in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
            t[s].append(dt * 1000)
        n_det += len(dets)
        if out:
            lines = [f"{CLASSES[c]} -1 -1 -10 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                     f"-1000 -1000 -1000 -1000 -1000 -1000 -10 {s:.6f}"
                     for c, s, x1, y1, x2, y2 in dets if c < len(CLASSES)]
            (out / f"{i}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    wall = time.perf_counter() - t_all0
    be.close()

    n = len(t["infer"])
    tot = np.sum([t[s] for s in stages], axis=0)
    print(f"\n[{a.backend}] {Path(a.hef or a.har).name}  {W}x{H}  {n}장 (warmup {a.warmup} 제외)  conf={a.conf}")
    print(f"{'구간':<8}{'mean':>9}{'p50':>9}{'p99':>9}  ms   비중")
    for s in stages:
        v = np.array(t[s])
        print(f"{s:<8}{v.mean():9.2f}{pct(v,50):9.2f}{pct(v,99):9.2f}      {100*v.sum()/tot.sum():5.1f}%")
    print(f"{'total':<8}{tot.mean():9.2f}{pct(tot,50):9.2f}{pct(tot,99):9.2f}")
    print(f"직렬 처리량: {n / wall:.1f} FPS  (NPU 단독 상한: {1000 / np.mean(t['infer']):.1f} FPS)  "
          f"장당 검출 {n_det / max(n,1):.1f}개")
    if a.backend == "emulator":
        print("  * emulator 시간은 PC 에뮬레이션 속도라 의미 없음 — 코드 경로 검증용")
    if a.json:
        rec = {"backend": a.backend, "model": Path(a.hef or a.har).name, "imgsz": f"{W}x{H}", "n": n,
               "conf": a.conf, "fps_serial": n / wall,
               **{f"{s}_{q}": round(f(np.array(t[s])), 3) for s in stages
                  for q, f in (("mean", np.mean), ("p50", lambda v: pct(v, 50)), ("p99", lambda v: pct(v, 99)))},
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        Path(a.json).write_text(json.dumps(rec, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
