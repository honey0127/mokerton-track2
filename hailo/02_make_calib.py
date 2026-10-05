#!/usr/bin/env python3
"""Hailo optimize 용 캘리브레이션 세트(.npy) 생성.

캘리브레이션 전처리가 추론 전처리와 어긋나면 양자화 스케일이 틀어져서
정확도가 이유 없이 떨어진다. 그래서 Ultralytics 와 동일한 letterbox
(비율 유지 + 중앙 패딩 114) 를 그대로 재현한다.

출력은 uint8 NHWC (N, H, W, 3) 배열이다. 0~255 그대로 두고, 정규화는
Hailo 모델 스크립트(.alls)의 normalization 레이어가 담당한다 — 그래야
NPU 안에서 처리되고 호스트 전처리가 가벼워진다.

중요: 캘리브레이션 이미지는 **train 또는 내부 val 에서만** 뽑는다.
공식 eval 1,000장을 쓰면 라벨을 안 쓰더라도 분포 정보 유출이다.

사용 (내부 val 648장 전부):
  python 02_make_calib.py --image-dir ../yolo_dataset/images/val --out calib_648.npy

사용 (개수 제한 + 시드 고정):
  python 02_make_calib.py --image-dir ../yolo_dataset/images/val -n 256 --out calib_256.npy
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np


def letterbox(img, new_h, new_w, pad=114):
    """Ultralytics 와 동일한 비율 유지 리사이즈 + 중앙 패딩."""
    h, w = img.shape[:2]
    r = min(new_h / h, new_w / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = _resize(img, nh, nw)
    out = np.full((new_h, new_w, 3), pad, dtype=np.uint8)
    dh, dw = (new_h - nh) // 2, (new_w - nw) // 2
    out[dh:dh + nh, dw:dw + nw] = resized
    return out


def _resize(img, nh, nw):
    try:
        import cv2  # noqa: PLC0415
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        pass
    h, w = img.shape[:2]
    yi = (np.arange(nh) + 0.5) * h / nh - 0.5
    xi = (np.arange(nw) + 0.5) * w / nw - 0.5
    y0 = np.clip(np.floor(yi).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(xi).astype(int), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = np.clip(yi - y0, 0, 1)[:, None, None]
    wx = np.clip(xi - x0, 0, 1)[None, :, None]
    a = img[np.ix_(y0, x0)].astype(np.float32)
    b = img[np.ix_(y0, x1)].astype(np.float32)
    c = img[np.ix_(y1, x0)].astype(np.float32)
    d = img[np.ix_(y1, x1)].astype(np.float32)
    top = a * (1 - wx) + b * wx
    bot = c * (1 - wx) + d * wx
    return (top * (1 - wy) + bot * wy).astype(np.uint8)


def read_image(path):
    try:
        import cv2  # noqa: PLC0415
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"읽기 실패: {path}")
        return im[:, :, ::-1]            # BGR -> RGB
    except ImportError:
        pass
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        raise SystemExit("opencv-python 또는 pillow 가 필요합니다.")
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Hailo 캘리브레이션 npy 생성")
    ap.add_argument("--image-dir", required=True,
                    help="train 또는 내부 val 이미지 폴더. eval 폴더는 쓰지 말 것")
    ap.add_argument("--out", default="calib.npy")
    ap.add_argument("--imgsz", nargs=2, type=int, default=[384, 1280], metavar=("H", "W"))
    ap.add_argument("-n", "--num", type=int, default=0,
                    help="사용할 장수. 0 이면 전부")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-list", default=None,
                    help="지정하면 이 목록의 인덱스가 섞여있는지 검사하고 있으면 중단")
    args = ap.parse_args(argv)

    H, W = args.imgsz
    d = Path(args.image_dir)
    if not d.is_dir():
        raise SystemExit(f"폴더가 없습니다: {d}")

    files = sorted(p for p in d.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not files:
        raise SystemExit(f"이미지가 없습니다: {d}")

    # leakage 방어
    if args.eval_list:
        ev = {l.strip() for l in Path(args.eval_list).read_text().splitlines() if l.strip()}
        bad = sorted({f.stem for f in files} & ev)
        if bad:
            raise SystemExit(
                f"[중단] 캘리브레이션 폴더에 공식 eval 인덱스 {len(bad)}개가 있습니다.\n"
                f"  예: {bad[:5]}\n  train 또는 내부 val 폴더를 지정하세요."
            )
        print(f"  leakage 검사 통과 (eval {len(ev)}개와 교집합 없음)")

    if args.num and args.num < len(files):
        random.Random(args.seed).shuffle(files)
        files = sorted(files[:args.num])

    print(f"캘리브레이션 {len(files)}장, {W}x{H} letterbox 처리 중...")
    buf = np.empty((len(files), H, W, 3), dtype=np.uint8)
    for k, f in enumerate(files):
        buf[k] = letterbox(read_image(f), H, W)
        if (k + 1) % 100 == 0 or k + 1 == len(files):
            print(f"  {k + 1}/{len(files)}", flush=True)

    out = Path(args.out)
    np.save(out, buf)
    print(f"\n저장: {out}  shape={buf.shape} dtype={buf.dtype}  "
          f"({out.stat().st_size / 1e6:.1f} MB)")
    print("  값 범위 0~255 (uint8). 정규화는 .alls 의 normalization 레이어가 처리")

    # 재현성 기록
    lst = out.with_suffix(".filelist.txt")
    lst.write_text("\n".join(f.stem for f in files) + "\n")
    print(f"  사용 목록: {lst}  ← check_leakage.py --calib-list 에 넘길 파일")
    return 0


if __name__ == "__main__":
    sys.exit(main())
