import os, glob, argparse
import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner

def letterbox(path, size=640, color=(114, 114, 114)):
    """YOLO 전처리와 동일: 종횡비 유지 + 중앙 패딩 (PIL 구현)"""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), color)
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return np.asarray(canvas)

ap = argparse.ArgumentParser()
ap.add_argument("--har", required=True)
ap.add_argument("--calib", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--limit", type=int, default=64)
ap.add_argument("--opt-level", type=int, default=0)
ap.add_argument("--compression-level", type=int, default=0)
a = ap.parse_args()
files = []
for p in ("*.jpg", "*.jpeg", "*.png"):
    files += glob.glob(os.path.join(os.path.expanduser(a.calib), "**", p), recursive=True)
files = sorted(files)[:a.limit]
assert files, f"이미지 없음: {a.calib}"
imgs = []
for f in files:
    try:
        imgs.append(letterbox(f, a.imgsz))
    except Exception as e:
        print("skip", f, e)
# Hailo는 NHWC. 정규화는 model script가 칩 안에서 처리하므로 0~255 그대로 전달
calib = np.stack(imgs).astype(np.float32)
print(f"[calib] {calib.shape}  range [{calib.min():.0f}, {calib.max():.0f}]")

runner = ClientRunner(har=a.har)
alls = f"""
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
model_optimization_flavor(optimization_level={a.opt_level}, compression_level={a.compression_level})
"""
print("[model script]", alls)
runner.load_model_script(alls)
runner.optimize(calib)
runner.save_har(a.out)
print("\nSAVED:", a.out, f"({os.path.getsize(a.out)/1e6:.2f} MB)")
