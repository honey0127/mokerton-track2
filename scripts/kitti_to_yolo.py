"""KITTI label_2 -> YOLO 포맷 변환 + Ultralytics 데이터셋 디렉터리 구성.

클래스 매핑 근거:
  평가 대상은 Car/Pedestrian/Cyclist 3종. 공식 eval은 Car 판정에서 Van을,
  Pedestrian 판정에서 Person_sitting을 '무시(ignore)'로 처리해 FP로 세지 않는다.
  따라서 이 둘을 학습에서 각각 Car/Pedestrian으로 병합하면
  - 눈감아주는 영역에 대한 recall이 올라가고
  - 오탐 페널티는 발생하지 않는다.
  기본 ON(--merge-neighbors). 끄려면 --no-merge-neighbors.

Truck/Tram/Misc/DontCare는 라벨을 만들지 않는다. 다만 DontCare 박스는
평가 단계에서 예측 필터링에 쓰이므로 원본 라벨을 그대로 보존한다.

  python scripts/kitti_to_yolo.py --kitti-root data/kitti_raw --out data/kitti_yolo \
      --imgsz 1280x384 --splits splits
"""
from __future__ import annotations
import argparse, shutil
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import CLASSES, NEIGHBOR  # noqa: E402

MERGE = {"Van": "Car", "Person_sitting": "Pedestrian"}


def convert_one(label_path, w, h, merge=True):
    """KITTI 라벨 한 장 -> YOLO 라인 리스트. 정규화 xywh."""
    lines, stats = [], Counter()
    if not Path(label_path).exists():
        return lines, stats
    for ln in Path(label_path).read_text().splitlines():
        f = ln.split()
        if len(f) < 15:
            continue
        name = f[0]
        if merge and name in MERGE:
            stats[f"merged:{name}->{MERGE[name]}"] += 1
            name = MERGE[name]
        if name not in CLASSES:
            stats[f"skip:{name}"] += 1
            continue
        x1, y1, x2, y2 = map(float, f[4:8])
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w), x2), min(float(h), y2)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            stats["skip:degenerate"] += 1
            continue
        lines.append(f"{CLASSES.index(name)} "
                     f"{(x1 + bw / 2) / w:.6f} {(y1 + bh / 2) / h:.6f} "
                     f"{bw / w:.6f} {bh / h:.6f}")
        stats[f"kept:{name}"] += 1
    return lines, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", required=True,
                    help="training/image_2, training/label_2 를 포함하는 경로")
    ap.add_argument("--out", default="data/kitti_yolo")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--merge-neighbors", dest="merge", action="store_true", default=True)
    ap.add_argument("--no-merge-neighbors", dest="merge", action="store_false")
    ap.add_argument("--link", action="store_true", default=True,
                    help="이미지를 복사하지 않고 심볼릭 링크 (12GB 중복 방지)")
    a = ap.parse_args()

    root = Path(a.kitti_root)
    img_dir = root / "training" / "image_2"
    lab_dir = root / "training" / "label_2"
    for d in (img_dir, lab_dir):
        if not d.is_dir():
            raise SystemExit(f"경로 없음: {d}\n"
                             f"data_object_image_2.zip / data_object_label_2.zip 압축 해제 확인")

    out = Path(a.out)
    total = Counter()
    for split in ("train", "holdout", "eval_val"):
        ids = (Path(a.splits) / f"{split}.txt").read_text().split()
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in ids:
            src = img_dir / f"{i}.png"
            dst = out / "images" / split / f"{i}.png"
            if not dst.exists():
                if a.link:
                    dst.symlink_to(src.resolve())
                else:
                    shutil.copy2(src, dst)
            # KITTI 이미지 크기는 장마다 다르다. PNG 헤더에서 직접 읽는다.
            w, h = png_size(src)
            lines, st = convert_one(lab_dir / f"{i}.txt", w, h, a.merge)
            (out / "labels" / split / f"{i}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            total.update(st)
        print(f"[{split}] {len(ids)}장 변환 완료")

    print("\n[클래스 통계]")
    for k, v in sorted(total.items()):
        print(f"  {k:32s} {v:7d}")
    print("\n주의: DontCare 박스는 YOLO 라벨에 넣지 않았다. "
          "평가 시 scripts/kitti_eval.py 가 원본 label_2 에서 직접 읽어 무시 영역으로 쓴다.")


def png_size(path):
    """PNG 헤더에서 (w, h). PIL/cv2 없이 동작."""
    with open(path, "rb") as f:
        head = f.read(26)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        import cv2
        im = cv2.imread(str(path))
        return im.shape[1], im.shape[0]
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


if __name__ == "__main__":
    main()
