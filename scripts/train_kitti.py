"""KITTI fine-tuning (YOLO11n, COCO 사전학습 가중치에서 시작).

기본은 정사각 캔버스 + mosaic 학습(rect=False)이다.
  - Ultralytics 는 rect=True 이면 mosaic / mixup / cutmix 를 강제로 끈다
    (8.4.135 data/dataset.py build_transforms 확인, 2026-10-06).
    6천 장 남짓한 KITTI 에서 mosaic 없이 학습하면 과적합 위험이 커진다.
  - YOLO 는 합성곱 네트워크라 학습 캔버스가 정사각이어도 추론을 1280x384 직사각으로 해도 된다.
    중요한 건 물체 크기(스케일)가 같은지인데, 둘 다 '긴 변 = imgsz'로 맞추므로 같다.
  - GPU 시간이 정말 부족할 때만 --rect 를 쓴다 (1280 기준 약 1/3 연산, 대신 mosaic 없음).

  python scripts/train_kitti.py --imgsz 1280 --epochs 100 --batch 16          # 정밀도 프로파일
  python scripts/train_kitti.py --imgsz 640  --epochs 100 --batch 32          # 초경량 프로파일
반드시 저장소 루트에서 실행한다.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def resolved_data_yaml(path):
    """yaml 의 path 를 저장소 루트 기준 절대경로로 바꾼 사본을 만든다 (실행 위치와 무관하게 동작)."""
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    for k in ("train", "val"):
        d = root / cfg[k]
        n = sum(1 for _ in d.glob("*.png")) if d.is_dir() else 0
        if n == 0:
            sys.exit(f"[중단] 학습 이미지가 없다: {d}\n  먼저 make data (kitti_to_yolo.py) 를 실행하라.")
        print(f"  {k:5s}: {n}장  ({d})")
    cfg["path"] = str(root)
    out = ROOT / "runs" / "_data_resolved.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--data", default=str(ROOT / "data" / "kitti.yaml"))
    ap.add_argument("--imgsz", type=int, default=1280, help="긴 변 길이 (1280 또는 640)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hours", type=float, default=None,
                    help="최대 학습 시간(시간). 지정하면 epochs 대신 이 시간에 맞춰 학습률 일정을 줄인다 "
                         "(Ultralytics time 인자). GPU 를 넘겨줘야 하는 날짜가 정해져 있을 때 쓴다")
    ap.add_argument("--batch", default="16",
                    help="정수(예: 16) 또는 -1(GPU 메모리 60%%에 맞춰 자동). VRAM 8GB 면 1280 에서 8 또는 -1")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="none", choices=["none", "ram", "disk"],
                    help="ram: 가장 빠르나 1280 기준 약 9GB RAM 필요 / disk: .npy 로 디스크 캐시")
    ap.add_argument("--rect", action="store_true",
                    help="직사각 배치 학습. 빠르지만 mosaic 이 꺼진다 (GPU 시간이 부족할 때만)")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", default=None, help="기본: kitti_<imgsz> -> runs/kitti_<imgsz>/weights/best.pt")
    a = ap.parse_args()

    name = a.name or f"kitti_{a.imgsz}"
    batch = float(a.batch) if "." in a.batch else int(a.batch)
    data = resolved_data_yaml(a.data)
    if a.rect:
        print("[주의] --rect: Ultralytics 가 mosaic/mixup/cutmix 를 끈다.")

    from ultralytics import YOLO
    m = YOLO(a.weights)
    extra = {"time": a.hours} if a.hours else {}
    r = m.train(
        **extra,
        data=data, imgsz=a.imgsz, epochs=a.epochs, batch=batch, device=a.device,
        workers=a.workers, cache=False if a.cache == "none" else a.cache,
        rect=a.rect, project=str(ROOT / "runs"), name=name,
        # KITTI 는 도로 주행 영상이라 상하 반전/회전은 해가 된다
        fliplr=0.5, flipud=0.0, degrees=0.0, mosaic=1.0, close_mosaic=10,
        scale=0.5, translate=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        patience=a.patience, seed=0, deterministic=True,
    )
    save_dir = Path(getattr(r, "save_dir", "") or getattr(m.trainer, "save_dir", ""))
    print(f"\n[완료] 가중치: {save_dir / 'weights' / 'best.pt'}")
    print("  다음: make export WEIGHTS=<위 경로> IMGSZ=1280x384  (640 이면 IMGSZ=640x192)")


if __name__ == "__main__":
    main()
