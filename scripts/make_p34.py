"""P5 검출 출력을 뗀 YOLO11n(P3+P4 전용)의 '학습 시작 가중치'를 만든다 — ablation B/D 용.

무엇을 떼나: 층 20~22 (P4 -> P5 로 내려가는 Conv / Concat / C3k2) + Detect 의 P5 가지.
무엇을 남기나: 백본의 P5 단계(층 7~10: Conv, C3k2, SPPF, C2PSA)는 남긴다. 위에서 내려오는 FPN 경로
            (층 11~19)가 이 특징을 P4/P3 로 전달하므로, 이것까지 떼면 '헤드 제거'가 아니라
            '백본 축소'가 되어 비교 대상이 섞인다.
가중치: A/C 와 같은 COCO 사전학습 yolo11n.pt 에서 옮긴다. 층 0~19 는 이름 그대로,
       Detect(원래 23번)의 P3/P4 가지는 입력(16, 19번 층)이 원본과 같으므로 20번으로 이름만 바꿔 옮긴다.
       -> C(원본 640)와 출발점이 'P5 부분이 없다'는 것 말고는 같다. 학습 설정은 train_kitti.py 그대로.

  python scripts/make_p34.py          # -> cfg/yolo11n-p34.yaml, runs/yolo11n-p34.pt
  python scripts/train_kitti.py --weights runs/yolo11n-p34.pt --imgsz 640 --name kitti_640_p34   # D
  python scripts/train_kitti.py --weights runs/yolo11n-p34.pt --imgsz 1280 --name kitti_1280_p34 # B
"""
from __future__ import annotations
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11n.pt", help="COCO 사전학습 가중치 (A/C 학습과 같은 파일)")
    ap.add_argument("--cfg-out", default=str(ROOT / "cfg" / "yolo11n-p34.yaml"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "yolo11n-p34.pt"))
    a = ap.parse_args()

    import yaml
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    # 1) 설계도: Ultralytics 에 들어 있는 yolo11.yaml(scale n)에서 P5 출력만 뗀다
    d = yaml_model_load("yolo11n.yaml")
    head = d["head"]
    if len(head) != 13 or head[-1][2] != "Detect" or list(head[-1][0]) != [16, 19, 22]:
        raise SystemExit(f"[중단] 예상과 다른 yolo11.yaml 구조다 (Ultralytics 버전 확인): {head[-1]}")
    d["head"] = head[:-4] + [[[16, 19], 1, "Detect", ["nc"]]]
    for k in ("yaml_file", "scale"):              # scale 은 파일명(yolo11n-...)의 n 에서 다시 읽힌다
        d.pop(k, None)
    cfg = Path(a.cfg_out)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    text = ["# Ultralytics yolo11.yaml (scale n) 에서 P5 출력(층 20~22)과 Detect 의 P5 가지를 뗀 설계도.",
            "# scripts/make_p34.py 가 만든다. 백본 P5 단계(층 7~10)는 남긴다."]
    flow = lambda x: yaml.safe_dump(x, default_flow_style=True).strip()     # noqa: E731  (리스트 -> [a, b])
    for k, v in d.items():
        if k in ("backbone", "head"):          # 원본 yaml 처럼 층 하나를 한 줄로
            start = 0 if k == "backbone" else len(d["backbone"])
            text.append(f"{k}:")
            text += [f"  - {flow(layer)}  # {i}" for i, layer in enumerate(v, start=start)]
        elif isinstance(v, dict):              # scales
            text.append(f"{k}:")
            text += [f"  {kk}: {flow(vv)}" for kk, vv in v.items()]
        else:                                  # nc
            text.append(yaml.safe_dump({k: v}, default_flow_style=False).strip())
    cfg.write_text("\n".join(text) + "\n", encoding="utf-8")

    # 2) 가중치 옮기기
    y = YOLO(a.weights)
    src = y.model.float().state_dict()
    det_old = len(y.model.model) - 1             # 23
    net = DetectionModel(str(cfg), nc=y.model.nc, verbose=False)
    det_new = len(net.model) - 1                 # 20
    tgt = net.state_dict()
    moved, dropped = {}, 0
    for k, v in src.items():
        _, i, rest = k.split(".", 2)
        i = int(i)
        if i == det_old:
            if rest.startswith(("cv2.2.", "cv3.2.")):        # Detect 의 P5 가지
                dropped += 1
                continue
            k2 = f"model.{det_new}.{rest}"
        elif det_new <= i < det_old:                         # P5 출력용 neck (층 20~22)
            dropped += 1
            continue
        else:
            k2 = k
        if k2 in tgt and tgt[k2].shape == v.shape:
            moved[k2] = v
    missing = [k for k in tgt if k not in moved and not k.endswith("num_batches_tracked")]
    if missing:
        raise SystemExit(f"[중단] 옮기지 못한 텐서 {len(missing)}개 (예: {missing[:3]})")
    net.load_state_dict(moved, strict=False)
    net.names = y.model.names                    # 원본 체크포인트와 같은 클래스 이름 (학습 시작 동작을 C 와 맞춤)

    n_src = sum(p.numel() for p in y.model.parameters())
    n_new = sum(p.numel() for p in net.parameters())
    y.model = net
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    y.save(out)

    chk = YOLO(str(out)).model
    det = chk.model[-1]
    try:
        rel = out.relative_to(ROOT).as_posix()
    except ValueError:
        rel = str(out)
    print(f"[완료] {cfg}\n       {out}")
    print(f"  Detect 출력 {det.nl}개 (stride {[int(s) for s in det.stride]}), 옮긴 텐서 {len(moved)}개 / 버린 텐서 {dropped}개")
    print(f"  파라미터(COCO 80클래스 기준) {n_src / 1e6:.3f}M -> {n_new / 1e6:.3f}M "
          f"(KITTI 3클래스·Conv-BN 합친 기준으로는 2.583M -> 1.845M)")
    print(f"  다음: python scripts/train_kitti.py --weights {rel} --imgsz 640 --name kitti_640_p34")


if __name__ == "__main__":
    main()
