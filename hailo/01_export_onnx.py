#!/usr/bin/env python3
"""v2(P5 제거) 모델 -> Hailo용 ONNX export + end node 자동 탐색.

Windows 쪽 conda env `kitti-yolo-baseline` 에서 실행한다 (ultralytics/torch 필요).

Hailo DFC 는 Detect 모듈의 **디코드 이전 raw Conv 출력**을 그래프 끝으로 삼아야 한다.
Ultralytics 가 nms=False 로 뽑아도 ONNX 안에는 DFL softmax + 앵커 연산 + Concat 이
그대로 들어있고, 이걸 DFC 에 넣으면 지원하지 않는 연산이거나 정확도가 망가진다.
그래서 parse 단계에서 `--end-node-names` 로 그래프를 잘라줘야 하는데,
P5 를 제거한 모델은 Detect 가 3-scale 이 아니라 2-scale 이므로 end node 개수와
이름이 표준 YOLOv8s 와 다르다. 이 스크립트가 그걸 찾아서 출력한다.

사용:
  python 01_export_onnx.py --weights ../mockathon/runs/v2/weights/best.pt
  python 01_export_onnx.py --weights ../runs/kitti_yolov8s_baseline/weights/best.pt --tag baseline
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def export(weights, imgsz, opset, out_dir, tag):
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        raise SystemExit("ultralytics 가 필요합니다. conda env kitti-yolo-baseline 에서 실행하세요.")

    H, W = imgsz
    for n, v in (("높이", H), ("너비", W)):
        if v % 32:
            raise SystemExit(f"{n} {v} 가 32의 배수가 아닙니다.")

    m = YOLO(weights)
    print(f"모델 로드: {weights}")
    p = Path(m.export(
        format="onnx",
        imgsz=[H, W],
        opset=opset,
        simplify=True,
        nms=False,       # Hailo 는 NMS 없는 raw head 를 요구
        dynamic=False,   # 고정 형상이어야 DFC 가 받는다
    ))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{tag}_{W}x{H}_op{opset}.onnx"
    shutil.move(str(p), dest)
    print(f"ONNX: {dest}  ({dest.stat().st_size / 1e6:.2f} MB)")
    return dest


def find_end_nodes(onnx_path):
    """Detect 모듈의 터미널 Conv 들을 찾는다 = Hailo parse 의 end node 후보.

    판별 기준:
      1. 노드 이름이 /model.<N>/... 형태일 때 N 이 가장 큰 모듈 = Detect
      2. 그 모듈 안의 Conv 중, 출력이 다른 Conv 의 입력으로 쓰이지 않는 것 = 분기 말단
    YOLOv8 의 경우 scale 당 2개(박스 cv2 / 클래스 cv3)가 나온다.
      3-scale 모델 -> 6개,  P5 제거 2-scale 모델 -> 4개
    """
    try:
        import onnx  # noqa: PLC0415
    except ImportError:
        raise SystemExit("onnx 가 필요합니다: pip install onnx")

    g = onnx.load(str(onnx_path)).graph

    # 어떤 텐서가 어떤 노드의 입력으로 쓰이는지
    consumers = defaultdict(list)
    for n in g.node:
        for i in n.input:
            consumers[i].append(n)

    convs = [n for n in g.node if n.op_type == "Conv"]
    if not convs:
        raise SystemExit("Conv 노드가 없습니다. ONNX 가 비정상입니다.")

    def module_idx(name):
        # "/model.22/cv2.0/cv2.0.2/Conv" -> 22
        parts = [p for p in name.split("/") if p.startswith("model.")]
        if not parts:
            return -1
        try:
            return int(parts[0].split(".")[1])
        except (IndexError, ValueError):
            return -1

    max_mod = max(module_idx(n.name) for n in convs)
    head_convs = [n for n in convs if module_idx(n.name) == max_mod]

    terminal = []
    for n in head_convs:
        downstream = consumers.get(n.output[0], [])
        if not any(d.op_type == "Conv" for d in downstream):
            terminal.append(n)

    terminal.sort(key=lambda n: n.name)

    info = {
        "detect_module": f"model.{max_mod}",
        "end_node_names": [n.name for n in terminal],
        "n_end_nodes": len(terminal),
        "scales": len(terminal) // 2 if len(terminal) % 2 == 0 else None,
        "graph_inputs": [
            {"name": i.name,
             "shape": [d.dim_value if d.HasField("dim_value") else None
                       for d in i.type.tensor_type.shape.dim]}
            for i in g.input
        ],
        "graph_outputs": [
            {"name": o.name,
             "shape": [d.dim_value if d.HasField("dim_value") else None
                       for d in o.type.tensor_type.shape.dim]}
            for o in g.output
        ],
    }
    # 각 end node 의 출력 채널 수 (박스=4*reg_max, 클래스=nc 판별용)
    init = {t.name: t for t in g.initializer}
    ch = []
    for n in terminal:
        w = init.get(n.input[1])
        ch.append({"node": n.name, "out_channels": w.dims[0] if w is not None else None})
    info["end_node_channels"] = ch
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Hailo용 ONNX export + end node 탐색")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", nargs=2, type=int, default=[384, 1280], metavar=("H", "W"))
    ap.add_argument("--opset", type=int, default=13,
                    help="기본 13. DFC 가 거부하면 11 로 낮춰볼 것")
    ap.add_argument("--out-dir", default="onnx")
    ap.add_argument("--tag", default=None, help="기본값: 가중치 폴더 이름")
    ap.add_argument("--onnx", default=None,
                    help="이미 export 한 ONNX 가 있으면 경로. export 를 건너뛰고 탐색만 한다")
    args = ap.parse_args(argv)

    if args.onnx:
        path = Path(args.onnx)
    else:
        tag = args.tag or Path(args.weights).parent.parent.name or "model"
        path = export(args.weights, args.imgsz, args.opset, args.out_dir, tag)

    info = find_end_nodes(path)

    print("\n" + "=" * 68)
    print(f"Detect 모듈      : {info['detect_module']}")
    print(f"end node 개수    : {info['n_end_nodes']}"
          f"  (scale {info['scales']}개로 추정)")
    print("입력:")
    for i in info["graph_inputs"]:
        print(f"  {i['name']:<12} {i['shape']}")
    print("출력:")
    for o in info["graph_outputs"]:
        print(f"  {o['name']:<12} {o['shape']}")
    print("\nend node (이걸 hailo parser 에 넘긴다):")
    for c in info["end_node_channels"]:
        print(f"  {c['node']:<52} out_ch={c['out_channels']}")

    if info["n_end_nodes"] == 6:
        print("\n  -> 3-scale 표준 YOLOv8 헤드")
    elif info["n_end_nodes"] == 4:
        print("\n  -> 2-scale (P5 제거) 헤드. Model Zoo 의 yolov8s 설정은 그대로 못 쓴다")
    else:
        print(f"\n  주의: end node {info['n_end_nodes']}개는 예상 밖입니다. 위 목록을 확인하세요")

    side = path.with_suffix(".endnodes.json")
    side.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"\n저장: {side}")

    print("\n다음 단계 — WSL 에서:")
    print(f"  bash 03_hailo_compile.sh {path.name} \\")
    print(f"      '{','.join(info['end_node_names'])}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
