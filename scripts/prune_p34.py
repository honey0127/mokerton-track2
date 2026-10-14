"""구조적 채널 프루닝 + 재학습 (2차 라운드 트랙 S, DECISION_R2.md).

B·D(P5 제거 YOLO11n) 학습 가중치에서 출발해 채널을 L2 크기 기준으로 덜어내고(torch-pruning, DepGraph),
같은 학습 설정으로 다시 학습한다. 채널을 '실제로' 지우므로 ONNX 가 작아지고 CPU·NPU 모두 빨라진다.

  # 1) 프루닝만 (CPU 로도 1분 안쪽) -> runs/prune/<이름>/pruned.pt, 연산량 표
  python scripts/prune_p34.py prune --weights runs/kitti_1280_p34/weights/best.pt --imgsz 1280x384 \
      --target 0.70 --name p34_1280_m70
  # 2) 재학습 (GPU PC). 학습 설정은 A~D 학습(args.yaml)과 같다: epochs 100, batch 8, seed 0, mosaic, close_mosaic 10
  python scripts/prune_p34.py train --pruned runs/prune/p34_1280_m70/pruned.pt --imgsz 1280 --name p34_1280_m70
  # 3) 내보내기·양자화·채점은 기존 경로 그대로
  python scripts/export_onnx.py --weights runs/p34_1280_m70/weights/best.pt --imgsz 1280x384 --opset 13

--target 은 남길 연산량(MACs) 비율이다. 0.70 = 연산량 30% 감소.

무엇을 지우지 않나 (ignored):
  - Detect 의 마지막 Conv (cv2.*.2 박스 64채널, cv3.*.2 클래스 nc 채널)와 DFL: 출력 형식이 바뀌면 안 된다
  - C2PSA (어텐션): head 수·key_dim 이 채널 수에 묶여 있어 채널을 지우면 reshape 가 깨진다
채널 수는 8의 배수로 맞춘다(round_to=8): ARM NEON·NPU 가 8/16 단위로 처리하므로 애매한 수는 속도 이득이 없다.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402


def parse_wh(s):
    w, h = (int(v) for v in s.lower().split("x"))
    return w, h


def load_detection_model(weights):
    from ultralytics import YOLO
    m = YOLO(weights).model
    m = m.float().eval()
    for p in m.parameters():
        p.requires_grad_(True)
    return m


def ignored_layers(model):
    from ultralytics.nn.modules import Detect
    from ultralytics.nn.modules.block import C2PSA
    ign = []
    for mod in model.modules():
        if isinstance(mod, Detect):
            for seq in list(mod.cv2) + list(mod.cv3):
                ign.append(seq[-1])          # 마지막 Conv2d (박스 4*reg_max / 클래스 nc)
            ign.append(mod.dfl)
        if isinstance(mod, C2PSA):
            ign.append(mod)
    return ign


def count(model, w, h):
    import torch_pruning as tp
    x = torch.zeros(1, 3, h, w)
    macs, params = tp.utils.count_ops_and_params(model, x)
    return macs, params


def cmd_prune(a):
    import torch_pruning as tp
    from c2f_v2 import replace_c2f

    w, h = parse_wh(a.imgsz)
    model = load_detection_model(a.weights)
    x = torch.rand(1, 3, h, w)
    with torch.no_grad():
        ref = model(x)[0]
    n = replace_c2f(model)
    model.eval()                     # 새로 만든 Conv(BN) 는 train 모드로 생긴다
    with torch.no_grad():
        same = model(x)[0]
    diff = (ref - same).abs().max().item()
    print(f"[C2f->C2fV2] {n}개 교체, 출력 최대 차이 {diff:.2e} (0 에 가까워야 정상)")
    if diff > 1e-3:
        sys.exit("[중단] C2fV2 교체 후 출력이 달라졌다")

    base_macs, base_params = count(model, w, h)
    imp = tp.importance.GroupMagnitudeImportance(p=2)
    pruner = tp.pruner.MetaPruner(
        model, x, importance=imp, pruning_ratio=a.max_ratio, iterative_steps=a.steps,
        ignored_layers=ignored_layers(model), round_to=8, global_pruning=False,
    )
    macs = base_macs
    for i in range(a.steps):
        pruner.step()
        macs, params = count(model, w, h)
        print(f"  step {i + 1:2d}: MACs {macs / base_macs:.3f}  params {params / base_params:.3f}")
        if macs <= a.target * base_macs:
            break
    model.eval()
    with torch.no_grad():
        out = model(x)[0]
    print(f"[결과] 출력 형상 {tuple(out.shape)} (원래 {tuple(ref.shape)})")
    assert out.shape == ref.shape, "출력 형상이 바뀌었다"

    od = ROOT / "runs" / "prune" / a.name
    od.mkdir(parents=True, exist_ok=True)
    ck = {"model": model.half(), "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "version": "prune_p34",
          "train_args": {"imgsz": max(w, h)}, "pruned_from": a.weights, "target": a.target}
    torch.save(ck, od / "pruned.pt")
    stats = {"weights": a.weights, "imgsz": a.imgsz, "target": a.target,
             "GFLOPs_before": round(2 * base_macs / 1e9, 3), "GFLOPs_after": round(2 * macs / 1e9, 3),
             "params_before_M": round(base_params / 1e6, 3), "params_after_M": round(params / 1e6, 3)}
    (od / "prune_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, ensure_ascii=False))
    print(f"-> {od / 'pruned.pt'}")


def cmd_train(a):
    sys.path.insert(0, str(HERE))
    from train_kitti import resolved_data_yaml
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import attempt_load_one_weight

    model, _ = attempt_load_one_weight(a.pruned)
    model = model.float()
    for p in model.parameters():
        p.requires_grad_(True)
    data = resolved_data_yaml(a.data)
    overrides = dict(
        model=a.pruned, data=data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=a.device,
        workers=a.workers, project=str(ROOT / "runs"), name=a.name, exist_ok=True,
        fliplr=0.5, flipud=0.0, degrees=0.0, mosaic=1.0, close_mosaic=10,
        scale=0.5, translate=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        patience=a.patience, seed=0, deterministic=True, lr0=a.lr0, warmup_epochs=a.warmup,
    )
    trainer = DetectionTrainer(overrides=overrides)
    trainer.model = model           # setup_model() 은 nn.Module 이면 그대로 쓴다 (yaml 로 다시 만들지 않음)
    trainer.train()
    print(f"\n[완료] {trainer.save_dir / 'weights' / 'best.pt'}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prune")
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", required=True, help="WxH, 예: 1280x384")
    p.add_argument("--target", type=float, default=0.70, help="남길 MACs 비율")
    p.add_argument("--max-ratio", type=float, default=0.6, help="층별 최대 채널 삭제 비율")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--name", required=True)
    t = sp.add_parser("train")
    t.add_argument("--pruned", required=True)
    t.add_argument("--imgsz", type=int, required=True, help="긴 변 (1280 또는 640)")
    t.add_argument("--data", default=str(ROOT / "data" / "kitti.yaml"))
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--device", default="0")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--patience", type=int, default=30)
    t.add_argument("--lr0", type=float, default=0.01)
    t.add_argument("--warmup", type=float, default=3.0)
    t.add_argument("--name", required=True)
    a = ap.parse_args()
    {"prune": cmd_prune, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
