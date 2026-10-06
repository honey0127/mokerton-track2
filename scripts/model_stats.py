"""파라미터 수 / FLOPs / 모델 크기 — 채점 항목(연산량·파라미터) 대응.

해상도별로 값이 달라지므로 반드시 --imgsz 를 명시한다.

  python scripts/model_stats.py --weights yolo11n.pt --imgsz 1280x384
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import parse_imgsz, jsonl_append  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", default="1280x384")
    ap.add_argument("--jsonl", default="benchmarks/results.jsonl")
    a = ap.parse_args()
    w, h = parse_imgsz(a.imgsz)

    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    import copy
    m = YOLO(a.weights)
    raw = m.model.float().eval()
    # 배포되는 ONNX 는 Conv+BN 이 합쳐진(fused) 그래프다. 채점 대상인 params/FLOPs 는 이 기준으로 센다.
    # (unfused 값도 함께 남긴다: YOLO11n-3cls 1280x384 에서 fused 7.66 / unfused 7.82 GFLOPs)
    net = copy.deepcopy(raw).fuse(verbose=False)

    def count(model):
        # Ultralytics get_flops 는 정사각 imgsz 가정 -> 직사각형은 thop 으로 직접 잰다
        try:
            import thop
            macs, _ = thop.profile(copy.deepcopy(model), inputs=(torch.zeros(1, 3, h, w),), verbose=False)
            return round(macs * 2 / 1e9, 2)   # MACs -> FLOPs
        except Exception as e:
            try:
                return round(get_flops(model, imgsz=max(w, h)), 2)
            except Exception:
                return f"thop 실패: {e}"

    params = get_num_params(net)
    rec = {"kind": "model_stats", "weights": str(a.weights), "imgsz": f"{w}x{h}",
           "params_M": round(params / 1e6, 3), "params": int(params),
           "GFLOPs": count(net), "GFLOPs_unfused": count(raw),
           "params_unfused": int(get_num_params(raw)), "count_method": "thop MACs x2, Conv-BN fused",
           "pt_size_MB": round(Path(a.weights).stat().st_size / 1e6, 2)}
    print(json.dumps(rec, indent=2, ensure_ascii=False))
    jsonl_append(a.jsonl, rec)


if __name__ == "__main__":
    main()
