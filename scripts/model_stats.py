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

    m = YOLO(a.weights)
    net = m.model.float().eval()

    params = get_num_params(net)
    # Ultralytics get_flops 는 정사각 imgsz 가정 -> 직사각형은 thop 으로 직접 잰다
    gflops = None
    try:
        import thop
        x = torch.zeros(1, 3, h, w)
        macs, _ = thop.profile(net, inputs=(x,), verbose=False)
        gflops = round(macs * 2 / 1e9, 2)   # MACs -> FLOPs
    except Exception as e:
        gflops = f"thop 실패: {e}"
        try:
            gflops = round(get_flops(net, imgsz=max(w, h)) * 2, 2)
        except Exception:
            pass

    rec = {"kind": "model_stats", "weights": str(a.weights), "imgsz": f"{w}x{h}",
           "params_M": round(params / 1e6, 3), "params": int(params),
           "GFLOPs": gflops, "pt_size_MB": round(Path(a.weights).stat().st_size / 1e6, 2)}
    print(json.dumps(rec, indent=2, ensure_ascii=False))
    jsonl_append(a.jsonl, rec)


if __name__ == "__main__":
    main()
