"""ONNX 그래프 검증 — opset / 로드 가능 여부 / 입출력 형상."""
from __future__ import annotations
import argparse, json, sys

import onnx, onnxruntime as ort


def shape_of(t):
    return [d.dim_value if d.dim_value else (d.dim_param or "?")
            for d in t.type.tensor_type.shape.dim]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    a = ap.parse_args()

    m = onnx.load(a.model)
    ops = {i.domain or "ai.onnx": i.version for i in m.opset_import}
    info = {
        "model": a.model,
        "opset": ops,
        "ir_version": m.ir_version,
        "producer": m.producer_name,
        "inputs": {i.name: shape_of(i) for i in m.graph.input},
        "outputs": {o.name: shape_of(o) for o in m.graph.output},
        "n_nodes": len(m.graph.node),
    }
    try:
        onnx.checker.check_model(m)
        info["checker"] = "PASS"
    except Exception as e:
        info["checker"] = f"FAIL: {e}"
    try:
        s = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
        info["ort_load"] = "PASS"
        info["ort_input"] = {i.name: i.shape for i in s.get_inputs()}
        info["ort_output"] = {o.name: o.shape for o in s.get_outputs()}
    except Exception as e:
        info["ort_load"] = f"FAIL: {e}"

    print(json.dumps(info, indent=2, ensure_ascii=False))
    bad = info["checker"] != "PASS" or info["ort_load"] != "PASS"
    if ops.get("ai.onnx", 0) < 13:
        print("\n[경고] ai.onnx opset < 13 — per-channel PTQ 불가", file=sys.stderr)
        bad = True
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
