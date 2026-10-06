"""층별 양자화 민감도 분석 — 어떤 층을 FP32 로 남길지 고르는 근거.

INT8 QDQ 모델 안의 모든 QuantizeLinear 지점에서
  - 국소 오차: 양자화 직전 값 vs 양자화->복원 값   (그 층 자체가 만드는 잡음)
  - 누적 오차: FP32 모델의 같은 텐서 vs INT8 모델의 값 (앞 층 잡음이 쌓인 결과)
를 SQNR(dB, 클수록 좋음)로 잰다. 이미지를 한 장씩 흘려 합계만 쌓으므로 메모리는 모델 2개 분량이면 된다.

결과를 Ultralytics 모듈(Conv+BN+SiLU 묶음) 단위로 모아 국소 SQNR 이 낮은 순으로 정렬하고,
'기존 제외목록 + 하위 k개 모듈' 제외목록 파일을 k 별로 만든다. 이걸로 quantize_ort.py 를 돌려
k 에 따른 정확도(AP40)·지연 곡선을 그리면 보고서의 '혼합 정밀도' 근거가 된다.

  python scripts/sensitivity.py --fp32 models/best_1280x384_op13_fp32.onnx \
      --qdq models/best_1280x384_op13_int8-mixed.onnx --base-exclude models/exclude_best_1280x384.txt \
      --images data/kitti_yolo/images/holdout --n 16 --topk 2,4,8
  -> benchmarks/sensitivity_<모델>.json, models/exclude_<모델>_sens_top{k}.txt   (make sensitivity 와 동일)

2026-10-06 COCO YOLO11n 대역 모델 시험: 가장 민감한 모듈은 C2PSA 어텐션(/model.10/m/m.0/attn/).
하위 2/4/8개 모듈을 FP32 로 남기자 클래스 행 SQNR 11.9 -> 13.4/13.1/13.8 dB, 박스 행 32.5 -> 33.7/34.2/35.1 dB.
KITTI 학습 모델에서는 순위가 달라질 수 있으니 반드시 다시 돌린다.
"""
from __future__ import annotations
import argparse, json, random, sys, tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import preprocess  # noqa: E402

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto


def add_outputs(model, names):
    have = {o.name for o in model.graph.output}
    for n in names:
        if n not in have:
            model.graph.output.append(helper.make_tensor_value_info(n, TensorProto.FLOAT, None))
    return model


def session(model):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL   # 텐서 이름 보존
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        return ort.InferenceSession(f.name, so, providers=["CPUExecutionProvider"])


def module_of(node_name):
    """'/model.2/m.0/cv1/conv/Conv' -> '/model.2/m.0/cv1/'  (Conv+BN+SiLU 를 한 모듈로 본다)"""
    for key in ("/conv/", "/act/", "/bn/"):
        if key in node_name:
            return node_name.split(key)[0] + "/"
    return node_name.rsplit("/", 1)[0] + "/"


def sqnr(sig, noise):
    return float(10 * np.log10(max(sig, 1e-12) / max(noise, 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--qdq", required=True, help="quantize_ort.py 가 만든 INT8 QDQ 모델 (보통 int8-mixed)")
    ap.add_argument("--base-exclude", default=None, help="그 QDQ 모델을 만들 때 쓴 제외목록 (head_nodes.py 출력)")
    ap.add_argument("--images", required=True, help="holdout 또는 train 이미지 폴더 (eval_val 금지)")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", default="2,4,8")
    ap.add_argument("--outdir", default="models")
    a = ap.parse_args()
    if "eval_val" in str(a.images):
        sys.exit("[중단] 민감도 분석으로 FP32 유지 층을 고르는 건 모델 선택이다. eval_val 대신 holdout 을 쓴다.")

    fm, qm = onnx.load(a.fp32), onnx.load(a.qdq)
    init = {i.name for i in qm.graph.initializer}
    consumers = defaultdict(list)
    for n in qm.graph.node:
        for i in n.input:
            consumers[i].append(n)
    pairs = {}                                   # 활성값 텐서 -> 그 텐서의 DequantizeLinear 출력
    for n in qm.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[0] not in init:
            dq = [c for c in consumers[n.output[0]] if c.op_type == "DequantizeLinear"]
            if dq:
                pairs[n.input[0]] = dq[0].output[0]
    producer = {o: n.name for n in fm.graph.node for o in n.output}
    pairs = {t: d for t, d in pairs.items() if t in producer}      # FP32 모델에도 있는 텐서만
    if not pairs:
        sys.exit("QDQ 지점을 찾지 못했다 — QDQ 형식으로 양자화한 모델인지 확인하라")

    fs = session(add_outputs(fm, list(pairs)))
    qs = session(add_outputs(qm, list(pairs) + list(pairs.values())))
    fin = fs.get_inputs()[0]
    h, w = fin.shape[2], fin.shape[3]
    fo = [o.name for o in fs.get_outputs()]
    qo = [o.name for o in qs.get_outputs()]
    out0 = fm.graph.output[0].name

    files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".png", ".jpg"))
    random.Random(a.seed).shuffle(files)
    files = files[: a.n]
    acc = defaultdict(lambda: np.zeros(4))      # [국소 신호, 국소 잡음, 누적 신호, 누적 잡음]
    for k, p in enumerate(files):
        x, *_ = preprocess(p, w, h)
        F = dict(zip(fo, fs.run(None, {fin.name: x})))
        Q = dict(zip(qo, qs.run(None, {qs.get_inputs()[0].name: x})))
        for t, d in pairs.items():
            pre, post, ref = Q[t].astype(np.float64), Q[d].astype(np.float64), F[t].astype(np.float64)
            acc[t] += [np.sum(pre ** 2), np.sum((pre - post) ** 2), np.sum(ref ** 2), np.sum((ref - pre) ** 2)]
        ro, qo0 = F[out0][0], Q[out0][0]
        acc["__box__"] += [np.sum(ro[:4] ** 2), np.sum((ro[:4] - qo0[:4]) ** 2), 0, 0]
        acc["__cls__"] += [np.sum(ro[4:] ** 2), np.sum((ro[4:] - qo0[4:]) ** 2), 0, 0]
        print(f"  {k + 1}/{len(files)}", end="\r")

    rows = []
    for t in pairs:
        s = acc[t]
        rows.append({"tensor": t, "producer": producer[t], "module": module_of(producer[t]),
                     "local_sqnr_db": round(sqnr(s[0], s[1]), 2), "accum_sqnr_db": round(sqnr(s[2], s[3]), 2)})
    mods = defaultdict(list)
    for r in rows:
        mods[r["module"]].append(r)
    mod_rank = sorted(({"module": m, "worst_local_db": min(r["local_sqnr_db"] for r in rs),
                        "accum_db": min(r["accum_sqnr_db"] for r in rs), "n_tensors": len(rs)}
                       for m, rs in mods.items()), key=lambda r: r["worst_local_db"])
    final = {"box_rows_sqnr_db": round(sqnr(*acc["__box__"][:2]), 2),
             "cls_rows_sqnr_db": round(sqnr(*acc["__cls__"][:2]), 2)}

    print(f"\n[출력 텐서] 박스 행 SQNR {final['box_rows_sqnr_db']} dB / 클래스 행 SQNR {final['cls_rows_sqnr_db']} dB")
    print(f"[국소 SQNR 하위 모듈] (낮을수록 그 층 자체의 양자화 잡음이 크다, 이미지 {len(files)}장)")
    for r in mod_rank[:15]:
        print(f"  {r['worst_local_db']:7.2f} dB  (누적 {r['accum_db']:6.2f} dB)  {r['module']}")

    stem = Path(a.qdq).stem
    base = []
    if a.base_exclude:
        base = [l.strip() for l in Path(a.base_exclude).read_text().splitlines()
                if l.strip() and not l.startswith("#")]
    all_nodes = [n.name for n in fm.graph.node]
    for k in [int(v) for v in a.topk.split(",") if v.strip()]:
        prefixes = [r["module"] for r in mod_rank[:k]]
        ex = sorted(set(base) | {n for n in all_nodes if any(n.startswith(p) for p in prefixes)})
        out = Path(a.outdir) / f"exclude_{stem}_sens_top{k}.txt"
        out.write_text("".join(f"{n}\n" for n in ex))
        print(f"  top{k}: 제외 {len(ex)}개 노드 -> {out}")
    rep = Path("benchmarks") / f"sensitivity_{stem}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps({"fp32": a.fp32, "qdq": a.qdq, "n_images": len(files), "output": final,
                               "modules": mod_rank, "tensors": rows}, indent=2, ensure_ascii=False))
    print(f"  -> {rep}")
    print("다음: python scripts/quantize_ort.py --model <fp32> --calib <train> "
          "--exclude <위 파일> --suffix sens<k>  로 만들고 AP40·지연을 k 별로 비교한다.")


if __name__ == "__main__":
    main()
