"""정적 INT8 PTQ — 직사각형 letterbox 캘리브레이션.

전처리는 scripts/common.py 의 letterbox 하나만 쓴다. 추론 전처리와 어긋나면
PTQ 스케일이 통째로 틀어진다 (KITTI 1242x375 -> 1280x384 는 패딩이 거의 0이므로
640x640 시절 전처리를 그대로 쓰면 안 된다).

  # 전체 양자화 (실패 재현용)
  python scripts/quantize_ort.py --model m.onnx --calib data/kitti_yolo/images/train --imgsz 1280x384
  # 채택안: Detect 디코드 경로 제외
  python scripts/head_nodes.py m.onnx > models/exclude.txt
  python scripts/quantize_ort.py --model m.onnx --calib ... --exclude models/exclude.txt
"""
from __future__ import annotations
import argparse, random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import parse_imgsz, preprocess  # noqa: E402

import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)


class LetterboxReader(CalibrationDataReader):
    def __init__(self, files, input_name, w, h):
        self.files, self.name, self.w, self.h = list(files), input_name, w, h
        self.i = 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        x, *_ = preprocess(self.files[self.i], self.w, self.h)
        self.i += 1
        return {self.name: x}

    def rewind(self):
        self.i = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    ap.add_argument("--calib", required=True, help="캘리브레이션 이미지 디렉터리")
    ap.add_argument("--imgsz", default="1280x384")
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--op-types", default=None,
                    help="쉼표구분. 예: Conv  (미지정=전체)")
    ap.add_argument("--exclude", default=None, help="제외 노드명 파일 (head_nodes.py 출력)")
    ap.add_argument("--per-channel", action="store_true", default=True)
    ap.add_argument("--calib-method", default="minmax",
                    choices=["minmax", "percentile", "entropy"])
    a = ap.parse_args()
    w, h = parse_imgsz(a.imgsz)

    import onnxruntime as ort
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    del sess

    files = sorted(p for p in Path(a.calib).iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    random.Random(a.seed).shuffle(files)
    files = files[: a.n_calib]
    if not files:
        raise SystemExit(f"캘리브레이션 이미지 없음: {a.calib}")

    excl = None
    if a.exclude:
        excl = [l.strip() for l in Path(a.exclude).read_text().splitlines()
                if l.strip() and not l.startswith("#")]

    tag = "int8"
    if a.op_types:
        tag += "-" + a.op_types.lower().replace(",", "")
    if excl:
        tag += "-mixed"
    out = a.out or str(Path(a.model).with_name(
        Path(a.model).stem.replace("_fp32", "") + f"_{tag}.onnx"))

    from onnxruntime.quantization import CalibrationMethod
    cm = {"minmax": CalibrationMethod.MinMax,
          "percentile": CalibrationMethod.Percentile,
          "entropy": CalibrationMethod.Entropy}[a.calib_method]

    kw = dict(
        calibration_data_reader=LetterboxReader(files, iname, w, h),
        quant_format=QuantFormat.QDQ,
        per_channel=a.per_channel,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=cm,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    if a.op_types:
        kw["op_types_to_quantize"] = [s.strip() for s in a.op_types.split(",")]
    if excl:
        kw["nodes_to_exclude"] = excl

    quantize_static(a.model, out, **kw)
    print(f"QUANTIZED: {out}")
    print(f"  calib={len(files)}장 method={a.calib_method} per_channel={a.per_channel} "
          f"op_types={a.op_types or 'ALL'} excluded={len(excl) if excl else 0}")


if __name__ == "__main__":
    main()
