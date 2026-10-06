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
from common import parse_imgsz, preprocess, model_input_wh  # noqa: E402

import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)


class LetterboxReader(CalibrationDataReader):
    """캘리브레이션 입력 공급기.

    __len__ / set_range 를 구현해 ORT 의 CalibStridedMinMax(청크 단위 수집)를 쓸 수 있게 한다.
    Percentile/Entropy 는 청크 없이 돌리면 이미지 1장당 약 350MB(1280x384 기준)씩
    메모리를 계속 쌓아 256장이면 수십 GB가 필요하다 (2026-10-06 측정).
    """

    def __init__(self, files, input_name, w, h):
        self.files, self.name, self.w, self.h = list(files), input_name, w, h
        self.start, self.end = 0, len(self.files)
        self.i = self.start

    def __len__(self):
        return len(self.files)

    def set_range(self, start_index, end_index):
        self.start, self.end = start_index, min(end_index, len(self.files))
        self.i = self.start

    def get_next(self):
        if self.i >= self.end:
            return None
        x, *_ = preprocess(self.files[self.i], self.w, self.h)
        self.i += 1
        return {self.name: x}

    def rewind(self):
        self.i = self.start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    ap.add_argument("--calib", required=True, help="캘리브레이션 이미지 디렉터리")
    ap.add_argument("--imgsz", default=None, help="미지정 시 모델 입력 크기를 그대로 쓴다")
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--op-types", default=None,
                    help="쉼표구분. 예: Conv  (미지정=전체)")
    ap.add_argument("--exclude", default=None, help="제외 노드명 파일 (head_nodes.py 출력)")
    ap.add_argument("--per-channel", action="store_true", default=True)
    ap.add_argument("--calib-method", default="minmax",
                    choices=["minmax", "percentile", "entropy"])
    ap.add_argument("--chunk", type=int, default=8,
                    help="캘리브레이션을 N장씩 끊어 수집한다(메모리 상한). 0 이면 한 번에 수집")
    ap.add_argument("--suffix", default="",
                    help="출력 파일명 꼬리표 추가 (예: --suffix boxfp32 -> ..._int8-mixed-boxfp32.onnx)")
    a = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    mwh = model_input_wh(sess)
    del sess
    w, h = parse_imgsz(a.imgsz) if a.imgsz else mwh
    if mwh and (w, h) != mwh:
        sys.exit(f"[중단] --imgsz {w}x{h} 가 모델 입력 {mwh[0]}x{mwh[1]} 과 다르다")

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
    if a.calib_method != "minmax":      # 캘리브레이션 비교 실험에서 파일이 서로 덮어쓰지 않게
        tag += f"-{a.calib_method}"
    if a.suffix:
        tag += f"-{a.suffix}"
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
    if a.chunk and len(files) > a.chunk:
        if len(files) % a.chunk:
            files = files[: len(files) // a.chunk * a.chunk]   # ORT 는 나누어떨어져야 한다
            kw["calibration_data_reader"] = LetterboxReader(files, iname, w, h)
        kw["extra_options"]["CalibStridedMinMax"] = a.chunk  # 이름과 달리 세 방식 모두에 적용된다
    if a.op_types:
        kw["op_types_to_quantize"] = [s.strip() for s in a.op_types.split(",")]
    if excl:
        kw["nodes_to_exclude"] = excl

    quantize_static(a.model, out, **kw)
    print(f"QUANTIZED: {out}")
    print(f"  calib={len(files)}장 method={a.calib_method} chunk={a.chunk} per_channel={a.per_channel} "
          f"op_types={a.op_types or 'ALL'} excluded={len(excl) if excl else 0}")


if __name__ == "__main__":
    main()
