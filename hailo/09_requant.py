#!/usr/bin/env python3
"""공용 PC (DFC 5.4, WSL) 정확도 개선 재양자화: parse 된 HAR 에서 양자화 설정만 바꿔 다시 optimize -> (선택) compile.

지금 HEF 는 optimization_level=0 (보정 이미지 실제 64장). 같은 보정 목록 256장으로 아래 변형을 만든다.

  Q1 : optimization_level=1 (equalization), calibset_size=256
  Q2 : Q1 + 박스 출력 conv(64채널, cv2.*.2) 를 16비트(a16_w16)   — 예선 분석에서 640x192 Car 손실 원인이 박스 경로
  Q3 : optimization_level=2 (finetune 포함, CPU 에서는 오래 걸림), calibset_size=256

  conda activate hailo-dfc
  # 0) 출력층 이름 확인 (Q2 가 어떤 층을 16비트로 바꾸는지)
  python hailo/09_requant.py --har hailo_out/D_640x192/D_640x192.har --list
  # 1) 양자화 + 에뮬레이터 holdout 채점
  python hailo/09_requant.py --har hailo_out/D_640x192/D_640x192.har --calib calib_D_256.npy --variant Q1
  python hailo/04_emulate_eval.py --har hailo_out/D_640x192/D_640x192_Q1_optimized.har --repo . \
      --images <holdout 이미지> --gt <label_2> --split splits/holdout.txt --context quantized --tag D_Q1_q
  # 2) 현재(D 90.44)보다 +0.30 이상이면 compile -> hailo/hef/D_640x192_Q1.hef
  python hailo/09_requant.py --har hailo_out/D_640x192/D_640x192_Q1_optimized.har --compile hailo/hef/D_640x192_Q1.hef

--har 는 03_hailo_compile.sh 의 1단계(parse) 결과 (<이름>.har, _optimized 가 아닌 것).
보드 판정은 DECISION_FINAL.md 2번 (보드 holdout +0.30, FPS 0.95배 이상).
※ 보드 없이 PC(클라우드)에서 작성했고 DFC 로는 아직 돌려보지 않았다. 명령 이름이 맞지 않으면 에러를 그대로 보내주세요.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np

BASE = """normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
"""
VARIANTS = {
    "Q1": "model_optimization_flavor(optimization_level=1, compression_level=0)\n"
          "model_optimization_config(calibration, batch_size=8, calibset_size=256)\n",
    "Q2": "model_optimization_flavor(optimization_level=1, compression_level=0)\n"
          "model_optimization_config(calibration, batch_size=8, calibset_size=256)\n"
          "quantization_param([{box}], precision_mode=a16_w16)\n",
    "Q3": "model_optimization_flavor(optimization_level=2, compression_level=0)\n"
          "model_optimization_config(calibration, batch_size=8, calibset_size=256)\n",
}


def head_layers(runner):
    """출력층 바로 앞 conv 와 채널 수. box = 64채널 (4 x reg_max 16), cls = 3채널."""
    hn = runner.get_hn_dict()
    layers = hn["layers"]
    res = []
    for name, L in layers.items():
        if L.get("type") != "output_layer":
            continue
        for src in L.get("input", []):
            S = layers[src]
            ch = (S.get("output_shapes") or [[None]])[0][-1]
            res.append((name, src, S.get("type"), ch, S.get("original_names")))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--har", required=True)
    ap.add_argument("--list", action="store_true", help="출력층·앞단 conv 이름만 출력")
    ap.add_argument("--calib", help="02_make_calib.py 의 npy (uint8 0~255, NHWC)")
    ap.add_argument("--variant", choices=sorted(VARIANTS))
    ap.add_argument("--compile", metavar="OUT_HEF", help="optimized HAR 를 compile 해서 HEF 저장")
    a = ap.parse_args()

    from hailo_sdk_client import ClientRunner
    runner = ClientRunner(har=a.har)

    if a.list:
        for out, src, typ, ch, orig in head_layers(runner):
            print(f"{out:<28} <- {src:<28} {typ:<10} ch={ch}  onnx={orig}")
        return

    if a.compile:
        t0 = time.time()
        hef = runner.compile()
        Path(a.compile).parent.mkdir(parents=True, exist_ok=True)
        Path(a.compile).write_bytes(hef)
        runner.save_har(a.compile.replace(".hef", "_compiled.har"))
        print(f"HEF 저장: {a.compile} ({len(hef) / 1e6:.2f} MB, {time.time() - t0:.0f}s)")
        print("context 수는 profiler 로: hailo profiler <compiled.har>")
        return

    if not (a.calib and a.variant):
        sys.exit("--calib 와 --variant 가 필요하다 (또는 --list / --compile)")
    script = BASE + VARIANTS[a.variant]
    if "{box}" in script:
        box = [src.split("/")[-1] for _, src, _, ch, _ in head_layers(runner) if ch == 64]
        if not box:
            sys.exit("64채널 박스 출력 conv 를 찾지 못했다 — --list 결과를 보내주세요")
        script = script.replace("{box}", ", ".join(box))
    print("== model script\n" + script, flush=True)
    calib = np.load(a.calib)
    print(f"== 보정 세트 {calib.shape} {calib.dtype} (eval_val 과 겹치지 않는 train 256장이어야 함)")
    runner.load_model_script(script)
    t0 = time.time()
    runner.optimize(calib)
    out = a.har.replace(".har", f"_{a.variant}_optimized.har")
    runner.save_har(out)
    Path(out.replace(".har", ".alls")).write_text(script)
    print(f"== 저장: {out} ({time.time() - t0:.0f}s)\n다음: 04_emulate_eval.py --har {out} --context quantized ...")


if __name__ == "__main__":
    main()
