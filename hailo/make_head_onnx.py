#!/usr/bin/env python3
"""PC 검증용: FP32 ONNX 를 Hailo parse 와 같은 위치(Detect 의 마지막 Conv 출력)에서 잘라 head ONNX 를 만든다.

HEF 는 이 head 출력(box 64채널 + cls 3채널, stride 별)을 내고 DFL·sigmoid·NMS 는 CPU 에서 한다.
07_board_fast.py / 08_demo.py 의 --backend onnx 가 이 파일을 쓰면 보드 없이 같은 코드 경로를 확인할 수 있다.

  python hailo/make_head_onnx.py models/best-p34_640x192_op13_fp32.onnx runs/head_D.onnx
"""
import sys
from pathlib import Path

import onnx
from onnx.utils import extract_model


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src)
    outs = [n.output[0] for n in m.graph.node
            if n.op_type == "Conv" and n.name.endswith(".2/Conv") and ("/cv2." in n.name or "/cv3." in n.name)]
    if len(outs) not in (4, 6):
        sys.exit(f"head Conv 를 {len(outs)}개 찾았다 (4 또는 6 이어야 함): {outs}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    extract_model(src, dst, [m.graph.input[0].name], outs)
    print(f"{dst}: 출력 {len(outs)}개")
    for o in outs:
        print("  ", o)


if __name__ == "__main__":
    main()
