"""Detect 모듈 디코드 경로 노드 자동 추출 (INT8 mixed 양자화용 제외 목록).

근거: YOLO 출력 [1, 4+nc, A] 은 행 0~3 이 박스 좌표(0~입력폭 픽셀),
행 4~ 가 클래스 확률(0~1)로 스케일이 2~3자릿수 다른 값을 한 텐서에 담는다.
이를 단일 스케일로 uint8 양자화하면 스케일이 (입력폭/255)로 잡혀
클래스 점수가 전부 0으로 뭉개진다 -> mAP 0.

따라서 Detect 모듈 내부의 non-Conv 연산과 그래프 출력 생성 노드만 FP32로 남긴다.

  python scripts/head_nodes.py model.onnx > exclude.txt
"""
from __future__ import annotations
import argparse, sys
import onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--head-prefix", default=None,
                    help="Detect 모듈 노드명 접두사. 미지정 시 자동 탐지 (예: /model.23/)")
    ap.add_argument("--keep-conv", action="store_true", default=True,
                    help="헤드 내부 Conv는 양자화 대상으로 남긴다 (기본)")
    a = ap.parse_args()

    m = onnx.load(a.model)
    nodes = list(m.graph.node)

    prefix = a.head_prefix
    if prefix is None:
        # 그래프 출력을 만드는 노드에서 역으로 /model.N/ 접두사를 찾는다
        outs = {o.name for o in m.graph.output}
        cands = [n.name for n in nodes if set(n.output) & outs]
        segs = [n.split("/")[1] for n in cands if n.startswith("/model.")]
        if not segs:
            sys.exit("Detect 모듈 접두사 자동 탐지 실패 — --head-prefix 로 지정하라")
        # 가장 큰 model.N 이 Detect
        idx = max(int(s.split(".")[1]) for s in segs)
        prefix = f"/model.{idx}/"
    print(f"# head prefix: {prefix}", file=sys.stderr)

    out_names = {o.name for o in m.graph.output}
    exclude = []
    for n in nodes:
        in_head = n.name.startswith(prefix)
        makes_output = bool(set(n.output) & out_names)
        if makes_output:
            exclude.append(n.name)
        elif in_head and not (a.keep_conv and n.op_type == "Conv"):
            exclude.append(n.name)

    exclude = sorted(set(exclude))
    print(f"# excluded {len(exclude)} nodes", file=sys.stderr)
    for n in exclude:
        print(n)


if __name__ == "__main__":
    main()
