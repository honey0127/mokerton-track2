"""C2f / C3k2 의 'cv1 출력을 둘로 쪼개는(chunk)' 구조를 Conv 두 개로 펼친 동치 모듈 — 채널 프루닝용.

원래 C2f.forward 는 y = cv1(x).chunk(2, 1) 로 한 Conv 의 출력 채널을 반으로 나눈다.
채널 프루닝 도구(torch-pruning)는 이 chunk 를 따라가며 '앞 절반·뒤 절반을 같은 수만큼' 지워야 해서
구조가 깨지기 쉽다. cv1 을 cv0(앞 절반), cv1(뒤 절반) 두 Conv 로 나누면 계산 결과는 완전히 같고,
두 갈래를 따로 줄일 수 있다.

이 파일은 프루닝된 체크포인트(.pt)를 불러올 때 pickle 이 클래스를 찾을 수 있도록 scripts/ 에 둔다.
scripts/ 가 sys.path 에 있어야 한다 (export_onnx.py, prune_p34.py 는 스스로 넣는다).
"""
from __future__ import annotations
import copy

import torch
import torch.nn as nn


class C2fV2(nn.Module):
    """C2f/C3k2 와 같은 계산. cv1 하나 대신 cv0·cv1 두 Conv."""

    def __init__(self, src: nn.Module):
        super().__init__()
        from ultralytics.nn.modules import Conv
        c = src.c
        conv, bn = src.cv1.conv, src.cv1.bn
        c1 = conv.in_channels
        self.cv0 = Conv(c1, c, 1, 1)
        self.cv1 = Conv(c1, c, 1, 1)
        for dst, sl in ((self.cv0, slice(0, c)), (self.cv1, slice(c, 2 * c))):
            dst.conv.weight.data.copy_(conv.weight.data[sl])
            if conv.bias is not None:
                dst.conv.bias = nn.Parameter(conv.bias.data[sl].clone())
            dst.bn.weight.data.copy_(bn.weight.data[sl])
            dst.bn.bias.data.copy_(bn.bias.data[sl])
            dst.bn.running_mean.data.copy_(bn.running_mean.data[sl])
            dst.bn.running_var.data.copy_(bn.running_var.data[sl])
            dst.bn.eps, dst.bn.momentum = bn.eps, bn.momentum
            dst.act = copy.deepcopy(src.cv1.act)
        self.cv2 = src.cv2
        self.m = src.m
        # Ultralytics 그래프 순회에 필요한 속성 (from 인덱스, 층 번호, 타입, 파라미터 수)
        for k in ("f", "i", "type", "np"):
            if hasattr(src, k):
                setattr(self, k, getattr(src, k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def replace_c2f(model: nn.Module) -> int:
    """model 안의 C2f/C3k2 를 모두 C2fV2 로 바꾼다. 바꾼 개수를 돌려준다."""
    from ultralytics.nn.modules.block import C2f
    n = 0
    for name, child in list(model.named_children()):
        if isinstance(child, C2f):
            setattr(model, name, C2fV2(child))
            n += 1
        else:
            n += replace_c2f(child)
    return n
