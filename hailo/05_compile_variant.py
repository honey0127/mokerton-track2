#!/usr/bin/env python3
"""optimized HAR 에 allocator/context 관련 .alls 줄만 덧붙여 compile 만 다시 한다 (양자화 재실행 없음).

  python compile_variant.py <optimized.har> <extra.alls> <out.hef>
"""
import sys
from hailo_sdk_client import ClientRunner

har, script, out = sys.argv[1:4]
extra = open(script).read()
print("== 추가 model script:\n" + extra, flush=True)
r = ClientRunner(har=har)
# 양자화된 HAR 은 append 불가. 비-append 로 불러도 normalization 등은 이미 그래프에 반영돼 있어
# 여기서는 allocator/context 같은 compile 단계 명령만 새로 적용된다.
r.load_model_script(extra)
hef = r.compile()
open(out, "wb").write(hef)
print(f"== HEF 저장: {out} ({len(hef)/1e6:.2f} MB)", flush=True)
