#!/bin/bash
# M1 Pro 속도·메모리 측정 (예선 보고서 4·5·6절용)
#
#   bash run_m1.sh check   # 1분: 환경 점검만 (arm64 / ORT 1.29 / CPU EP / 모델 해시 / 전원)
#   bash run_m1.sh quick   # 약 5분: B·D 의 FP32 대비 INT8 속도 비만 빠르게 본다
#   bash run_m1.sh key     # 약 15~25분: 주력 관련 8개, 4스레드, 15라운드 -> 보고서 표의 원본  ★ 이것이 본 측정
#   bash run_m1.sh all     # 약 30~50분: 22개 전부, 4스레드 + P코어 수, 5라운드 (부록용, 선택)
#
# 측정 조건: 전원 어댑터 연결 / 저전력 모드 끔 / 다른 앱 종료 / 덮개 열고 그대로 둔다.
# 결과는 results/ 에 쌓인다 (저장소에서는 OUT=benchmarks/m1pro 로 준다). 끝나면 results 폴더를 통째로 돌려주면 된다.
set -euo pipefail
cd "$(dirname "$0")"
MODE=${1:-key}
OUT=${OUT:-results}
mkdir -p "$OUT"

# ---------- 0. 환경 점검 ----------
if [ "$(uname -m)" != "arm64" ]; then
  echo "[중단] 이 터미널은 $(uname -m) 이다 (Rosetta). 터미널 앱 '정보 가져오기'에서 Rosetta 를 끄고 다시 실행."; exit 1
fi

PY=${PY:-}
if [ -z "$PY" ]; then
  for c in python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then PY=$c; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "[중단] Python 3.10 이상이 없다. 'brew install python@3.12' 후 다시 실행."; exit 1
fi

if [ ! -d .venv-m1 ]; then
  echo "[준비] 가상환경 .venv-m1 생성 ($PY)"
  "$PY" -m venv .venv-m1
fi
# shellcheck disable=SC1091
. .venv-m1/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements-m1.txt

python - <<'EOF'
import platform, sys, onnxruntime as ort
m = platform.machine()
assert m == "arm64", f"파이썬이 {m} 로 돈다 (x86 파이썬 + Rosetta). arm64 파이썬을 써야 한다"
assert ort.__version__ == "1.29.0", f"onnxruntime {ort.__version__} — 1.29.0 이어야 한다"
assert "CPUExecutionProvider" in ort.get_available_providers()
print(f"[OK] python {sys.version.split()[0]} arm64 / onnxruntime {ort.__version__} / CPUExecutionProvider")
print("     (측정 스크립트는 CPUExecutionProvider 만 쓴다. CoreML/Neural Engine 은 쓰지 않는다)")
EOF

echo "[점검] 모델 SHA256"
grep " models/" MODELS.sha256 | shasum -a 256 -c --quiet && echo "[OK] 모델 22개 해시 일치"

PCORE=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 8)
ECORE=$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo 2)
{
  echo "date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "cpu: $(sysctl -n machdep.cpu.brand_string)"
  echo "cores: P=$PCORE E=$ECORE logical=$(sysctl -n hw.logicalcpu)"
  echo "memory_GB: $(( $(sysctl -n hw.memsize) / 1073741824 ))"
  echo "macos: $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  echo "python: $(python -c 'import sys;print(sys.version.split()[0])')"
  echo "onnxruntime: $(python -c 'import onnxruntime as o;print(o.__version__)')"
  echo "numpy: $(python -c 'import numpy as n;print(n.__version__)')"
  echo "power: $(pmset -g batt | head -1)"
  echo "lowpowermode: $(pmset -g | awk '/lowpowermode/{print $2}')"
} > "$OUT/env_m1.txt"
cat "$OUT/env_m1.txt"

if ! pmset -g batt | head -1 | grep -q "AC Power"; then
  echo "[경고] 배터리로 돌고 있다. 전원 어댑터를 꽂고 다시 실행하는 것을 권한다."
fi
if [ "$(pmset -g | awk '/lowpowermode/{print $2}')" = "1" ]; then
  echo "[중단] 저전력 모드가 켜져 있다. 시스템 설정 > 배터리 에서 끄고 다시 실행."; exit 1
fi

[ "$MODE" = "check" ] && { echo "[끝] 점검만 했다."; exit 0; }

A=models/best_1280x384_op13; B=models/best-p34_1280x384_op13
C=models/best_640x192_op13;  D=models/best-p34_640x192_op13
I8=int8-mixed-percentile
KEY8="${A}_fp32.onnx ${A}_$I8.onnx ${B}_fp32.onnx ${B}_$I8.onnx ${C}_fp32.onnx ${C}_$I8.onnx ${D}_fp32.onnx ${D}_$I8.onnx"

# caffeinate: 측정 중 잠자기·화면 꺼짐 방지
case "$MODE" in
  quick)
    caffeinate -dims python scripts/bench_suite.py "$B"_fp32.onnx "$B"_$I8.onnx "$D"_fp32.onnx "$D"_$I8.onnx \
      --threads 4 --rounds 3 --runs 150 --soak 30 --tag m1-quick \
      --jsonl "$OUT/bench_quick_m1pro.jsonl" --raw "$OUT/bench_quick_m1pro_rounds.jsonl"
    python scripts/bench_compare.py --jsonl "$OUT/bench_quick_m1pro.jsonl" --threads 4 --out "$OUT/bench_quick_m1pro.md"
    ;;
  key)
    # 2026-10-14 노트북(i7-10510U) bench_key15 와 같은 조건: 4스레드, 15라운드, 예열 120초, 300회
    caffeinate -dims python scripts/bench_suite.py $KEY8 \
      --threads 4 --rounds 15 --soak 120 --tag key15-m1pro \
      --jsonl "$OUT/bench_key15_m1pro.jsonl" --raw "$OUT/bench_key15_m1pro_rounds.jsonl"
    python scripts/bench_compare.py --jsonl "$OUT/bench_key15_m1pro.jsonl" --threads 4 --out "$OUT/bench_key15_m1pro.md"
    ;;
  all)
    caffeinate -dims python scripts/bench_suite.py models/*.onnx \
      --threads 4,"$PCORE" --rounds 5 --soak 120 --tag all-m1pro \
      --jsonl "$OUT/bench_all_m1pro.jsonl" --raw "$OUT/bench_all_m1pro_rounds.jsonl"
    ;;
  *) echo "모드: check | quick | key | all"; exit 1 ;;
esac
echo "[끝] 결과: $(pwd)/$OUT"
