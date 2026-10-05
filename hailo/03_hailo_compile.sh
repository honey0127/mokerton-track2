#!/usr/bin/env bash
# Hailo DFC 파이프라인: ONNX -> HAR -> 양자화 -> HEF -> 성능 예측
#
# WSL Ubuntu, conda env `hailo-dfc` 에서 실행한다.
#   source ~/miniconda3/bin/activate hailo-dfc
#   bash 03_hailo_compile.sh <onnx> <end_node_names> [calib.npy] [arch]
#
# 예:
#   bash 03_hailo_compile.sh onnx/v2_1280x384_op13.onnx \
#     '/model.16/cv2.0/cv2.0.2/Conv,/model.16/cv3.0/cv3.0.2/Conv,/model.16/cv2.1/cv2.1.2/Conv,/model.16/cv3.1/cv3.1.2/Conv' \
#     calib_648.npy hailo10h
#
# end_node_names 는 01_export_onnx.py 가 출력해준 목록을 쉼표로 이어 붙인 것.

set -uo pipefail

ONNX="${1:?ONNX 경로를 넘기세요}"
END_NODES="${2:?end node 이름(쉼표 구분)을 넘기세요}"
CALIB="${3:-calib.npy}"
ARCH="${4:-hailo10h}"

NAME="$(basename "${ONNX%.onnx}")"
OUT="hailo_out/${NAME}"
mkdir -p "$OUT"
LOG="${OUT}/pipeline.log"

# RTX 5070(Blackwell) + TensorFlow GPU 비호환 우회 — DFC 가 import 시 TF로
# GPU 테스트를 하는데 CUDA_ERROR_INVALID_HANDLE 로 죽는다. CPU 강제.
export CUDA_VISIBLE_DEVICES=''

say()  { printf '\n\033[1m== %s\033[0m\n' "$*" | tee -a "$LOG"; }
fail() { printf '\n\033[1;31m[실패] %s\033[0m\n' "$*" | tee -a "$LOG"; }

say "환경 확인"
{
  hailo --version 2>&1 | head -3
  python -c "import hailo_sdk_client as c; print('hailo_sdk_client', getattr(c,'__version__','?'))" 2>&1
  echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'  (비어있어야 정상)"
} | tee -a "$LOG"

say "입력"
{
  echo "ONNX      : $ONNX"
  echo "end nodes : $END_NODES"
  echo "calib     : $CALIB"
  echo "arch      : $ARCH"
  ls -la "$ONNX" "$CALIB" 2>&1
} | tee -a "$LOG"

[ -f "$ONNX" ]  || { fail "ONNX 가 없습니다"; exit 1; }
[ -f "$CALIB" ] || { fail "캘리브레이션 npy 가 없습니다 (02_make_calib.py 먼저 실행)"; exit 1; }

# ---------------------------------------------------------------- 지원 아키텍처
say "지원 하드웨어 아키텍처 확인"
hailo parser onnx --help 2>&1 | grep -iA4 'hw-arch' | tee -a "$LOG" || true
echo "  -> 위 목록에 '$ARCH' 가 없으면 4번째 인자로 올바른 이름을 넘기세요" | tee -a "$LOG"

# ---------------------------------------------------------------- 1. parse
say "1/4  parse (ONNX -> HAR)"
hailo parser onnx "$ONNX" \
  --hw-arch "$ARCH" \
  --har-path "${OUT}/${NAME}.har" \
  --end-node-names ${END_NODES//,/ } \
  2>&1 | tee -a "$LOG"

if [ ! -f "${OUT}/${NAME}.har" ]; then
  fail "parse 실패"
  cat <<'EOF' | tee -a "$LOG"

  흔한 원인:
    - end node 이름 오타 → 01_export_onnx.py 출력을 그대로 복사
    - opset 미지원 → 01_export_onnx.py --opset 11 로 다시 export
    - 지원하지 않는 연산 → 로그에서 'Unsupported' 줄을 찾아 보고
    - --end-node-names 플래그명이 DFC 버전마다 다름 → hailo parser onnx --help 확인
EOF
  exit 1
fi

# ---------------------------------------------------------------- 모델 스크립트
ALLS="${OUT}/model_script.alls"
if [ ! -f "$ALLS" ]; then
  cat > "$ALLS" <<'EOF'
# 입력 정규화를 NPU 안에서 처리한다. 캘리브레이션 npy 를 uint8 0~255 로
# 만들었으므로 여기서 255 로 나눈다 (Ultralytics 전처리와 동일).
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])

# 1차 시도는 기본 최적화 레벨로 간다.
# optimization_level 3~4 와 Adaround/QFT 는 GPU 가 필요한데
# Blackwell(RTX 5070) + TensorFlow 비호환으로 CPU 밖에 못 쓴다 → 매우 느림.
# 일단 .hef 가 나오는 것을 확인한 뒤에 올릴 것.
model_optimization_flavor(optimization_level=0, compression_level=0)
EOF
  echo "모델 스크립트 생성: $ALLS" | tee -a "$LOG"
fi

# ---------------------------------------------------------------- 2. optimize
say "2/4  optimize (양자화 — 여기가 가장 오래 걸립니다)"
hailo optimize "${OUT}/${NAME}.har" \
  --hw-arch "$ARCH" \
  --calib-set-path "$CALIB" \
  --model-script "$ALLS" \
  --output-har-path "${OUT}/${NAME}_optimized.har" \
  2>&1 | tee -a "$LOG"

if [ ! -f "${OUT}/${NAME}_optimized.har" ]; then
  fail "optimize 실패"
  cat <<'EOF' | tee -a "$LOG"

  흔한 원인:
    - 캘리브레이션 배열 형상 불일치 → npy 는 (N, H, W, 3) uint8 이어야 함
    - 메모리 부족 → .alls 에 model_optimization_config(calibration, batch_size=4) 추가
    - TF GPU 오류 → CUDA_VISIBLE_DEVICES 가 비어있는지 확인
EOF
  exit 1
fi

# ---------------------------------------------------------------- 3. compile
say "3/4  compile (HAR -> HEF)"
hailo compiler "${OUT}/${NAME}_optimized.har" \
  --hw-arch "$ARCH" \
  --output-dir "$OUT" \
  2>&1 | tee -a "$LOG"

HEF="$(find "$OUT" -name '*.hef' -newer "$ONNX" | head -1)"
if [ -z "$HEF" ]; then
  fail "compile 실패 — .hef 가 생성되지 않았습니다"
  echo "  리소스 초과면 로그에 'resources' / 'utilization' 경고가 있습니다." | tee -a "$LOG"
  echo "  해상도를 640x192 로 낮추거나 모델을 더 줄여야 합니다." | tee -a "$LOG"
  exit 1
fi

# ---------------------------------------------------------------- 4. profile
say "4/4  profile (예상 성능)"
hailo profiler "${OUT}/${NAME}_optimized.har" \
  --out-path "${OUT}/profile.html" \
  2>&1 | tee -a "$LOG" || \
  echo "  profiler 실패 (플래그명이 버전마다 다름). hailo profiler --help 확인" | tee -a "$LOG"

# ---------------------------------------------------------------- 결과
say "완료"
{
  echo "HEF      : $HEF  ($(du -h "$HEF" | cut -f1))"
  echo "HAR      : ${OUT}/${NAME}_optimized.har"
  echo "profile  : ${OUT}/profile.html"
  echo "로그     : $LOG"
  echo
  echo "보고서에 들어갈 것:"
  echo "  - .hef 크기"
  echo "  - profiler 의 예상 FPS / latency / 리소스 사용률"
  echo "  - 로그에서 레이어별 양자화 방식 (어디가 INT8, 어디가 INT4/FP16)"
} | tee -a "$LOG"
