#!/bin/bash
# 본선 보드 측정 한 번에 돌리기 (Pi 5 + AI HAT+ 2). 저장소 루트에서 실행한다.
#
#   IMG=<holdout 이미지 폴더> GT=<KITTI label_2 폴더> bash hailo/run_board_final.sh <모드>
#
#   check : 빠른 디코드가 06 디코드와 같은지 (4개 모델 x 출력 형식 2개). 2분
#   sweep : batch {1,2,4,8} x inflight {1,2,4} 처리량·지연·CPU·메모리 + 06 방식(sync) 기준선. 모델당 3~5분
#   best  : sweep 에서 FPS 최고 조합을 3번 더 (규칙상 3회 중앙값). 모델당 2분
#   ap    : holdout 500장 정확도 덤프(conf 0.001) -> 보드에서 채점. 06 결과(ap_board.txt)와 같아야 한다
#   soak  : 30분 연속 구동 발열 시험 (D 와 B 각각 최대 속도, 0.5초마다 FPS·CPU·온도 기록). 1시간
#   demo  : 시연 프로그램 20초 headless 점검 (창 없이) + 녹화 1개
#   all   : check -> sweep -> best -> ap -> demo
#   정리  : python3 hailo/summarize_final.py  -> hailo/results/board/final/SUMMARY.md (서버 vs 엣지 표 + 규칙 판정)
#
# 결과는 hailo/results/board/final/ 에 모인다. 끝나면 그 폴더를 커밋해서 올려주면 된다.
# 측정 중에는 다른 프로그램을 띄우지 말고, 케이스·쿨러를 장착한 상태로 둔다.
set -euo pipefail
MODE="${1:-all}"
: "${IMG:?IMG=<holdout 이미지 폴더> 를 지정하세요}"
OUT=hailo/results/board/final
mkdir -p "$OUT"
PY=python3
MODELS="${MODELS:-A_1280x384 B_1280x384 C_640x192 D_640x192}"   # 새 HEF 만: MODELS="D_640x192_Q1" bash ...
wh() { echo "$1" | grep -o '[0-9]*x[0-9]*'; }
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }

env_info() {
  { date; uname -a; hailortcli fw-control identify 2>&1 | grep -E "Firmware|Architecture" || true
    $PY -c "import hailo_platform, numpy, cv2; print('hailo_platform', hailo_platform.__version__, 'numpy', numpy.__version__, 'cv2', cv2.__version__)"
    vcgencmd measure_temp; vcgencmd get_throttled; git rev-parse HEAD; } > "$OUT/env.txt" 2>&1
}

do_check() {
  for m in $MODELS; do
    log "check $m"
    $PY hailo/07_board_fast.py --model hailo/hef/$m.hef --imgsz "$(wh $m)" --images "$IMG" \
        --split splits/holdout.txt -n 50 --conf 0.001 --check-decode 2>&1 | tee "$OUT/check_$m.txt"
  done
}

do_sweep() {
  for m in $MODELS; do
    log "sweep $m"
    sleep 20                                    # 앞 모델 열 식히기
    {
      $PY hailo/07_board_fast.py --model hailo/hef/$m.hef --imgsz "$(wh $m)" --images "$IMG" \
          --split splits/holdout.txt -n 64 --frames 200 --mode sync --json "$OUT/fast_$m.jsonl"
      $PY hailo/07_board_fast.py --model hailo/hef/$m.hef --imgsz "$(wh $m)" --images "$IMG" \
          --split splits/holdout.txt -n 64 --frames 600 --batch 1,2,4,8 --inflight 1,2,4 --json "$OUT/fast_$m.jsonl"
      $PY hailo/07_board_fast.py --model hailo/hef/$m.hef --imgsz "$(wh $m)" --images "$IMG" \
          --split splits/holdout.txt -n 64 --frames 600 --batch 1,4,8 --inflight 2 --out-format raw --json "$OUT/fast_$m.jsonl"
    } 2>&1 | tee "$OUT/sweep_$m.txt"
  done
}

do_best() {                                     # sweep 에서 FPS 최고 조합을 골라 3번 더 (규칙: 3회 중앙값)
  for m in $MODELS; do
    [ -f "$OUT/fast_$m.jsonl" ] || { echo "sweep 먼저: $m"; continue; }
    read -r B I F < <($PY -c "
import json,sys
rows=[json.loads(l) for l in open('$OUT/fast_$m.jsonl') if l.strip()]
r=max((r for r in rows if r['mode']=='async'), key=lambda r: r['fps'])
print(r['batch'], r['inflight'], r['out_format'])")
    log "best $m: batch $B inflight $I $F x3"
    for rep in 1 2 3; do
      sleep 15
      $PY hailo/07_board_fast.py --model hailo/hef/$m.hef --imgsz "$(wh $m)" --images "$IMG" \
          --split splits/holdout.txt -n 64 --frames 1000 --batch "$B" --inflight "$I" --out-format "$F" \
          --json "$OUT/best_$m.jsonl" 2>&1 | sed -n 3p | tee -a "$OUT/best.txt"
    done
  done
}

do_ap() {
  : "${GT:?GT=<KITTI training/label_2 폴더> 를 지정하세요}"
  for m in $MODELS; do
    log "ap $m"
    rm -rf "runs/fast_pred_$m"
    $PY hailo/07_board_fast.py --model hailo/hef/$m.hef --imgsz "$(wh $m)" --images "$IMG" \
        --split splits/holdout.txt -n 500 --frames 500 --warmup 0 --source disk --conf 0.001 \
        --batch 4 --inflight 2 --dump "runs/fast_pred_$m" 2>&1 | tail -3
    $PY scripts/kitti_eval.py --gt "$GT" --pred "runs/fast_pred_$m" --split splits/holdout.txt \
        --tag "board_fast_holdout_$m" --out "$OUT/ap40_fast_holdout_$m.json" --jsonl "$OUT/results.jsonl" \
        2>&1 | tail -7 | tee -a "$OUT/ap_holdout.txt"
  done
}

do_demo() {
  log "demo"
  $PY hailo/08_demo.py --models hailo/hef/B_1280x384.hef --source kitti --images "$IMG" \
      --split splits/holdout.txt --rate 10 --headless --seconds 20 --save "$OUT/demo_B.mp4" 2>&1 | tail -2 | tee "$OUT/demo.txt"
  $PY hailo/08_demo.py --models hailo/hef/D_640x192.hef --source kitti --images "$IMG" \
      --split splits/holdout.txt --rate 0 --headless --seconds 20 2>&1 | tail -2 | tee -a "$OUT/demo.txt"
}

do_soak() {
  for m in D_640x192 B_1280x384; do
    log "soak $m 30분"
    sleep 60
    $PY hailo/08_demo.py --models hailo/hef/$m.hef --source kitti --images "$IMG" --split splits/holdout.txt \
        --rate 0 --headless --seconds "${SOAK_SEC:-1800}" --stats-csv "$OUT/soak_$m.csv" 2>&1 | tail -3 | tee -a "$OUT/soak.txt"
    vcgencmd get_throttled | tee -a "$OUT/soak.txt"
  done
}

env_info
case "$MODE" in
  check) do_check ;;
  sweep) do_sweep ;;
  best)  do_best ;;
  ap)    do_ap ;;
  demo)  do_demo ;;
  soak)  do_soak ;;
  all)   do_check; do_sweep; do_best; do_ap; do_demo ;;
  *) echo "모드: check | sweep | best | ap | demo | soak | all"; exit 1 ;;
esac
vcgencmd measure_temp | tee -a "$OUT/run.log"; vcgencmd get_throttled | tee -a "$OUT/run.log"
$PY hailo/summarize_final.py > /dev/null 2>&1 || true
log "끝 — $OUT 폴더를 커밋해서 올려주세요 (mp4 는 50MB 넘으면 빼고)"
