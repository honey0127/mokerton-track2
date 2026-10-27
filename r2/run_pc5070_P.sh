#!/bin/bash
# 2차 트랙 P (P1~P3) 를 공용 PC 에서 B·D 기준 FP32 (r2/pc5070/models/base_*) 로 다시 만든다.
# 비교 기준: 같은 PC 의 base_*_int8-mixed-percentile (r2/S_prune_holdout.md 에서 B 96.70 / D 90.15)
set -u
cd /d/mokerton-track2
PY=.venv-train/Scripts/python.exe
O=r2/pc5070
M=$O/models
CAL=data/kitti_yolo/images/train
mkdir -p $O/ap40 $O/logs

score() {  # $1 = onnx path
  t=$(basename "$1" .onnx)
  [ -f $O/ap40/$t.json ] && return
  $PY scripts/predict_kitti.py --model "$1" --images data/kitti_yolo/images/holdout --split splits/holdout.txt \
    --out runs/pcpred_$t --threads 2 --jsonl $O/results.jsonl > $O/logs/p_$t.log 2>&1 || { echo "FAIL pred $t"; return; }
  $PY scripts/kitti_eval.py --gt data/kitti_raw/training/label_2 --pred runs/pcpred_$t --split splits/holdout.txt \
    --tag $t --out $O/ap40/$t.json --jsonl $O/results.jsonl > $O/logs/e_$t.log 2>&1 || echo "FAIL eval $t"
  grep -a '공식 점수' $O/logs/e_$t.log
}

q() {  # $1 = 출력 이름 꼬리, 나머지 = quantize_ort.py 추가 인자
  tail=$1; shift
  out=$M/${base}_op13_${tail}.onnx
  echo "--- $tail $(date +%T)"
  [ -f $out ] || $PY scripts/quantize_ort.py --model $fp --calib $CAL --calib-method percentile --out $out "$@" \
      > $O/logs/q_${base}_$tail.log 2>&1 || { echo "FAIL quant $tail"; tail -3 $O/logs/q_${base}_$tail.log; return; }
  score $out
}

for base in base_B_1280x384 base_D_640x192; do
  echo "=== $base $(date +%T)"
  fp=$M/${base}_op13_fp32.onnx
  ex=$M/exclude_${base%_*}_${base##*_}.txt
  # P1: 박스 경로 FP32
  $PY scripts/head_nodes.py $fp --box-fp32 > $M/exclude_${base}_box.txt
  q P1_box --exclude $M/exclude_${base}_box.txt
  # P3: 보정 조정
  q P3_p9999   --exclude $ex --percentile 99.99
  q P3_p999999 --exclude $ex --percentile 99.9999
  q P3_n512    --exclude $ex --n-calib 512
  q P3_n1024   --exclude $ex --n-calib 1024
  out=$M/${base}_op13_P3_entropy.onnx
  [ -f $out ] || $PY scripts/quantize_ort.py --model $fp --calib $CAL --calib-method entropy --exclude $ex --out $out \
      > $O/logs/q_${base}_P3_entropy.log 2>&1 || echo "FAIL quant entropy"
  [ -f $out ] && score $out
  # P2: 층별 민감도 (holdout 16장) 하위 k 모듈 FP32
  $PY scripts/sensitivity.py --fp32 $fp --qdq $M/${base}_op13_int8-mixed-percentile.onnx --base-exclude $ex \
      --images data/kitti_yolo/images/holdout --n 16 --topk 2,4,8 --outdir $M > $O/logs/sens_$base.log 2>&1 || echo "FAIL sens"
  for k in 2 4 8; do
    q P2_sens$k --exclude $M/exclude_${base}_op13_int8-mixed-percentile_sens_top$k.txt
  done
done
echo "ALL DONE $(date +%T)"
