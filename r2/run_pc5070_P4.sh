#!/bin/bash
# 트랙 P4 (P1~P3 상위 두 가지 조합) + 보정 seed 재현성 확인. 공용 PC, run_pc5070_P.sh 와 같은 조건.
# B 상위: P3_n512 97.15, P3_n1024 96.95 (같은 축이라 조합 불가) -> 다음 P1_box / P2_sens4 (둘 다 96.74) 와 각각 조합
# D 상위: P3_entropy 90.87, P1_box 90.71 -> 조합
set -u
cd /d/mokerton-track2
PY=.venv-train/Scripts/python.exe
O=r2/pc5070
M=$O/models
CAL=data/kitti_yolo/images/train

score() {  # $1 = onnx path
  t=$(basename "$1" .onnx)
  [ -f $O/ap40/$t.json ] && return
  $PY scripts/predict_kitti.py --model "$1" --images data/kitti_yolo/images/holdout --split splits/holdout.txt \
    --out runs/pcpred_$t --threads 2 --jsonl $O/results.jsonl > $O/logs/p_$t.log 2>&1 || { echo "FAIL pred $t"; return; }
  $PY scripts/kitti_eval.py --gt data/kitti_raw/training/label_2 --pred runs/pcpred_$t --split splits/holdout.txt \
    --tag $t --out $O/ap40/$t.json --jsonl $O/results.jsonl > $O/logs/e_$t.log 2>&1 || echo "FAIL eval $t"
  grep -aoE '"mAP/Moderate": *[0-9.]+' $O/ap40/$t.json
}

q() {  # $1 = 출력 이름 꼬리, $2 = calib method, 나머지 = quantize_ort.py 추가 인자
  tail=$1; meth=$2; shift 2
  out=$M/${base}_op13_${tail}.onnx
  echo "--- $tail $(date +%T)"
  [ -f $out ] || $PY scripts/quantize_ort.py --model $fp --calib $CAL --calib-method $meth --out $out "$@" \
      > $O/logs/q_${base}_$tail.log 2>&1 || { echo "FAIL quant $tail"; tail -n 3 $O/logs/q_${base}_$tail.log; return; }
  score $out
}

# D 먼저 (빠름)
base=base_D_640x192; echo "=== $base $(date +%T)"
fp=$M/${base}_op13_fp32.onnx; ex=$M/exclude_base_D_640x192.txt; box=$M/exclude_${base}_box.txt
q P4_entropy_box   entropy    --exclude $box
q P3_entropy_s1    entropy    --exclude $ex --seed 1
q P3_entropy_s2    entropy    --exclude $ex --seed 2
q base_s1          percentile --exclude $ex --seed 1

base=base_B_1280x384; echo "=== $base $(date +%T)"
fp=$M/${base}_op13_fp32.onnx; ex=$M/exclude_base_B_1280x384.txt; box=$M/exclude_${base}_box.txt
q P4_n512_box      percentile --exclude $box --n-calib 512
q P4_n512_sens4    percentile --exclude $M/exclude_${base}_op13_int8-mixed-percentile_sens_top4.txt --n-calib 512
q P3_n512_s1       percentile --exclude $ex --n-calib 512 --seed 1
q P3_n512_s2       percentile --exclude $ex --n-calib 512 --seed 2
q base_s1          percentile --exclude $ex --seed 1
echo "ALL DONE $(date +%T)"
