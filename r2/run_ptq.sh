#!/bin/bash
# 2차 라운드 PTQ 후보: 양자화 -> holdout 채점 -> (클라우드 2스레드) 상대 속도.  usage: run_ptq.sh jobs.txt
# jobs.txt 한 줄: 이름|FP32 모델|제외목록|방법|추가인자
cd /home/claude/mokerton-track2
. /home/claude/venv/bin/activate
while IFS='|' read -r name fp32 excl meth extra; do
  [ -z "$name" ] && continue; [[ "$name" == \#* ]] && continue
  out=r2/models/$name.onnx
  echo "=== $name $(date +%T)"
  if [ ! -f $out ]; then
    python scripts/quantize_ort.py --model $fp32 --calib data/kitti_yolo/images/train --exclude $excl \
      --calib-method $meth $extra --out $out > r2/logs/q_$name.log 2>&1 || { echo "FAIL quant $name"; tail -3 r2/logs/q_$name.log; continue; }
  fi
  if [ ! -f r2/ap40/$name.json ]; then
    python scripts/predict_kitti.py --model $out --images data/kitti_yolo/images/holdout --split splits/holdout.txt \
      --out runs/r2pred_$name --threads 2 --jsonl r2/results.jsonl > r2/logs/p_$name.log 2>&1 || { echo "FAIL pred $name"; continue; }
    python scripts/kitti_eval.py --gt data/kitti_raw/training/label_2 --pred runs/r2pred_$name --split splits/holdout.txt \
      --tag $name --out r2/ap40/$name.json --jsonl r2/results.jsonl > r2/logs/e_$name.log 2>&1 || echo "FAIL eval $name"
    grep '공식 점수' r2/logs/e_$name.log
  fi
done < "$1"
echo "ALL DONE $(date +%T)"
