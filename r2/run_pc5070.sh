#!/bin/bash
# 공용 PC(RTX 5070 / i9-14900KF)에서 B·D 기준 + 프루닝 4개를 같은 환경으로 export -> INT8(mixed-percentile) -> holdout 채점
set -u
cd /d/mokerton-track2
PY=.venv-train/Scripts/python.exe
O=r2/pc5070
mkdir -p $O/w $O/models $O/ap40 $O/logs
cp -n /d/ibzago/mokerton-train/runs/kitti_1280_p34/weights/best.pt $O/w/base_B.pt
cp -n /d/ibzago/mokerton-train/runs/kitti_640_p34/weights/best.pt  $O/w/base_D.pt
for n in B_m75 B_m60 D_m75 D_m60; do cp -n runs/$n/weights/best.pt $O/w/S_$n.pt; done
sha256sum $O/w/*.pt > $O/weights.sha256

score() {  # $1 = onnx path
  t=$(basename "$1" .onnx)
  [ -f $O/ap40/$t.json ] && return
  $PY scripts/predict_kitti.py --model "$1" --images data/kitti_yolo/images/holdout --split splits/holdout.txt \
    --out runs/pcpred_$t --threads 2 --jsonl $O/results.jsonl > $O/logs/p_$t.log 2>&1 || { echo "FAIL pred $t"; return; }
  $PY scripts/kitti_eval.py --gt data/kitti_raw/training/label_2 --pred runs/pcpred_$t --split splits/holdout.txt \
    --tag $t --out $O/ap40/$t.json --jsonl $O/results.jsonl > $O/logs/e_$t.log 2>&1 || echo "FAIL eval $t"
  grep -a '공식 점수' $O/logs/e_$t.log
}

for spec in base_B:1280x384 S_B_m75:1280x384 S_B_m60:1280x384 base_D:640x192 S_D_m75:640x192 S_D_m60:640x192; do
  name=${spec%%:*}; sz=${spec##*:}
  echo "=== $name $sz $(date +%T)"
  fp=$O/models/${name}_${sz}_op13_fp32.onnx
  q=$O/models/${name}_${sz}_op13_int8-mixed-percentile.onnx
  [ -f $fp ] || $PY scripts/export_onnx.py --weights $O/w/$name.pt --imgsz $sz --opset 13 --outdir $O/models > $O/logs/x_$name.log 2>&1 || { echo "FAIL export $name"; continue; }
  $PY scripts/head_nodes.py $fp > $O/models/exclude_${name}_${sz}.txt
  [ -f $q ] || $PY scripts/quantize_ort.py --model $fp --calib data/kitti_yolo/images/train --exclude $O/models/exclude_${name}_${sz}.txt \
      --calib-method percentile --out $q > $O/logs/q_$name.log 2>&1 || { echo "FAIL quant $name"; tail -3 $O/logs/q_$name.log; continue; }
  score $fp
  score $q
done
echo "ALL DONE $(date +%T)"
