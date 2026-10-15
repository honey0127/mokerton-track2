#!/bin/bash
# 프루닝 재학습 경로가 끝까지 도는지 CPU 로 확인 (holdout 일부로 만든 작은 데이터, 1 epoch, imgsz 320).
# 결과 수치는 의미 없다. 확인할 것: trainer 가 프루닝 모델을 그대로 쓰는지, best.pt 저장·재로드, ONNX 내보내기.
set -e
cd /home/claude/mokerton-track2
PY=/home/claude/venv-pt/bin/python
T=/tmp/claude-0/-home-claude/7035cd5b-eed0-5f82-ad32-cf1a64806afa/scratchpad/smoke
rm -rf $T; mkdir -p $T/splits
head -40 splits/holdout.txt > $T/splits/train.txt
sed -n 41,56p splits/holdout.txt > $T/splits/holdout.txt
head -5 splits/eval_val.txt > $T/splits/eval_val.txt
$PY scripts/kitti_to_yolo.py --kitti-root data/kitti_raw --out $T/yolo --splits $T/splits 2>&1 | tail -2
printf "path: $T/yolo\ntrain: images/train\nval: images/holdout\nnames:\n  0: Car\n  1: Pedestrian\n  2: Cyclist\n" > $T/smoke.yaml
$PY scripts/prune_p34.py train --pruned runs/prune/test_1280_m70/pruned.pt --imgsz 320 --data $T/smoke.yaml \
    --epochs 1 --batch 4 --device cpu --workers 0 --name smoke_prune 2>&1 | tail -15
ls -la runs/smoke_prune/weights/
$PY - <<'EOF'
import sys; sys.path.insert(0, 'scripts')
from ultralytics import YOLO
m = YOLO('runs/smoke_prune/weights/best.pt')
print('reload ok', type(m.model.model[2]).__name__)
p = m.export(format='onnx', imgsz=[384, 1280], opset=13, simplify=True, dynamic=False)
print('export ok', p)
EOF
