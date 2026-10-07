#!/usr/bin/env bash
set -e
TRAIN=$HOME/mokerton/.venv-train/bin/python
HAILO=$HOME/mokerton/.venv-hailo/bin/python
cd $HOME/mokerton

for SZ in 512 416; do
  echo ""
  echo "=============== imgsz=$SZ ==============="
  ONNX=models/baseline/yolo11n_${SZ}_op13.onnx

  $TRAIN scripts/export_onnx.py --model yolo11n --imgsz $SZ --opset 13
  $TRAIN scripts/hailo_end_nodes.py $ONNX > hailo/end_nodes_${SZ}.txt
  echo "end nodes: $(wc -l < hailo/end_nodes_${SZ}.txt)"

  $HAILO scripts/hailo_parse.py --onnx $ONNX \
      --end-nodes-file hailo/end_nodes_${SZ}.txt \
      --name yolo11n_${SZ} --hw-arch hailo10h --imgsz $SZ \
      --out hailo/yolo11n_${SZ}.har

  $HAILO scripts/hailo_optimize.py --har hailo/yolo11n_${SZ}.har \
      --calib $HOME/mokerton/calib/coco128 \
      --out hailo/yolo11n_${SZ}_quantized.har \
      --imgsz $SZ --limit 64 --opt-level 0

  $HAILO scripts/hailo_compile.py --har hailo/yolo11n_${SZ}_quantized.har \
      --out models/hef/yolo11n_${SZ}_h10h.hef \
      --out-har hailo/yolo11n_${SZ}_compiled.har
done
