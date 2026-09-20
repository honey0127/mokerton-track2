# 예선 재현 경로 — 심사위원이 이 파일 하나로 전 과정을 재현할 수 있어야 한다.
# 사용: make setup && make data && make train && make bench-all

KITTI_ROOT ?= data/kitti_raw
YOLO_DIR   ?= data/kitti_yolo
WEIGHTS    ?= runs/kitti_1280/weights/best.pt
IMGSZ      ?= 1280x384
THREADS    ?= 12
REPEAT     ?= 3
MODELS     := models
STEM       := $(shell basename $(WEIGHTS) .pt)_$(subst x,x,$(IMGSZ))
FP32       := $(MODELS)/$(shell basename $(WEIGHTS) .pt)_$(shell echo $(IMGSZ)|cut -dx -f1)x$(shell echo $(IMGSZ)|cut -dx -f2)_op13_fp32.onnx

.PHONY: setup splits data train export quantize bench-all eval selftest clean

setup:
	pip install -r requirements.txt

selftest:                   ## 데이터 없이 돌아가는 검증
	python scripts/make_split.py --verify splits/eval_val.txt
	python scripts/kitti_eval.py --selftest

splits:
	python scripts/make_split.py --emit
	python scripts/make_split.py --write-splits

data: splits               ## KITTI -> YOLO 변환 (KITTI_ROOT 필요)
	python scripts/kitti_to_yolo.py --kitti-root $(KITTI_ROOT) --out $(YOLO_DIR) --splits splits

train:
	python scripts/train_kitti.py --imgsz $(shell echo $(IMGSZ)|cut -dx -f1) --epochs 100

export:
	python scripts/export_onnx.py --weights $(WEIGHTS) --imgsz $(IMGSZ) --opset 13
	python scripts/check_onnx.py $(FP32)
	python scripts/model_stats.py --weights $(WEIGHTS) --imgsz $(IMGSZ)

quantize: export
	python scripts/head_nodes.py $(FP32) > $(MODELS)/exclude.txt
	python scripts/quantize_ort.py --model $(FP32) --calib $(YOLO_DIR)/images/train --imgsz $(IMGSZ)
	python scripts/quantize_ort.py --model $(FP32) --calib $(YOLO_DIR)/images/train --imgsz $(IMGSZ) --op-types Conv
	python scripts/quantize_ort.py --model $(FP32) --calib $(YOLO_DIR)/images/train --imgsz $(IMGSZ) --exclude $(MODELS)/exclude.txt

bench-all:
	@for m in $(MODELS)/*.onnx; do \
	  python scripts/bench_onnx.py $$m --imgsz $(IMGSZ) --threads $(THREADS) --repeat $(REPEAT); \
	  python scripts/bench_onnx.py $$m --imgsz $(IMGSZ) --threads 4 --repeat $(REPEAT); \
	done

eval:                      ## 공식 점수. MODEL= 로 대상 지정
	python scripts/predict_kitti.py --model $(MODEL) --images $(YOLO_DIR)/images/eval_val \
	  --split splits/eval_val.txt --imgsz $(IMGSZ) --out runs/pred_$(notdir $(basename $(MODEL)))
	python scripts/kitti_eval.py --gt $(KITTI_ROOT)/training/label_2 \
	  --pred runs/pred_$(notdir $(basename $(MODEL))) --split splits/eval_val.txt \
	  --tag $(notdir $(basename $(MODEL)))

clean:
	rm -rf runs/pred_* benchmarks/ap40_*.json
