# 예선 재현 경로 — 심사위원이 이 파일 하나로 전 과정을 재현할 수 있어야 한다.
# 사용: make setup && make data && make train && make quantize && make bench-all && make eval-all && make summary
# 640 프로파일: make train IMGSZ=640x192 → make quantize WEIGHTS=runs/kitti_640/weights/best.pt IMGSZ=640x192
# 모든 명령은 저장소 루트에서 실행한다.

KITTI_ROOT ?= data/kitti_raw
YOLO_DIR   ?= data/kitti_yolo
IMGSZ      ?= 1280x384
LONG       := $(shell echo $(IMGSZ)|cut -dx -f1)
WEIGHTS    ?= runs/kitti_$(LONG)/weights/best.pt
EPOCHS     ?= 100
BATCH      ?= 16
THREADS    ?= 12
# 측정 라운드 수. bench_suite.py 가 라운드마다 (모델, 스레드) 순서를 섞어 1회씩 잰다
ROUNDS     ?= 5
# thread-sweep 대상 (기본: 1280 FP32 와 INT8-mixed)
SWEEP      ?= $(MODELS)/$(WSTEM)_$(IMGSZ)_op13_fp32.onnx $(MODELS)/$(WSTEM)_$(IMGSZ)_op13_int8-mixed.onnx
NCALIB     ?= 256
SPLIT      ?= eval_val
TOPK       ?= 2,4,8
# 측정 원본. 학습 전 리허설은 JSONL=benchmarks/rehearsal.jsonl 로 분리해 보고서 표와 섞지 않는다
JSONL      ?= benchmarks/results.jsonl
MODELS     := models
WSTEM      := $(shell basename $(WEIGHTS) .pt)
FP32       := $(MODELS)/$(WSTEM)_$(IMGSZ)_op13_fp32.onnx
EXCL       := $(MODELS)/exclude_$(WSTEM)_$(IMGSZ).txt
EXCL_BOX   := $(MODELS)/exclude_$(WSTEM)_$(IMGSZ)_box.txt
CALIB      := $(YOLO_DIR)/images/train
Q          := python scripts/quantize_ort.py --model $(FP32) --calib $(CALIB) --n-calib $(NCALIB) --imgsz $(IMGSZ)

.PHONY: setup splits data train export quantize quantize-extra sensitivity bench-all thread-sweep eval eval-all summary selftest clean

setup:
	pip install -r requirements.txt

selftest:                   ## 데이터 없이 돌아가는 검증 (평가셋 규격 + 공식 AP40 규칙 17종)
	python scripts/make_split.py --verify splits/eval_val.txt
	python scripts/kitti_eval.py --selftest

splits:
	python scripts/make_split.py --emit
	python scripts/make_split.py --write-splits

data: splits               ## KITTI -> YOLO 변환 (KITTI_ROOT 필요). 윈도우면 COPY=--copy
	python scripts/kitti_to_yolo.py --kitti-root $(KITTI_ROOT) --out $(YOLO_DIR) --splits splits $(COPY)

train:                     ## GPU PC 에서 실행. 기본 정사각+mosaic (train_kitti.py 설명 참조)
	python scripts/train_kitti.py --imgsz $(LONG) --epochs $(EPOCHS) --batch $(BATCH)

export:
	python scripts/export_onnx.py --weights $(WEIGHTS) --imgsz $(IMGSZ) --opset 13
	python scripts/check_onnx.py $(FP32)
	python scripts/model_stats.py --weights $(WEIGHTS) --imgsz $(IMGSZ) --jsonl $(JSONL)

quantize: export           ## A 계열: 전체 INT8(실패 재현) / Conv-only / mixed(채택안)
	python scripts/head_nodes.py $(FP32) > $(EXCL)
	$(Q)
	$(Q) --op-types Conv
	$(Q) --exclude $(EXCL)

quantize-extra:            ## 추가 실험: 박스 경로 FP32 / 캘리브레이션 방식 비교 (quantize 다음에)
	python scripts/head_nodes.py $(FP32) --box-fp32 > $(EXCL_BOX)
	$(Q) --exclude $(EXCL_BOX) --suffix boxfp32
	$(Q) --exclude $(EXCL) --calib-method percentile
	$(Q) --exclude $(EXCL) --calib-method entropy

sensitivity:               ## 층별 민감도 -> 하위 k개 모듈 FP32 유지 모델 (holdout 이미지 사용)
	python scripts/sensitivity.py --fp32 $(FP32) --qdq $(MODELS)/$(WSTEM)_$(IMGSZ)_op13_int8-mixed.onnx \
	  --base-exclude $(EXCL) --images $(YOLO_DIR)/images/holdout --n 16 --topk $(TOPK)
	@for k in $(shell echo $(TOPK) | tr , ' '); do \
	  $(Q) --exclude $(MODELS)/exclude_$(WSTEM)_$(IMGSZ)_op13_int8-mixed_sens_top$$k.txt --suffix sens$$k; \
	done

bench-all:                 ## 지연·메모리. 성능 담당자 PC 한 대에서만. 모델·스레드 순서를 섞어 ROUNDS 라운드
	python scripts/bench_suite.py $(MODELS)/*.onnx --threads $(THREADS),4 --rounds $(ROUNDS) --jsonl $(JSONL)

thread-sweep:              ## 주 스레드 조건 재확인 (FP32·INT8-mixed, 2~16 스레드, 기록은 따로)
	python scripts/bench_suite.py $(SWEEP) --threads 2,4,6,8,12,16 --rounds 3 --runs 150 \
	  --tag thread-sweep --jsonl benchmarks/thread_sweep.jsonl --raw benchmarks/thread_sweep_rounds.jsonl

# eval 의 선택 인자: EXTRA="--drop-stride 32" SUFFIX=_noP5  -> P5 헤드를 뗀 것처럼 채점 (재학습 없는 근사)
EXTRA      ?=
SUFFIX     ?=
ETAG        = $(notdir $(basename $(MODEL)))$(SUFFIX)

eval:                      ## 점수 1개. MODEL=models/xxx.onnx  (튜닝·선택은 SPLIT=holdout, 보고 수치는 기본 eval_val)
	python scripts/predict_kitti.py --model $(MODEL) --images $(YOLO_DIR)/images/$(SPLIT) \
	  --split splits/$(SPLIT).txt --out runs/pred_$(SPLIT)_$(ETAG) --jsonl $(JSONL) $(EXTRA)
	python scripts/kitti_eval.py --gt $(KITTI_ROOT)/training/label_2 \
	  --pred runs/pred_$(SPLIT)_$(ETAG) --split splits/$(SPLIT).txt \
	  --tag $(ETAG) --out benchmarks/ap40_$(SPLIT)_$(ETAG).json --jsonl $(JSONL)

eval-all:                  ## models/ 의 모든 ONNX 채점 (SPLIT=holdout 가능)
	@for m in $(MODELS)/*.onnx; do $(MAKE) --no-print-directory eval MODEL=$$m SPLIT=$(SPLIT) JSONL=$(JSONL); done

summary:                   ## results.jsonl -> 보고서 표 (Drop Rate, 속도배율 포함). SPLIT 별로 따로 만든다
	python scripts/summarize.py --jsonl $(JSONL) --split splits/$(SPLIT).txt --out benchmarks/summary_$(SPLIT).md \
	  --plot benchmarks/pareto_$(SPLIT).png

clean:
	rm -rf runs/pred_*
