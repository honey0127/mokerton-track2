# mokerton-track2 — 임베디드 On-device AI 최적화 챌린지 [트랙2 지정주제]

자율주행 객체인지(Object Detection) / KITTI / Hailo-10H 이식.

- 예선 마감 **2026-11-01 23:59** (보고서 PDF + 모델 파일 + 재현 코드)
- 공식 평가: **KITTI Moderate AP40** (Car / Pedestrian / Cyclist 평균)
- 고정 평가셋: `splits/eval_val.txt` **1,000장** — 학습 포함 시 즉시 실격

## 빠른 시작

```bash
pip install -r requirements.txt
make selftest          # 데이터 없이 돌아가는 검증 (평가셋 규격 + 공식 AP40 규칙 17종)
make data KITTI_ROOT=data/kitti_raw          # 윈도우/다른 PC 로 옮길 때: make data COPY=--copy
make train                                   # GPU PC. 640 은 make train IMGSZ=640x192
make quantize                                # 640 은 WEIGHTS=runs/kitti_640/weights/best.pt IMGSZ=640x192
make quantize-extra                          # 박스경로 FP32 / Percentile / Entropy 비교
make sensitivity                             # 층별 민감도 -> 하위 2/4/8 모듈 FP32 유지 모델
make thread-sweep                            # 주 스레드 조건 재확인 (최종 측정 전에 1번)
make bench-all                               # 지연·메모리: 성능 담당 PC 한 대에서만, 순서 섞은 5라운드
make eval-all SPLIT=holdout                  # 고르기·튜닝은 holdout 으로
make eval-all                                # 보고 수치: eval_val 1,000장
make summary                                 # -> benchmarks/summary_eval_val.md (Drop Rate 표)
```

모든 명령은 **저장소 루트에서** 실행한다 (`data/kitti.yaml`의 `path`가 실행 위치 기준이다).

## 데이터 준비

KITTI Object Detection Evaluation 2012에서 **2개 파일만** 받는다.
`data_object_image_2.zip` (~12GB), `data_object_label_2.zip` (~16MB).
Raw Data / Velodyne 포인트 클라우드는 이 태스크에 불필요하므로 받지 않는다.

```
data/kitti_raw/training/image_2/000000.png ...
data/kitti_raw/training/label_2/000000.txt ...
```

## Split 규칙 — leakage 방지의 단일 진입점

`scripts/make_split.py` 외의 어떤 경로로도 split을 만들지 않는다.

| 파일 | 장수 | 용도 |
|---|---|---|
| `splits/eval_val.txt` | 1,000 | **주최측 고정 평가셋.** 학습 절대 금지 |
| `splits/holdout.txt` | 500 | 자체 검증 / 모델 선택 |
| `splits/train.txt` | 5,981 | 학습 |

평가셋 검증 결과(2026-09-20): 1,000개, 중복 0, 범위 0~7480, 전체 7,481장을
**stride 7.5로 균일 샘플링**한 구조. `make_split.py --verify`가 이를 매번 재확인한다.

> **번호가 붙어 있다고 연속 프레임이 아니다 (2026-10-06 정정).** KITTI devkit `readme.txt`는
> `train_rand.txt`를 "학습 이미지에 번호를 매긴 **무작위 순열**"이라고 설명한다 (예: 000000 → 2011_09_28
> drive 106 frame 48). devkit 매핑으로 확인한 결과 번호가 이웃한 두 장이 같은 주행에서 나온 비율은
> **3.0%** 뿐이다. 그래서 `--guard`(eval 번호 ±N 제외)는 무관한 사진만 지울 뿐 효과가 없다 — `--guard 0` 유지.
>
> 실제 상황은 이렇다: eval_val 1,000장의 **100%** 가 train에도 같은 주행(141개 중)의 사진을 갖고 있고,
> **90.2%** 는 같은 주행 ±5프레임 안에 train 사진이 있다. 이는 주최측 split의 성질이라 모든 팀에 공통이며,
> 공지 규정(eval 번호 본체 금지)도 지킨다. 우리 holdout도 같은 성질(±5프레임 90.8%)이므로
> holdout 점수가 eval_val 점수의 좋은 대리 지표가 된다. 보고서 '데이터' 절에 이 사실을 그대로 적는다.
> (devkit 매핑 파일은 분석에만 썼고 학습·평가 파이프라인에는 쓰지 않는다.)

## 클래스 매핑

학습 시 `Van -> Car`, `Person_sitting -> Pedestrian` 병합(기본 ON, `--no-merge-neighbors`로 해제).
근거: 공식 eval이 Car 판정에서 Van을, Pedestrian 판정에서 Person_sitting을 **무시**로 처리해
FP로 세지 않는다. 병합하면 recall은 오르고 오탐 페널티는 없다.
`Truck / Tram / Misc / DontCare`는 라벨을 만들지 않는다. DontCare는 평가 단계에서
원본 `label_2`로부터 직접 읽어 예측 필터링에 쓴다.

## 해상도

KITTI 원본 1242x375. 정사각 640x640은 letterbox 패딩에 NPU 연산을 낭비한다.
공지 상한은 가로 1280, 32의 배수, 직사각형 권장.

| 프로파일 | 해상도 | GFLOPs (YOLO11n 3클래스, fused) | Params | 비고 |
|---|---|---|---|---|
| 정밀도 | `1280x384` | 7.66 | 2.583M | 패딩 거의 0 (1242x375 -> 1272x384) |
| 초경량 | `640x192` | 1.90 | 2.583M | 연산량 1/4 |

계산법: thop MACs x 2, Conv+BN 합친(fused) 그래프 기준 = 실제 배포 ONNX 와 같은 구조.
(이전 표의 7.94/1.98 은 COCO 80클래스·unfused 값이었다. 3클래스는 분류 헤드가 좁아져 값이 다르다.)
구조로만 정해지는 값이라 학습이 끝나도 바뀌지 않지만, `make export` 가 매번 다시 기록한다.

`scripts/common.py`의 `parse_imgsz`가 32의 배수와 가로 1280 상한을 강제 검증한다.

## 학습

`scripts/train_kitti.py` 기본값은 **정사각 캔버스 + mosaic** (`rect=False`).

- Ultralytics 는 `rect=True` 면 mosaic / mixup / cutmix 를 **강제로 끈다**
  (8.4.135 `data/dataset.py` `build_transforms` 확인). 6천 장 남짓한 KITTI 에서 mosaic 없이 학습하면 과적합 위험이 크다.
- YOLO 는 합성곱 네트워크라 정사각으로 학습해도 1280x384 직사각으로 추론할 수 있다. 중요한 건 물체 크기인데,
  둘 다 '긴 변 = imgsz' 로 맞추므로 같다.
- `rect=True` 는 1280 기준 연산이 약 1/3 이라 빠르다. GPU 시간이 부족할 때만 `--rect` 로 쓰고, 쓰면 보고서에 이유를 적는다.
- 모델 선택(val)은 holdout 500장, 공식 점수는 eval_val 1,000장. eval_val 로 하이퍼파라미터를 고르지 않는다.

## 스크립트

| 파일 | 역할 |
|---|---|
| `scripts/common.py` | **전처리/후처리 단일 소스.** letterbox·decode·NMS. 여기 외에 중복 구현 금지 |
| `scripts/make_split.py` | 고정 평가셋 검증 + leakage 없는 train/holdout 생성 |
| `scripts/kitti_to_yolo.py` | KITTI -> YOLO 라벨 변환 (이웃 클래스 병합), 클래스 통계 저장 |
| `scripts/train_kitti.py` | fine-tuning. 기본 정사각+mosaic, `--rect` 선택 (아래 '학습' 참조) |
| `scripts/export_onnx.py` | ONNX export. **opset 13 고정**, `--imgsz WxH` 직사각형 |
| `scripts/check_onnx.py` | opset / 로드 가능 여부 / 입출력 형상 검증 |
| `scripts/head_nodes.py` | Detect 디코드 경로 노드 자동 추출 (양자화 제외 목록) |
| `scripts/quantize_ort.py` | 정적 INT8 PTQ. `--op-types` / `--exclude` / `--calib-method` |
| `scripts/diag_outputs.py` | FP32 대비 출력 텐서 분포 비교 (클래스 점수 소실 진단) |
| `scripts/bench_onnx.py` | 결정론적 지연/메모리 측정, jsonl 누적 |
| `scripts/model_stats.py` | params / GFLOPs (해상도별) |
| `scripts/predict_kitti.py` | ONNX -> KITTI 포맷 예측 덤프 + 전처리/추론/후처리 시간 기록 |
| `scripts/kitti_eval.py` | **Moderate AP40** (공식 판정 엔진) + 규칙 자체검증 17종 |
| `scripts/sensitivity.py` | 층별 양자화 민감도(SQNR) -> FP32 로 남길 모듈 후보와 제외목록 |
| `scripts/summarize.py` | `results.jsonl` -> 보고서 표 (Drop 점수·비율, 속도 배율, params/GFLOPs) |
| `third_party/kitti_object_eval_python/` | 공식 KITTI 평가 포트 (OpenPCDet, MIT) — 2D 평가만 사용 |

## 측정 방법론

코드가 방법론을 강제한다 (`bench_onnx.py` 상수).

- batch=1 고정. 자율주행 실시간 인지는 처리량이 아니라 지연이 지표
- warm-up **30회** 제외 -> **300회** 반복 -> 정렬 후 p50/p95/p99
- 입력 시드 고정 `np.random.default_rng(0)`
- ORT: `intra_op` 명시, `inter_op=1`, `ORT_SEQUENTIAL`, 최적화 레벨 ALL
- 메모리: psutil peak RSS
- 보고 지표는 평균이 아니라 **p50 + p99**. 최악 프레임 지연이 안전과 직결
- **5라운드, 라운드마다 (모델, 스레드) 순서를 섞어 1회씩** 재고 중앙값을 보고한다 (`bench_suite.py`, seed 0).
  측정 1회는 새 프로세스로 띄운다 (peak RSS 에 앞 모델 메모리가 섞이지 않게).
  라운드 간 흔들림 = (최대-최소)/중앙값 을 함께 기록하고, 10% 미만 차이는 결론으로 삼지 않는다
- 측정 조건: 전원 연결, 윈도우 전원 모드 '최고 성능', 다른 앱 종료

왜 순서를 섞나 (2026-10-07 리허설, COCO YOLO11n 1280x384): 모델마다 3회를 **연속**으로 재던 방식에서
같은 모델의 3회 p50 이 최대 17~34% 달랐다 (int8 4스레드 27.54 / 22.31 / 20.51 ms).
노트북 CPU 는 터보·전력 한도·발열로 시간에 따라 속도가 변해서, 연속 측정이면 '언제 쟀는지'가 모델 비교에 섞인다.

스레드 조건: 주 **12** / 보조 **4**(Raspberry Pi 5의 4코어 모사) — 640x640 FP32 스윕(9/1)에서 정한 값.
vCPU 16을 초과하면(22스레드) 컨텍스트 스위칭으로 p99가 6배 악화된다.
**재확인 필요**: 같은 리허설에서 1280x384 는 7개 모델 모두 4스레드 p50 이 12스레드보다 3.5~8.9% 빨랐고,
INT8 6개는 p99 도 4스레드가 같거나 좋았다 (FP32 p99 만 12스레드가 좋음). 흔들림 범위 안이라 아직 결론 아님.
최종 측정 전에 `make thread-sweep` 으로 실제 모델에서 다시 정하고, 정한 조건을 측정 전에 커밋한다.

## AP40 평가 — 공식 판정 엔진 사용

**Ultralytics `val()`로는 이 수치를 낼 수 없다.** `kitti_eval.py`는 `third_party/kitti_object_eval_python`
(공식 C++ devkit `evaluate_object.cpp`를 줄 단위로 옮긴 파이썬 포트, OpenPCDet 수록)으로 채점한다.

교차 검증 (2026-10-06, 합성 KITTI 형식 데이터 5세트 x 1,000장, 3클래스 x 3난이도 = 칸 9개):

| 비교 | 칸별 최대 차이 |
|---|---|
| 공식 C++ (devkit 로직, AP40 판) vs 이 엔진 | **0.005** (소수점 둘째 자리 반올림 오차) |
| 공식 C++ vs 이전 자체 구현 | **1.5점** (클래스별로 위아래 제각각) |

Drop Rate 는 FP32 와 INT8 의 작은 차이를 재는 지표라 1점 안팎의 채점 오차도 결론을 바꿀 수 있어 교체했다.
이전 구현과 달랐던 점: ① 매칭이 '점수 순 greedy'가 아니라 'GT 마다 겹침 최대' ② AP40 의 41개 임계값을
TP 점수에서 골라 임계값마다 TP/FP 를 다시 센다 ③ DontCare 무시 기준이 고정 0.5 가 아니라 클래스 IoU 임계
(Car 0.7) ④ 비교는 `>=` 가 아니라 `>`.

판정 규칙 (보고서 '평가 방법' 절):
1. 난이도 — Moderate: 높이≥25px, occlusion≤1, truncation≤0.30
2. 이웃 클래스(Car 의 Van, Pedestrian 의 Person_sitting)는 FN 도 FP 도 아니다
3. 난이도 미달 GT 는 FN 이 아니고, 거기 맞은 예측도 FP 가 아니다
4. 높이 25px 미만 예측은 FP 로 세지 않는다
5. DontCare — 예측 박스 면적 대비 겹침이 클래스 IoU 임계를 넘으면 버린다
6. IoU 임계 — Car 0.70 / Pedestrian 0.50 / Cyclist 0.50, GT 마다 겹침이 가장 큰 예측을 고른다
7. AP40 — TP 점수로 재현율 1/40 간격의 임계값을 고르고, 1~40번째 정밀도 평균 (GT 가 적으면 0 에 가깝게 나온다)

`python scripts/kitti_eval.py --selftest`로 위 규칙 17종을 엔진 위에서 단위 검증한다.

**실제 예측으로 한 번 더 교차 검증하려면** (선택, 10분): 공식 로직 그대로인 C++ 평가기
`asharakeh/kitti_native_evaluation`(AP40 반영판)을 빌드해 같은 예측 폴더를 넣는다.
```bash
sudo apt install -y libboost-dev cmake g++
git clone https://github.com/asharakeh/kitti_native_evaluation && cd kitti_native_evaluation
g++ -O3 -w -o evaluate_object_3d_offline src/evaluate_object_3d_offline.cpp -Iinclude
mkdir -p /tmp/res/data && cp ~/mokerton/runs/pred_<모델>/*.txt /tmp/res/data/
./evaluate_object_3d_offline ~/mokerton/data/kitti_raw/training/label_2 /tmp/res | grep detection_AP
# 출력: car_detection_AP : Easy Moderate Hard  -> kitti_eval.py 와 소수점 둘째 자리까지 같아야 한다
```

## 트러블슈팅 기록 (보고서 소재)

### 1. per-channel PTQ는 opset 13 이상 필요
opset 11 그래프에 `per_channel=True`를 적용하면 `DequantizeLinear`에 opset13 문법인
`axis` 속성이 생성되어 `INVALID_GRAPH`로 로드가 실패한다. export를 opset 13으로 상향하여 해결.
Hailo DFC v5는 opset 13을 지원하므로 이식성 손실 없음.

### 2. 전체 양자화 시 클래스 점수 완전 소실 -> mAP 0
출력 `[1, 4+nc, A]`는 행 0~3이 **박스 좌표(0~입력폭 픽셀)**, 행 4~가 **클래스 확률(0~1)**로
스케일이 2~3자릿수 다른 값을 한 텐서에 담는다. 단일 스케일로 uint8 양자화하면
스케일이 (입력폭/255)로 잡혀 클래스 점수가 전부 0으로 뭉개진다.

1280x384 재현 결과 (`diag_outputs.py`, 클래스 행의 **고유값 개수**가 결정적 신호):

| 모델 | cls_max | 클래스 행 고유값 수 |
|---|---|---|
| FP32 | 기준 | 1822 |
| INT8 전체 | 0.00000 | **1** (완전 붕괴) |
| INT8 mixed | 보존 | 216 |

### 3. Conv-only 양자화는 정확도를 지키지만 속도를 잃는다
비-Conv 연산이 FP32로 남으면 ORT가 양자화 서브그래프를 fusion하지 못하고
Conv 경계마다 int8<->float 변환을 삽입한다. 속도 이득이 축소되고 변환 버퍼 때문에
peak RSS가 FP32보다 오히려 증가한다.

### 4. 채택안 — Detect 모듈의 디코드 경로만 FP32 유지
`head_nodes.py`가 Detect 접두사를 자동 탐지해 내부 non-Conv 연산 + 그래프 출력 생성 노드를
`nodes_to_exclude`로 넘긴다. YOLO11n/1280x384에서 **59개** 노드가 제외된다.

남은 위험: 박스 회귀의 마지막 Conv(`cv2.*.2`)와 DFL Conv 는 여전히 INT8 이다. KITTI Car 는 IoU 0.7 이라
몇 픽셀 오차로도 TP 가 FP 로 바뀐다. `head_nodes.py --box-fp32` (63개 제외) 로 만든 `int8-mixed-boxfp32` 와
정확도·지연을 비교한다 (`make quantize-extra`).

### 5. Percentile / Entropy 캘리브레이션은 이미지 수에 비례해 메모리가 폭증한다
ORT 히스토그램 캘리브레이터는 모든 이미지의 중간 텐서를 메모리에 쌓은 뒤 한 번에 히스토그램을 만든다.
1280x384, 2 vCPU 컨테이너 실측 (2026-10-06):

| 방식 | 4장 | 16장 | 16장 + `--chunk 4` |
|---|---|---|---|
| MinMax | 855MB | 862MB | 849MB (스케일 294개 완전 동일) |
| Percentile | 1,618MB | 5,774MB | 1,622MB |
| Entropy | 1,593MB | 5,796MB | 1,600MB |

장당 약 350MB 씩 늘어 256장이면 약 90GB 가 필요하다 (WSL 24GB 에서 불가능). `quantize_ort.py --chunk 8`(기본)이
ORT 의 `CalibStridedMinMax` 로 8장씩 끊어 수집해 상한을 고정한다. MinMax 는 결과가 완전히 같고, Percentile 은
스케일 차이 최대 1.4%, Entropy 는 히스토그램 칸 나누기가 달라져(첫 묶음 기준 칸 폭 유지) 일부 텐서 스케일이 달라진다.

### 6. 데이터 yaml 의 상대경로는 '실행 위치' 기준이다
Ultralytics 는 `path:` 상대경로를 yaml 파일 위치가 아니라 현재 작업 폴더(실패 시 `datasets_dir`) 기준으로 푼다.
`path: ../data/kitti_yolo` 는 저장소 루트에서 실행하면 저장소 바깥을 찾아 학습이 시작되지 않았다.
`path: data/kitti_yolo` 로 고치고, `train_kitti.py` 가 루트 기준 절대경로로 바꾼 사본을 넘긴다.

## Hailo-10H 컴파일 (R3, 본선 대비)

Ultralytics 8.4.135 에 `format="hailo"` 내보내기가 들어 있다 (`engine/exporter.py` `export_hailo` 확인).
DFC 5.x(Hailo-10H 용)가 같은 파이썬 환경에 있어야 하고, Linux x86_64(WSL2 포함)에서만 된다.

```python
from ultralytics import YOLO
YOLO("runs/kitti_1280/weights/best.pt").export(
    format="hailo", name="hailo10h", imgsz=[384, 1280],          # 직사각형 지원 (LetterBox new_shape)
    data="data/kitti.yaml", split="train", fraction=0.2)  # 저장소 루트에서. 보정 이미지 약 1,200장 (eval_val 금지)
# -> runs/kitti_1280/weights/best_hailo_model/best.hef, metadata.yaml, nms_config.json
```

내부 동작 (보고서 소재): ONNX 를 Detect 의 **마지막 Conv 6개**(`cv2.i.2` 박스, `cv3.i.2` 클래스)에서 자르고,
클래스 출력에 sigmoid 를 붙인 뒤 DFL·박스 디코드·NMS 는 `nms_postprocess(..., engine=cpu)` 로 칩 밖에서 한다.
최적화는 `optimization_level=2` + `post_quantization_optimization(finetune)`.
**우리 ORT INT8-mixed 의 경계(디코드 경로만 FP32)와 같은 자리다** — 트러블슈팅 2·4 의 결론이 Hailo 공식 경로와 일치한다.

주의: `.venv-hailo` 에 ultralytics 를 같은 버전으로 설치해야 한다(DFC 의 numpy/protobuf 고정을 깨지 않게 확인).
GPU 없이 돌리면 finetune 단계가 매우 느리거나 DFC 가 최적화 수준을 낮출 수 있다 — 로그를 확인하고 보고서에 적는다.

## 환경

| 구분 | 사양 |
|---|---|
| 호스트 | Windows 11 Home |
| 실행 | WSL2 / Ubuntu 24.04.4 LTS |
| CPU | Intel Core Ultra 7 155H |
| WSL vCPU | **16** (`.wslconfig`의 `processors=16`) — 재현에 필수 기재 |
| WSL 메모리 | 24GB (물리 31.6GB) |
| GPU | Intel Arc iGPU. **NVIDIA 없음 -> CUDA 미사용** |
| Python | 3.12.3 |
| PyTorch / torchvision | 2.13.0+cpu / 0.28.0+cpu |
| ONNX / ONNX Runtime | 1.22.0 / 1.29.0 (CPUExecutionProvider) |
| Ultralytics | 8.4.135 |

> WSL2는 P/E 코어 구분을 게스트에 노출하지 않는다. `lscpu`는 균질한 16 vCPU로만 보고한다.

venv 2개 분리 운영: `.venv-train`(Ultralytics/ONNX) / `.venv-hailo`(DFC 전용).
numpy·protobuf 버전이 충돌하므로 **절대 합치지 않는다.**

버전 고정 (requirements.txt): ultralytics 8.4.135 / onnx 1.22.0 / onnxruntime 1.29.0.
학습 PC(GPU)와 측정 PC(WSL)는 같은 버전을 쓴다. 지연·메모리는 **이 표의 PC 한 대에서만** 잰다
(다른 PC 수치를 한 표에 섞지 않는다). 학습은 팀원 NVIDIA PC 에서 하고 `best.pt` 만 넘겨받는다.

**제약**: WSL2에서 DFC는 GPU를 사용할 수 없다(공식 문서 명시). Hailo 고급 최적화
(optimization_level 3~4, Adaround/QFT)는 CPU로는 매우 느리다.
대비책 — 팀원 NVIDIA GPU 활용, 또는 Colab(Ubuntu x86_64 + T4)에 DFC whl 설치 검증.

**참고 목표치**: Hailo Model Zoo 기준 YOLO11n / Hailo-10H = 303 FPS.
