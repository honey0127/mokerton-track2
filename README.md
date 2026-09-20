# mokerton-track2 — 임베디드 On-device AI 최적화 챌린지 [트랙2 지정주제]

자율주행 객체인지(Object Detection) / KITTI / Hailo-10H 이식.

- 예선 마감 **2026-11-01 23:59** (보고서 PDF + 모델 파일 + 재현 코드)
- 공식 평가: **KITTI Moderate AP40** (Car / Pedestrian / Cyclist 평균)
- 고정 평가셋: `splits/eval_val.txt` **1,000장** — 학습 포함 시 즉시 실격

## 빠른 시작

```bash
pip install -r requirements.txt
make selftest          # 데이터 없이 돌아가는 검증 (평가셋 규격 + AP40 규칙 14종)
make data KITTI_ROOT=data/kitti_raw
make train
make quantize
make bench-all
make eval MODEL=models/best_1280x384_op13_int8-mixed.onnx
```

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

> **인접 프레임 주의.** KITTI는 연속 주행 프레임을 포함하므로 eval 인덱스의 ±1~3 프레임이
> 학습에 남으면 낙관적 결과가 나온다. 다만 eval_val이 stride 7.5로 촘촘해 가드 비용이 매우 크다
> (`--guard 0` 6,481장 / `1` 4,486장 / `2` 2,492장 / `3` 498장).
> 공지는 eval 인덱스 **본체만** 금지하므로 기본값은 `--guard 0`이며, 모델 선택은 holdout으로 하고
> eval_val은 최종 수치 산출에만 쓴다.

## 클래스 매핑

학습 시 `Van -> Car`, `Person_sitting -> Pedestrian` 병합(기본 ON, `--no-merge-neighbors`로 해제).
근거: 공식 eval이 Car 판정에서 Van을, Pedestrian 판정에서 Person_sitting을 **무시**로 처리해
FP로 세지 않는다. 병합하면 recall은 오르고 오탐 페널티는 없다.
`Truck / Tram / Misc / DontCare`는 라벨을 만들지 않는다. DontCare는 평가 단계에서
원본 `label_2`로부터 직접 읽어 예측 필터링에 쓴다.

## 해상도

KITTI 원본 1242x375. 정사각 640x640은 letterbox 패딩에 NPU 연산을 낭비한다.
공지 상한은 가로 1280, 32의 배수, 직사각형 권장.

| 프로파일 | 해상도 | GFLOPs (YOLO11n) | 비고 |
|---|---|---|---|
| 정밀도 | `1280x384` | 7.94 | 패딩 거의 0 (1242x375 -> 1272x384) |
| 초경량 | `640x192` | 1.98 | 연산량 1/4 |

`scripts/common.py`의 `parse_imgsz`가 32의 배수와 가로 1280 상한을 강제 검증한다.

## 스크립트

| 파일 | 역할 |
|---|---|
| `scripts/common.py` | **전처리/후처리 단일 소스.** letterbox·decode·NMS. 여기 외에 중복 구현 금지 |
| `scripts/make_split.py` | 고정 평가셋 검증 + leakage 없는 train/holdout 생성 |
| `scripts/kitti_to_yolo.py` | KITTI -> YOLO 라벨 변환 (이웃 클래스 병합) |
| `scripts/train_kitti.py` | `rect=True` 직사각형 fine-tuning |
| `scripts/export_onnx.py` | ONNX export. **opset 13 고정**, `--imgsz WxH` 직사각형 |
| `scripts/check_onnx.py` | opset / 로드 가능 여부 / 입출력 형상 검증 |
| `scripts/head_nodes.py` | Detect 디코드 경로 노드 자동 추출 (양자화 제외 목록) |
| `scripts/quantize_ort.py` | 정적 INT8 PTQ. `--op-types` / `--exclude` / `--calib-method` |
| `scripts/diag_outputs.py` | FP32 대비 출력 텐서 분포 비교 (클래스 점수 소실 진단) |
| `scripts/bench_onnx.py` | 결정론적 지연/메모리 측정, jsonl 누적 |
| `scripts/model_stats.py` | params / GFLOPs (해상도별) |
| `scripts/predict_kitti.py` | ONNX -> KITTI 포맷 예측 덤프 |
| `scripts/kitti_eval.py` | **Moderate AP40** 평가 + 규칙 자체검증 |

## 측정 방법론

코드가 방법론을 강제한다 (`bench_onnx.py` 상수).

- batch=1 고정. 자율주행 실시간 인지는 처리량이 아니라 지연이 지표
- warm-up **30회** 제외 -> **300회** 반복 -> 정렬 후 p50/p95/p99
- 입력 시드 고정 `np.random.default_rng(0)`
- ORT: `intra_op` 명시, `inter_op=1`, `ORT_SEQUENTIAL`, 최적화 레벨 ALL
- 메모리: psutil peak RSS
- 보고 지표는 평균이 아니라 **p50 + p99**. 최악 프레임 지연이 안전과 직결
- 전체 측정을 **3회 반복**. 단발 측정의 10% 미만 차이는 결론으로 삼지 않는다

스레드 조건: 주 **12** / 보조 **4**(Raspberry Pi 5의 4코어 모사).
vCPU 16을 초과하면(22스레드) 컨텍스트 스위칭으로 p99가 6배 악화된다.

## AP40 평가 규칙

`kitti_eval.py`가 공식 `eval.cpp`의 판정 규칙을 재현한다. **Ultralytics `val()`로는 이 수치를 낼 수 없다.**

1. 난이도 — Moderate: 높이≥25px, occlusion≤1, truncation≤0.30
2. 이웃 클래스(Van/Person_sitting)는 GT로도 FP로도 세지 않는다
3. 난이도 미달 GT는 FN이 아니고, 거기 맞은 예측도 FP가 아니다
4. DontCare 영역과 IoA≥0.5인 예측은 버린다
5. 높이 미달 예측은 FP로 세지 않는다
6. IoU 임계값 — Car 0.70 / Pedestrian 0.50 / Cyclist 0.50
7. AP40 — recall 1/40..40/40의 40개 지점 보간 precision 평균

`python scripts/kitti_eval.py --selftest`로 위 규칙 14종을 단위 검증한다.

> 최종 제출 전 공식 KITTI devkit 바이너리와 **교차 검증**할 것. 이 구현은 예측 우선 greedy
> 매칭을 쓰며 공식 구현의 GT 우선 매칭과 미세하게 다를 수 있다.

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

## 환경

| 구분 | 사양 |
|---|---|
| 호스트 | Windows 11 Home |
| 실행 | WSL2 / Ubuntu 24.04.4 LTS |
| CPU | Intel Core Ultra 7 155H |
| WSL vCPU | **16** (`.wslconfig`의 `processors=16`) — 재현에 필수 기재 |
| WSL 메모리 | 24GB (물리 31.6GB) |
| GPU | Intel Arc iGPU. **NVIDIA 없음 -> CUDA 미사용** |

> WSL2는 P/E 코어 구분을 게스트에 노출하지 않는다. `lscpu`는 균질한 16 vCPU로만 보고한다.

venv 2개 분리 운영: `.venv-train`(Ultralytics/ONNX) / `.venv-hailo`(DFC 전용).
numpy·protobuf 버전이 충돌하므로 **절대 합치지 않는다.**

**제약**: WSL2에서 DFC는 GPU를 사용할 수 없다(공식 문서 명시). Hailo 고급 최적화
(optimization_level 3~4, Adaround/QFT)는 CPU로는 매우 느리다.
대비책 — 팀원 NVIDIA GPU 활용, 또는 Colab(Ubuntu x86_64 + T4)에 DFC whl 설치 검증.

**참고 목표치**: Hailo Model Zoo 기준 YOLO11n / Hailo-10H = 303 FPS.
