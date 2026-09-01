# 트랙2 지정주제 — 실험 환경 명세 및 Baseline (2026-09-01 기준)

예선 보고서(11/1 마감)의 "실험 환경 명세" / "벤치마크 및 측정 방법" / "Baseline 대비 정량 비교" 섹션의 원본 기록.

---

## 1. 실험 환경 명세

| 구분 | 사양 |
|---|---|
| 호스트 OS | Windows 11 Home |
| 실행 환경 | WSL2 / Ubuntu 24.04.4 LTS (noble) |
| CPU | Intel Core Ultra 7 155H |
| **WSL vCPU 할당** | **16** (`.wslconfig`의 `processors=16`) |
| WSL 메모리 할당 | 24GB (물리 31.6GB) |
| GPU | Intel Arc iGPU (내장). **NVIDIA GPU 없음 → CUDA 미사용** |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cpu |
| torchvision | 0.28.0+cpu |
| ONNX | 1.22.0 |
| ONNX Runtime | 1.29.0 (CPUExecutionProvider) |
| Ultralytics | 8.4.135 |

> **주의**: WSL2는 P-코어/E-코어 구분을 게스트에 노출하지 않는다. `lscpu`는 균질한 16 vCPU(8코어×2스레드)로만 보고한다. vCPU 개수 16은 `.wslconfig`로 우리가 지정한 값이므로 **환경 명세에 반드시 기재**해야 재현이 가능하다.

## 2. 측정 방법론

- **워크로드**: batch=1 고정 (자율주행 실시간 인지 → 처리량이 아닌 지연이 지표)
- **절차**: warm-up 30회 제외 → 300회 반복 → 정렬 후 p50/p95/p99 산출
- **입력**: `np.random.default_rng(0)` seed 고정
- **ORT 설정**: `intra_op_num_threads` 명시 고정, `inter_op=1`, `ORT_SEQUENTIAL`, 최적화 레벨 ALL
- **메모리**: `psutil` peak RSS
- **보고 지표**: 평균이 아닌 **p50 + p99**. 자율주행에서는 최악 프레임 지연이 안전과 직결

### 스레드 수 확정 (3회 반복 측정)

| threads | p50 (3회) | p99 (3회) | 판정 |
|---|---|---|---|
| 4 | 22.7 / 22.9 / 23.3 | 32.8 / 33.8 / 35.8 | p50 동률, 꼬리 나쁨 |
| **12** | 22.6 / 23.1 / 23.4 | **26.9 / 27.0 / 27.4** | **채택 (재현성 최고)** |
| 16 | 21.7 / 21.9 / 22.5 | 24.9 / 24.9 / 35.1 | p50 최고나 꼬리 불안정 |
| 22 | 39.3 / 54.0 / 50.3 | 136 / 176 / 165 | 붕괴 (vCPU 16 초과 오버서브스크립션) |

**확정**: 주 조건 threads=12 / 보조 조건 threads=4 (Raspberry Pi 5의 4코어 모사)

**발견 1** — p50은 4스레드에서 이미 포화. 스레드 증가는 처리량이 아니라 **꼬리(p99)를 개선**한다 (34ms → 27ms, 3회 일관).
**발견 2** — vCPU 수를 초과하면 컨텍스트 스위칭으로 p99가 6배 악화된다.
**방법론 교훈** — 1회 200런 측정에서 관찰된 10% 미만 차이(threads 6·8의 "역행")는 3회 반복에서 재현되지 않았다. 단발 측정의 소폭 차이를 결론으로 삼지 않는다.

## 3. Baseline 및 양자화 결과 (YOLO11n, 640×640, opset 13)

모델: Ultralytics YOLO11n COCO 사전학습 가중치. ONNX export 시 `nms=False, dynamic=False, simplify=True` (Hailo가 NMS 없는 raw head를 요구).
평가 데이터셋: **coco128 (128장, 임시)** — 지정 벤치마크 공지 전 파이프라인 검증용. 보고서에는 사용하지 않음.
캘리브레이션: coco128 128장, letterbox 전처리 일치, MinMax, per-channel, QDQ.

| 전략 | p50(12t) | 배속 | 크기 | peak RSS | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| FP32 baseline | 24.44 ms | 1.00× | 10.74 MB | 166.5 MB | 0.670 | 0.502 |
| INT8 전체 양자화 | 12.63 ms | 1.93× | 3.22 MB | 104.9 MB | **0.000** | **0.000** |
| INT8 Conv만 | 21.89 ms | 1.12× | 3.18 MB | 168.9 MB | 0.658 | 0.495 |
| **INT8 mixed (채택)** | **14.23 ms** | **1.72×** | **3.26 MB** | **112.1 MB** | **0.650** | **0.484** |

## 4. 핵심 트러블슈팅 기록 (보고서 소재)

### 4-1. per-channel PTQ는 ONNX opset 13 이상 필요
opset 11로 export한 모델에 `per_channel=True`를 적용하면 `DequantizeLinear`에 opset 13 문법인 `axis` 속성이 생성되어 그래프 로드가 실패한다(`INVALID_GRAPH`). baseline export 사양을 **opset 13**으로 상향하여 해결. Hailo DFC v5는 opset 13을 지원하므로 이식성 손실 없음.

### 4-2. 전체 양자화 시 클래스 점수 완전 소실 → mAP 0
YOLO 출력 `output0 [1,84,8400]`은 행 0~3이 **박스 좌표(0~640 픽셀)**, 행 4~83이 **클래스 확률(0~1)** 로 스케일이 2~3자릿수 다른 값이 한 텐서에 섞여 있다. 이를 단일 스케일로 uint8 양자화하면 스케일이 약 2.5(=640/255)로 잡혀 **클래스 점수가 전부 0으로 뭉개진다.**

진단 (동일 이미지 1장의 출력 통계):

| 모델 | cls max | conf>0.25 앵커 수 |
|---|---|---|
| FP32 | 0.92945 | 49 |
| INT8 전체 | **0.00000** | **0** |
| INT8 Conv만 | 0.93249 | 51 |
| INT8 mixed | 0.91861 | 48 |

### 4-3. Conv-only 양자화는 정확도를 지키지만 속도를 잃는다
비-Conv 연산이 모두 FP32로 남으면 ORT가 양자화 서브그래프를 통째로 fusion하지 못하고 Conv 경계마다 int8↔float 변환을 삽입한다. 결과적으로 속도 이득이 1.12×로 축소되고 **peak RSS가 FP32(166.5MB)보다 오히려 증가(168.9MB)** — 변환 버퍼 때문.

### 4-4. 채택안 — Detect 모듈의 디코드 경로만 FP32 유지
`/model.23/` 내부의 non-Conv 연산 + 그래프 출력 생성 노드 총 59개를 `nodes_to_exclude`로 제외하고 나머지는 전부 양자화. 속도 1.72×, 정확도 손실 mAP50-95 -3.6%(상대).

## 5. 저장소 구조

`~/mokerton` (GitHub: `honey0127/mokerton-track2`, private)

| 파일 | 역할 |
|---|---|
| `scripts/export_onnx.py` | YOLO → ONNX export (`--imgsz`, `--opset`), 파일명에 스펙 인코딩 |
| `scripts/bench_onnx.py` | 결정론적 지연/메모리 측정, jsonl 누적 기록 |
| `scripts/quantize_ort.py` | 정적 INT8 PTQ (letterbox 캘리브레이션, `--op-types`, `--exclude`) |
| `scripts/check_onnx.py` | opset·로드 가능 여부·입출력 형상 검증 |
| `scripts/diag_outputs.py` | FP32/INT8 출력 텐서 분포 비교 진단 |
| `scripts/head_nodes.py` | Detect 모듈 디코드 경로 노드 자동 추출 |
| `benchmarks/results.jsonl` | 모든 측정 원본 |

venv 2개 분리 운영: `.venv-train`(Ultralytics/ONNX) / `.venv-hailo`(DFC 전용). numpy·protobuf 버전이 충돌하므로 절대 합치지 않는다.

## 6. 미결 및 다음 단계

- [ ] **Hailo Developer Zone 가입 및 승인** — 크리티컬 패스. DFC v5.3.0 + HailoRT v5.3.0 (Hailo-10H는 v5.x 전용, v3.x는 Hailo-8용)
- [ ] **지정 벤치마크 공지 확인** (9월 첫째 주) — 데이터셋 확정 전까지 coco128 결과로 튜닝하지 않는다. 128장 표본에서 mAP 0.01 차이는 유의하지 않음
- [ ] 캘리브레이션 방법 비교 (MinMax vs Percentile vs Entropy)
- [ ] Layer-wise sensitivity analysis → INT4/INT8/FP16 mixed precision (Hailo-10H가 셋 다 지원)
- [ ] 해상도 스윕 (640/512/416) → Pareto 곡선
- [ ] FP32 ONNX → Hailo DFC → `.hef` 컴파일 성공 + profiler 예상 성능 (예선 보고서 "이식 계획"의 결정적 차별화 요소)
- [ ] 팀원 PC 사양 취합 — NVIDIA GPU 보유자가 있으면 학습·DFC 최적화 전담

**제약**: WSL2에서는 DFC가 GPU를 사용할 수 없다(공식 문서 명시). Hailo의 고급 최적화(optimization_level 3~4, Adaround/QFT)는 CPU로 돌리면 매우 느리다. 대비책 — 팀원 NVIDIA GPU 활용, 또는 Colab(Ubuntu x86_64 + T4)에 DFC whl 설치 검증.

**참고 목표치**: Hailo Model Zoo 기준 YOLO11n / 640 / Hailo-10H = **303 FPS (mAP 39.0)**. 현재 PC INT8이 70 FPS이므로 본선 NPU 이식 시 약 4~5× 추가 이득이 기대치.
