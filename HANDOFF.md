# 인계 메모 (2026-09-20, tranogang)

레포에 이미 있던 것(`scripts/common.py`, `quantize_ort.py`, `head_nodes.py`,
`splits/`, README 트러블슈팅 기록)은 **하나도 건드리지 않았다.** 아래 4개만 추가했다.

| 추가된 것 | 내용 |
|---|---|
| `scripts/analyze_box_sizes.py` | GT 박스 크기 → P3/P4/P5 헤드 유휴도 실측 |
| `results/box_sizes.md` | **그 실측 결과 + 해석 + 검증할 예측** ← 먼저 읽을 것 |
| `hailo/` (4개 파일) | 본선용 Hailo DFC 파이프라인 (ONNX → `.hef`) |
| `HANDOFF.md` | 이 파일 |

---

## 1. 지금 가장 중요한 결과

`results/box_sizes.md` 에 전부 있지만 한 줄로:

> **1280×384 에서 P5 헤드는 유휴가 아니다 (Moderate GT 의 16.69%).
> 640×192 로 내리면 1.28% 로 떨어진다. 13배 차이.**

즉 **"P5 제거"는 해상도와 묶여야만 성립한다.** 1280×384 를 쓸 거면 P5 는 남겨야
하고, P5 를 떼려면 640×192 로 가야 한다. 둘을 따로 결정하면 안 된다.

이게 README 의 해상도 표(정밀도 1280×384 / 초경량 640×192)와 직접 연결된다.
초경량 프로파일은 "해상도만 낮춘 것"이 아니라 **"해상도 + 헤드를 같이 줄이는 것"**
이 되어야 일관성이 생긴다.

## 2. 다음에 할 일 — ablation 4개

`results/box_sizes.md` 의 표 그대로:

| # | 구성 | 해상도 |
|---|---|---|
| A | YOLO11n baseline | 1280×384 |
| B | P5 헤드 제거 | 1280×384 |
| C | YOLO11n baseline | 640×192 |
| D | P5 헤드 제거 | 640×192 |

핵심 비교는 **B vs D**. 분석이 맞다면 B 는 깨지고 D 는 거의 안 깨진다.
그리고 B 에서 **Car 가 가장 많이 깨져야 한다** (예측, 틀릴 수도 있음).

구성마다 기록: params / GFLOPs → `scripts/model_stats.py`,
p50·p99 → `scripts/bench_onnx.py`, Moderate AP40 → `scripts/kitti_eval.py`.

아직 **학습을 한 번도 안 돌렸다.** A 부터 필요하다.

## 3. 기계 분담 제안

README 환경 섹션에 "NVIDIA 없음 → CUDA 미사용 / 대비책: 팀원 NVIDIA GPU 활용"
이라고 적혀 있는데, 그 GPU 머신이 지금 준비됐다.

| | 학과 공용 PC (tranogang) | 기존 개발 머신 |
|---|---|---|
| CPU | Intel i9-14900KF | Intel Core Ultra 7 155H |
| GPU | **RTX 5070 12GB** (driver 591.86) | Intel Arc iGPU |
| RAM | 31.8GB | 31.6GB |
| torch | 2.11.0+cu128 (sm_120 OK) | — |
| 역할 | **학습 / QAT** | 양자화 실험 · 지연 측정 · 보고서 |

**학습은 GPU 머신에서, 지연 측정은 한 머신에서만.** 데스크톱 지연은 예선 심사
항목이라 머신이 섞이면 수치를 비교할 수 없다. 기존 머신의 `bench_onnx.py` 조건
(vCPU 16, 주 12 / 보조 4 스레드)을 계속 기준으로 쓰는 게 맞다.

**머신 간 교환 단위는 `.pt` 가 아니라... 아니, `.pt` 다.** 두 머신의
Ultralytics / ONNX 버전이 다르다 (8.4.135 / 1.22.0 vs 8.4.173 / 1.23.1).
`.onnx` 를 건너주면 어느 버전으로 뽑았는지가 수치에 섞인다. **`best.pt` 를
넘기고 export 는 측정하는 머신에서 한 번만** 하는 게 안전하다.

### RTX 50 시리즈 환경 메모 (삽질 기록)

- Blackwell 은 **sm_120** 이라 구버전 PyTorch 가 `no kernel image is available
  for execution on the device` 로 죽는다. cu128 이상 휠 필요.
- Hailo DFC 는 import 시 TensorFlow 로 GPU 테스트를 하는데 Blackwell 에서
  `CUDA_ERROR_INVALID_HANDLE` 로 죽는다 → `export CUDA_VISIBLE_DEVICES=''`.
  `hailo/03_hailo_compile.sh` 에 이미 들어가 있다.

## 4. 본선 준비 — 지금 시작해야 하는 것

`hailo/README.md` 참고. 요점은 **예선 모델이 확정되기 전에 컴파일 경로를 뚫어둘 것.**
`.onnx` → `.hef` 가 안 나오면 본선에서 아무것도 못 한다.

특히: **P5 를 제거하면 Hailo parse 의 end node 가 6개 → 4개로 바뀐다.**
Model Zoo 의 yolov8s 설정을 그대로 못 쓴다. `hailo/01_export_onnx.py` 가
end node 를 자동으로 찾아준다.

## 5. 지키고 있는 선

- 공식 eval 1,000장은 **학습 / PTQ 캘리브레이션 / 모델·하이퍼파라미터 선택**
  어디에도 쓰지 않는다. 최종 수치 산출만. 모델 선택은 `holdout.txt` 로 한다.
  - `analyze_box_sizes.py` — `eval_val` 파일명이 들어오면 중단
  - `hailo/02_make_calib.py` — `--eval-list` 로 교집합 검사, 걸리면 중단
- 학과 공용 PC 에 **다른 팀(경쟁 팀) 작업 폴더가 있다.** 읽기만 했고 수정·이동·
  삭제 없음. 거기서 가져온 것은 공개 KITTI 데이터와 주최측이 배포한
  `eval_val.txt` 뿐이다. 그쪽 코드 / 가중치 / 측정값 / 분석은 쓰지 않았고,
  이 레포에도 들어있지 않다. 위 `results/box_sizes.md` 수치는 전부 우리 split
  에서 직접 측정한 것이다.
- 작업 폴더는 `C:\Users\PC00\ibzago\` (공용 PC 기준).

## 6. 남은 TODO

- [ ] A 구성 학습 (YOLO11n / 1280×384) — 아직 시작 안 함
- [ ] B·C·D ablation, `results/box_sizes.md` 의 예측 검증
- [ ] `.hef` 컴파일 1회 성공시키기 (모델 확정 전에)
- [ ] `kitti_eval.py` 를 공식 KITTI devkit 바이너리와 교차 검증 (README 에도 적혀 있음)
- [ ] 보고서 초안 — 예선 마감 **2026-11-01 23:59**
