# GT 박스 크기 분포 → 탐지 헤드(P3/P4/P5) 유휴도 실측

측정일 2026-09-20 · `scripts/analyze_box_sizes.py` · 측정자 tranogang (RTX 5070 머신)

## 왜 이걸 쟀는가

"P5 헤드를 떼면 파라미터와 연산이 줄어든다"는 건 맞지만, **떼도 되는지**는
데이터에 큰 객체가 실제로 얼마나 있는지에 달려 있다. 추측 대신 우리 split 에서
직접 셌다. 아키텍처를 줄이는 결정의 근거가 이 표다.

## 측정 조건

| 항목 | 값 |
|---|---|
| 대상 | `splits/train.txt` + `splits/holdout.txt` = **6,481장** (공식 eval 1,000장 제외) |
| GT 필터 | **Moderate** (높이 > 25px, occlusion ≤ 1, truncation ≤ 0.30) — 공지 지정 지표 |
| GT 개수 | **17,671개** (Car 13,615 / Pedestrian 3,096 / Cyclist 960) |
| 클래스 | Car / Pedestrian / Cyclist (이웃 클래스 미포함) |
| 헤드 배정 | 모델 입력 좌표계에서 `sqrt(bbox 면적)` 기준, 경계 64 / 128 px |
| 좌표 변환 | 이미지마다 실제 원본 해상도를 읽어 letterbox 배율 개별 계산 |

> 경계 64/128 은 FPN/FCOS 계열의 관례값이다. 절대 규칙이 아니므로 보고서에는
> 경계값을 명시하고, 민감도가 걱정되면 `--bounds` 를 바꿔 몇 번 더 돌릴 것.
>
> **eval_val.txt 는 쓰지 않았다.** 평가셋 분포를 보고 아키텍처를 고르면
> 그 셋에 맞춘 설계가 되고, 사실상 평가셋 오버핏이다. 스크립트가 파일명으로
> 이를 막는다(`eval_val` 이 들어오면 즉시 중단).

## 결과

### 전체 (Moderate GT 17,671개)

| 해상도 | letterbox 배율 | P3 (stride 8) | P4 (stride 16) | P5 (stride 32) | **P5 비율** | 중위 높이 |
|---|---|---|---|---|---|---|
| **1280×384** | 1.025 | 9,088 | 5,634 | 2,949 | **16.69%** | 56.5px |
| **640×192** | 0.513 | 14,722 | 2,722 | 227 | **1.28%** | 28.3px |

### 클래스별 P5 비율

| 클래스 | Moderate GT | P5 @ 1280×384 | P5 @ 640×192 |
|---|---|---|---|
| Car | 13,615 | **18.97%** | 1.65% |
| Pedestrian | 3,096 | 8.88% | 0.03% |
| Cyclist | 960 | 9.48% | 0.21% |
| 전체 | 17,671 | 16.69% | 1.28% |

## 해석 — 예상과 반대였고, 그게 더 중요하다

**1. 1280×384 에서 P5 는 유휴가 아니다.**
Moderate GT 의 16.69%, 2,949개가 P5 담당이다. 이 해상도에서 P5 를 떼면
그 2,949개에서 recall 손실이 난다. "큰 객체가 거의 없으니 P5 는 낭비"라는
직관은 이 해상도에서 **틀렸다**.

**2. P5 유휴도를 결정하는 건 아키텍처가 아니라 해상도다.**
같은 데이터, 같은 GT 인데 비율이 16.69% → 1.28%, **13배** 차이가 난다.
letterbox 배율이 1.025 → 0.513 으로 절반이 되면서 모든 객체가 한 단계
작은 헤드로 밀렸기 때문이다. 즉 **"P5 제거"는 단독 선택지가 아니고
"640×192 + P5 제거"로만 성립한다.** 1280×384 를 쓸 거면 P5 는 남겨야 한다.

**3. Car 가 가장 P5 의존적이다.**
1280×384 에서 Car 18.97% vs Pedestrian 8.88% / Cyclist 9.48%.
KITTI 는 근거리 차량이 화면을 크게 차지하므로 당연하지만, 이게 중요한 이유는
**검증 가능한 예측**이 되기 때문이다 →

> **예측:** 1280×384 에서 P5 를 제거하면 Moderate AP40 하락폭이
> **Car > Cyclist ≈ Pedestrian** 순으로 나타난다.

ablation 한 번으로 맞/틀림이 갈린다. 맞으면 분석 → 예측 → 검증 루프가
보고서에 그대로 들어가고, 틀리면 헤드 배정 경계(64/128)가 우리 데이터에
안 맞는다는 뜻이므로 그것도 결과다. 어느 쪽이든 쓸 수 있다.

## 그래서 뭘 할 건가 (검증 대기 중)

| # | 구성 | 해상도 | 확인할 것 |
|---|---|---|---|
| A | YOLO11n 그대로 (baseline) | 1280×384 | 기준 AP40 |
| B | YOLO11n, P5 헤드 제거 | 1280×384 | **예측 1 검증** — Car 가 가장 많이 깨지는가 |
| C | YOLO11n 그대로 | 640×192 | 해상도만 낮춘 효과 |
| D | YOLO11n, P5 헤드 제거 | 640×192 | **예측 2 검증** — C 대비 손실이 거의 없는가 |

핵심 비교는 **B vs D** 다. 위 분석이 맞다면 B 는 유의미하게 깨지고 D 는 거의
안 깨진다. 그러면 "P5 제거는 저해상도에서만 공짜다"를 우리 데이터로 입증한 셈이다.

각 구성마다 기록할 것: params / GFLOPs (`scripts/model_stats.py`),
p50·p99 지연 (`scripts/bench_onnx.py`), Moderate AP40 (`scripts/kitti_eval.py`).

## 재현

```bash
python scripts/analyze_box_sizes.py \
    --label-dir data/kitti_raw/training/label_2 \
    --image-dir data/kitti_raw/training/image_2 \
    --split splits/train.txt splits/holdout.txt \
    --imgsz 384 1280 --json-out results/box_sizes_1280x384.json

python scripts/analyze_box_sizes.py \
    --label-dir data/kitti_raw/training/label_2 \
    --image-dir data/kitti_raw/training/image_2 \
    --split splits/train.txt splits/holdout.txt \
    --imgsz 192 640 --json-out results/box_sizes_640x192.json
```

원래 측정은 `splits/` 가 아직 없던 환경에서 "전체 7,481장 − eval 1,000장 = 6,481장"
으로 돌렸다. `train.txt`(5,981) + `holdout.txt`(500) = 6,481 로 같은 모집단이므로
위 명령이 동일한 수치를 낸다. 숫자가 다르게 나오면 split 이 바뀐 것이니
먼저 `make_split.py --verify` 를 확인할 것.
