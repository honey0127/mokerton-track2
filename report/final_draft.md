# [트랙 2-본선] 최종 결과 보고서 — 초안 (2026-11-21)

> 공식 양식이 나오면 이 내용을 옮긴다. 【측정 후】 표시는 보드 측정(`hailo/run_board_final.sh`)이 끝나면
> `hailo/results/board/final/SUMMARY.md` 의 값으로 채운다. 숫자는 모두 저장소 파일에서 왔다 (괄호에 출처).

## 1. 팀 정보
팀명 Tranogang · 팀원 【이름·학번】 · 대표 연락처 【】 · 저장소 https://github.com/honey0127/mokerton-track2
(브랜치 `final/hailo-convert`, 태그 【final 태그】)

## 2. 결과 요약

| | 1280x384 (정확도 모드) | 640x192 (속도 모드) |
|---|---|---|
| 최종 모델 | 【A 또는 B, DECISION_FINAL 판정】 | 【D 또는 C】 |
| 보드 정확도 (holdout / eval_val, Moderate AP40) | 【】 | 【】 |
| 보드 처리 속도 (전처리~NMS 포함) | 【】 FPS | 【】 FPS |
| NPU 단독 | 199.6~201.9 FPS | 365.5~419.7 FPS |
| 자원 | CPU 【】%, RSS 【】 MB, SoC 【】°C | 【】 |

핵심 세 가지
1. 예선 FP32 모델 4종(A·B·C·D)을 Hailo-10H HEF 로 모두 변환했고, 변환 직전(FP) 점수가 ONNX FP32 와 소수점까지
   같아 전처리·디코드 경로를 검증했다 (`hailo/RESULTS.md`).
2. 보드에서 한 장씩 처리하면 NPU 가 대부분 쉬었다 (B: NPU 단독 202 FPS, 전체 18 FPS). 디코드를 다시 짜고
   (결과 동일, PC 기준 1280x384 약 10ms → 0.6~1.6ms, 보드 【】ms) 전처리·NPU·후처리를 겹쳐 돌려 【18 → 】 FPS 로 올렸다.
3. 예선에서 주력이던 P5 제거(B·D)는 M1 INT8 에서 1.08배 빨랐지만, Hailo NPU 단독으로는 1280x384 이득이 1.2% 였다
   (칩 분할 수가 A·B 모두 5개). 640x192 는 1.15배. 사전 등록한 규칙대로 【결과】.

## 3. 예선 대비 변경 사항

| 항목 | 예선 (서버 CPU) | 본선 (엣지) | 이유 |
|---|---|---|---|
| 실행 장치 | ONNX Runtime CPU (M1 Pro 4스레드로 측정) | Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H), HailoRT 5.1.1 | 대회 본선 규정 |
| 양자화 | ORT 정적 INT8, QDQ, Detect 디코드 FP32 유지, Percentile 보정, 7비트 가중치 | Hailo DFC 5.4 INT8 (`optimization_level=0`), 보정 train 256장(실사용 64장) | HEF 는 DFC 로만 만든다. 같은 FP32 ONNX 에서 출발 |
| 그래프 | 전체 (디코드 포함) | Detect 마지막 Conv 4개(B·D) / 6개(A·C)에서 자름. DFL·sigmoid·NMS 는 Pi CPU | Hailo 권장 구성, 예선 3.3절 계획대로 |
| 정규화 | CPU 에서 /255 | HEF 안 (`normalization` 층) | 전처리 CPU 시간 절약 |
| 후처리 | 모든 앵커 DFL 후 거름 | 클래스 점수로 먼저 거르고 남은 앵커만 DFL (`hailo/hailo_rt.py`) | 결과 동일 (4모델 x conf 3종 확인), PC 기준 6~20배 빠름 |
| 실행 방식 | 모델 추론만 측정 | 전처리·NPU(run_async)·후처리 병렬, batch·동시 진행 수 조정 | 실제 시연 조건 |
| 모델 선택 | B·D (holdout + M1 지연) | 【DECISION_FINAL.md 판정】 | 같은 규칙을 보드 수치로 다시 적용 |

## 4. 엣지 이식 과정

1. **ONNX → HAR (parse)**: end node 를 Detect 의 마지막 Conv 로 지정 (`hailo/end_nodes*.txt`).
2. **양자화 (optimize)**: train 보정 256장(eval_val 교집합 검사 통과, `hailo/results/<모델>/calib_filelist.txt`).
   GPU(RTX 5070, Blackwell)가 DFC 의 TensorFlow 와 맞지 않아 CPU 로만 돌렸고, 그래서 가장 가벼운 level 0 으로 시작했다.
3. **에뮬레이터 채점**: holdout 500장. FP 단계 = ONNX FP32 (A 97.22, B 97.17, C 92.41, D 91.67) 로 경로 검증,
   INT8 손실 A −0.72, B −1.14, C −1.87, D −1.23.
4. **compile → HEF**: 네 모델 모두 칩 한 번에 들어가지 않아 여러 조각(context)으로 나뉜다. 원인은 LCU 자원
   (P5 있음 142, 없음 118 / 한도 80) 으로, 해상도가 아니라 구조가 정한다. 공식 YOLO11n 도 142 였다. 그래서 P5 를
   지워도 조각 수가 줄지 않았다 (1280: 5→5, 640: 4→4).
5. **보드 구동**: HailoRT InferModel API, 입력 uint8 NHWC. 11/19 직렬 측정 → 병목 확인 → 비동기 파이프라인.
6. **겪은 문제**: Windows 체크아웃의 CRLF 로 WSL 스크립트 실패, profiler 가 compiled HAR 를 봐야 context 정보가 나옴,
   양자화된 HAR 에 model script append 불가, COCO 모델 parse 의 대화형 질문 (`hailo/RESULTS.md` '알려진 함정').

## 5. 실측 결과 (보드)

### 5.1 측정 방법 — 예선과 같은 원칙
- 정확도: holdout 500장 / eval_val 1000장, 예선과 같은 채점 엔진(`scripts/kitti_eval.py`, KITTI 공식 AP40 포팅), Moderate.
- 속도: NPU 단독(`hailortcli run`, 3회), 처리 전체(`07_board_fast.py`: 예열 20장 제외 600장, 최고 조합 3회 중앙값).
  예선의 "모델만" 지연에 대응하는 값이 NPU 단독이고, 시연 속도에 대응하는 값이 처리 전체다.
- 자원: 프로세스 CPU%·RSS, SoC 온도, `vcgencmd get_throttled` (스로틀링 없을 때만 유효).
- 발열: 30분 연속 구동 (`run_board_final.sh soak`) 【측정 후】.

### 5.2 처리 단계별 시간 (11/19, 한 장씩 직렬, ms)

| | 읽기 | 전처리 | NPU | 디코드 | NMS | 합계 | FPS |
|---|---|---|---|---|---|---|---|
| A 1280x384 | 13.2 | 4.9 | 27.2 | 12.0 | 0.6 | 58.0 | 17.2 |
| B 1280x384 | 10.7 | 5.0 | 26.9 | 11.3 | 0.6 | 54.5 | 18.3 |
| C 640x192  | 10.8 | 1.7 | 9.1 | 3.2 | 0.4 | 25.2 | 39.7 |
| D 640x192  | 10.8 | 1.7 | 8.7 | 2.7 | 0.4 | 24.2 | 41.3 |

(`hailo/results/board/e2e_summary.txt`) NPU 단독 처리량(B 202 FPS = 장당 5ms)에 비해 한 장씩 동기 호출한 NPU 구간이
27ms 로 길다. 읽기(PNG 디코드)는 카메라 입력에는 없는 구간이다.

### 5.3 개선 후 【측정 후 — SUMMARY.md 2절】

### 5.4 정확도 【측정 후 — SUMMARY.md 1절】

## 6. 서버 vs 엣지 비교

| 모델 | 서버 INT8 holdout | 보드 holdout | 서버 INT8 eval_val | 보드 eval_val | M1 INT8 추론 ms | 보드 NPU 단독 ms (1/FPS) | 보드 처리 전체 FPS |
|---|---|---|---|---|---|---|---|
| A | 96.76 | 【07】 (11/19: 96.85) | 93.74 | 93.65 | 21.52 | 5.01 | 【】 |
| B | 96.57 | 【07】 (96.24) | 93.22 | 94.29 | 19.93 | 4.95 | 【】 |
| C | 90.25 | 【07】 (90.16) | 89.67 | 89.35 | 6.51 | 2.74 | 【】 |
| D | 90.27 | 【07】 (90.48) | 88.99 | 88.36 | 6.02 | 2.38 | 【】 |

서버 = 예선 ONNX Runtime int8-mixed-percentile (정확도 Xeon VNNI, 지연 M1 Pro 4스레드).
정확도는 서버와 엣지가 ±1점 안에서 비슷하고 【해석】, 속도는 NPU 단독 기준 1280x384 에서 약 4배, 640x192 에서 약 2.5배 빠르다.
보드 eval_val 은 11/19 에 네 모델 모두 채점했고 모델 선택에는 쓰지 않았다 (DECISION_FINAL.md).

## 7. 한계점
- 양자화를 가장 가벼운 설정(level 0)으로 했다. level 1·2, 박스 출력 16비트 시험 결과 【측정 후】.
- 보정 이미지 seed 반복을 하지 않았다. 예선에서 seed 만으로 holdout 이 0.45~1.64점 움직였으므로 0.3점 수준 차이는 잡음일 수 있다.
- 보드 holdout(11/19)이 에뮬레이터와 최대 0.38점 달랐다. 원인 【conf 설정 확인 후】.
- 보드 eval_val 을 결정 규칙 작성 전에 이미 봤다. 그래서 규칙에서 eval_val 을 선택 근거에서 뺐다.
- 측정은 보드 1대, 실내 온도 한 조건.

## 8. 시연
`hailo/08_demo.py`: 카메라(또는 KITTI 이미지) → Hailo-10H → 박스 + FPS·NPU 지연·CPU·온도 화면 표시,
m 키로 정확도 모드(1280x384) ↔ 속도 모드(640x192) 전환. 카메라 화면은 가운데 띠를 KITTI 비율로 잘라 넣는다.

## 9. 제출물·재현
- HEF: `hailo/hef/*.hef` (+ `HEF.sha256`), 변환 스크립트 `hailo/01~05, 09`, 설정 `hailo/results/<모델>/model_script.alls`
- 보드 재현: `IMG=... GT=... bash hailo/run_board_final.sh all` → `python3 hailo/summarize_final.py`
- 시연: `python3 hailo/08_demo.py --models hailo/hef/<1280>.hef,hailo/hef/<640>.hef --source picam --crop kitti`
- 결정 기록: `DECISION.md`, `DECISION_R2.md` (예선), `DECISION_FINAL.md` (본선)
