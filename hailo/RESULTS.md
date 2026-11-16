# 본선 Hailo-10H 변환 결과 (2026-11-16)

DFC v5.4.0, WSL2 Ubuntu (CPU 전용, i9-14900KF). 파이프라인은 `README.md` 순서 그대로:
`02_make_calib.py` → `03_hailo_compile.sh` → `04_emulate_eval.py`.

- 양자화: `optimization_level=0, compression_level=0` (가장 가벼운 설정), 보정 세트는 train 256장
  (eval_val 교집합 검사 통과, 실제 사용 64장). 목록은 `results/<모델>/calib_filelist.txt`.
- 채점: holdout 500장, Hailo SDK 에뮬레이터. `fp` = 양자화 전(SDK_FP_OPTIMIZED),
  `quantized` = NPU 비트 정확도 에뮬레이션(SDK_QUANTIZED). 후처리는 `scripts/common.py` 의
  `decode_yolo` 그대로, 채점은 `scripts/kitti_eval.py`.

## 정확도 — holdout 500장 Moderate AP40

| | ONNX FP32 | Hailo fp | **Hailo INT8** | 양자화 손실 | 예선 ORT INT8 |
|---|---|---|---|---|---|
| A (1280x384, P5 있음) | 97.22 | 97.22 | **96.50** | −0.72 | 96.76 |
| B (1280x384, P5 제거) | 97.17 | 97.17 | **96.03** | −1.14 | 96.57 |
| C (640x192, P5 있음)  | 92.41 | 92.41 | **90.54** | −1.87 | 90.25 |
| D (640x192, P5 제거)  | 91.67 | 91.67 | **90.44** | −1.23 | 90.27 |

Hailo INT8 클래스별 Moderate (괄호: fp 대비)

| | Car | Pedestrian | Cyclist |
|---|---|---|---|
| A | 98.36 (−0.47) | 92.14 (−1.43) | 98.99 (−0.26) |
| B | 97.89 (−0.22) | 92.48 (−2.25) | 97.72 (−0.95) |
| C | 95.52 (−1.86) | 88.10 (−1.15) | 87.99 (−2.61) |
| D | 95.34 (+0.05) | 88.00 (−0.97) | 87.97 (−2.79) |

- Hailo fp 가 네 모델 모두 ONNX FP32 와 소수점까지 일치 → 전처리·디코드·채점 경로 검증됨.
- 640x192 에서 C 와 D 의 Hailo INT8 차이는 0.10 (C 가 높음) — 정확도로는 둘을 가를 수 없다.
- B 는 Hailo INT8 에서 A 보다 0.47 낮다 (예선 ORT INT8 에서는 0.19). DECISION.md 의 0.5 경계 안쪽이지만
  context 수가 A 와 같아(아래) "느리지 않다" 조건을 아직 확인할 수 없다. 보드 FPS 로 결정.

## 컴파일

| | HEF | context 수 | 예상 FPS |
|---|---|---|---|
| A | 5.6MB | 5 | N/A |
| B | 4.9MB | 5 | N/A |
| C | 4.7MB | 4 | N/A |
| D | 4.1MB | 4 | N/A |
| 참고: Model Zoo yolov11n (COCO 640x640, 6.55 GOPs) | 5.3MB | **3** | **307.5 FPS / 3.25ms (보드 실측)** |

- 세 모델 모두 single context 실패(LCU 초과) → multi-context. multi-context 는 profiler 가
  FPS/latency 를 추정하지 않는다(N/A). 실측: `hailortcli run hef/<모델>.hef`.
- Model Zoo 참고값 출처: `ModelZoo/Compiled/v5.4.0/hailo10h/yolov11n_profiler_results_compiled_runtime_data.html`
  (`performance_details`: batch 1, fps 307.53, latency 3.25ms, number_of_contexts 3).
  즉 Hailo-10H 에서 multi-context 자체는 정상.

## context 수 진단 (보드 없이, `05_compile_variant.py`)

`context_switch_param(mode=disabled)` 로 compile 하면 single context 에 못 들어가는 이유가 로그에 나온다.
네 모델과 대조군 모두 **LCU 하나만** 초과한다 (다른 자원 언급 없음). 로그 요약은 `results/diag/`.

| 모델 | LCU 수요 / 한도 | context 수 |
|---|---|---|
| Model Zoo yolov11n 640x640 — **우리 파이프라인(03)** 으로 compile | 142 / 80 | **4** |
| Model Zoo yolov11n — 공식 빌드 | – | 3 |
| A (1280x384, P5 있음) | 142 / 80 | 5 |
| C (640x192, P5 있음)  | 142 / 80 | 4 |
| B (1280x384, P5 제거) | 118 / 80 | 5 |
| D (640x192, P5 제거)  | 118 / 80 | 4 |

- **LCU 수요는 해상도가 아니라 구조로 정해진다.** P5 있으면 142, 없으면 118, 해상도와 무관.
  공식 yolov11n 도 142. 그래서 1.72 GFLOPs 인 D 도 칩 한도(80)를 넘는다 — D 가 이상한 게 아니다.
- **P5 제거는 context 수를 줄이지 못했다** (1280: 5→5, 640: 4→4). B·D 의 Hailo 속도 이점은 보드 실측 전엔 근거 없음.
- **같은 yolov11n 인데 공식 빌드 3 vs 우리 4.** 차이는 빌드 설정에 있다.
  `allocator_param(width_splitter_defuse=disabled)` (Model Zoo `.alls` 에 있는 줄) 는 D·yolov11n 모두
  context 수를 바꾸지 못함 → 원인 아님. 남은 후보(미확인): 공식 빌드의 양자화 레벨,
  `change_output_activation(sigmoid)`, `nms_postprocess`, hailomz 의 compile 기본값.
- profiler 의 "Model Parameters 3.90M"(A) 은 기존 2.58M 과 다르다. 원인 미확인 — 보고서에는 기존 값 사용.

## 파일

- `hef/*.hef`, `hef/HEF.sha256` — 컴파일 결과 (`*.hef` 는 .gitignore 대상이라 강제 추가)
- `results/<모델>/` — `ap40_*_holdout_{quantized,fp}.json`, `model_script.alls`, `compile_summary.txt`,
  `profile_compiled_summary.txt`, `calib_filelist.txt`, `emu_results.jsonl`
- `end_nodes_p34.txt` — P5 제거 모델(B·D)의 parse 절단 노드 4개
- `04_emulate_eval.py` — 에뮬레이터 채점 스크립트
- `05_compile_variant.py` — optimized HAR 에 compile 단계 `.alls` 줄만 바꿔 다시 compile (양자화 재실행 없음)
- `results/diag/` — 위 진단 로그 요약
- `06_board_e2e.py` — 보드 end-to-end 구간별 지연(읽기/전처리/NPU/디코드/NMS) + 예측 덤프.
  `--backend emulator` 로 PC 검증: D 16장 덤프가 04 채점 예측과 바이트 단위 일치.
  `--backend hailort` 경로(hailo_platform InferModel API)는 보드에서 처음 돈다 — 미검증

## 알려진 함정 (이번에 겪은 것)

- 저장소가 `core.autocrlf=true` 로 체크아웃되어 `.sh`/`end_nodes*.txt` 가 CRLF → WSL 에서
  `$'\r': command not found`. WSL 로 복사한 뒤 `sed -i 's/\r$//'` 필요.
- `03_hailo_compile.sh` 의 profile 이 optimized HAR 를 보고 있어 context 정보가 빠졌다 → compiled HAR 로 수정.
- 양자화된 HAR 에는 `load_model_script(..., append=True)` 가 안 된다 → 비-append 로 compile 명령만 로드.
- COCO 80클래스 yolov11n 은 parse 중 "nms postprocess 추가?(y/n)" 를 물어 비대화식 실행이 멈춘다 → `yes n |`.
