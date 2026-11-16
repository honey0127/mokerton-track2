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
| D (640x192, P5 제거)  | 91.67 | 91.67 | **90.44** | −1.23 | 90.27 |

Hailo INT8 클래스별 Moderate (괄호: fp 대비)

| | Car | Pedestrian | Cyclist |
|---|---|---|---|
| A | 98.36 (−0.47) | 92.14 (−1.43) | 98.99 (−0.26) |
| B | 97.89 (−0.22) | 92.48 (−2.25) | 97.72 (−0.95) |
| D | 95.34 (+0.05) | 88.00 (−0.97) | 87.97 (−2.79) |

- Hailo fp 가 세 모델 모두 ONNX FP32 와 소수점까지 일치 → 전처리·디코드·채점 경로 검증됨.
- B 는 Hailo INT8 에서 A 보다 0.47 낮다 (예선 ORT INT8 에서는 0.19). DECISION.md 의 0.5 경계 안쪽이지만
  context 수가 A 와 같아(아래) "느리지 않다" 조건을 아직 확인할 수 없다. 보드 FPS 로 결정.

## 컴파일

| | HEF | context 수 | 예상 FPS |
|---|---|---|---|
| A | 5.6MB | 5 | N/A |
| B | 4.9MB | 5 | N/A |
| D | 4.1MB | 4 | N/A |
| 참고: Model Zoo yolov11n (COCO 640x640, 6.55 GOPs) | 5.3MB | **3** | **307.5 FPS / 3.25ms (보드 실측)** |

- 세 모델 모두 single context 실패(LCU 초과) → multi-context. multi-context 는 profiler 가
  FPS/latency 를 추정하지 않는다(N/A). 실측: `hailortcli run hef/<모델>.hef`.
- Model Zoo 참고값 출처: `ModelZoo/Compiled/v5.4.0/hailo10h/yolov11n_profiler_results_compiled_runtime_data.html`
  (`performance_details`: batch 1, fps 307.53, latency 3.25ms, number_of_contexts 3).
  즉 Hailo-10H 에서 multi-context 자체는 정상. 다만 D(1.72 GFLOPs)가 4 context 인 것은 설명이 안 된다 → 진단 중.
- profiler 의 "Model Parameters 3.90M"(A) 은 기존 2.58M 과 다르다. 원인 미확인 — 보고서에는 기존 값 사용.

## 파일

- `hef/*.hef`, `hef/HEF.sha256` — 컴파일 결과 (`*.hef` 는 .gitignore 대상이라 강제 추가)
- `results/<모델>/` — `ap40_*_holdout_{quantized,fp}.json`, `model_script.alls`, `compile_summary.txt`,
  `profile_compiled_summary.txt`, `calib_filelist.txt`, `emu_results.jsonl`
- `end_nodes_p34.txt` — P5 제거 모델(B·D)의 parse 절단 노드 4개
- `04_emulate_eval.py` — 에뮬레이터 채점 스크립트

## 알려진 함정 (이번에 겪은 것)

- 저장소가 `core.autocrlf=true` 로 체크아웃되어 `.sh`/`end_nodes*.txt` 가 CRLF → WSL 에서
  `$'\r': command not found`. WSL 로 복사한 뒤 `sed -i 's/\r$//'` 필요.
- `03_hailo_compile.sh` 의 profile 이 optimized HAR 를 보고 있어 context 정보가 빠졌다 → compiled HAR 로 수정.
