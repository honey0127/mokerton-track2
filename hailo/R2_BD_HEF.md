# B·D HEF 컴파일 (2026-10-15 작성) — 학과 공용 PC 에서 실행

목표: 주력 B(1280x384)·D(640x192)의 FP32 ONNX 를 Hailo-10H HEF 로 만들고 profiler 수치를 받는다.
보고서 3.3절을 '검토'에서 '실증'으로 바꾸는 근거다. 1차 목표는 **HEF 가 나오는 것**, 정확도 튜닝은 그다음이다.

## 0. 준비 (한 번)

- WSL2 Ubuntu 또는 네이티브 Linux x86_64, DFC 5.x (hailo10h) 가 설치된 venv/conda (`hailo/README.md` 환경 절)
- 저장소 최신 코드 + `models/` 의 FP32 ONNX 2개 (해시: `MODELS.sha256`)
- train 이미지 (`data/kitti_yolo/images/train`, eval_val 이 섞이면 02 스크립트가 멈춘다)

## 1. end node 확인 (export 생략 — 보고서와 같은 ONNX 를 그대로 쓴다)

```bash
python hailo/01_export_onnx.py --weights x --onnx models/best-p34_1280x384_op13_fp32.onnx
python hailo/01_export_onnx.py --weights x --onnx models/best-p34_640x192_op13_fp32.onnx
# 기대: end node 4개 (/model.20/cv2.0/cv2.0.2/Conv, cv3.0.2, cv2.1.2, cv3.1.2)
```

## 2. 보정 세트 (train 에서 256장)

```bash
python hailo/02_make_calib.py --image-dir data/kitti_yolo/images/train --eval-list splits/eval_val.txt \
    --imgsz 384 1280 -n 256 --out calib_B_256.npy
python hailo/02_make_calib.py --image-dir data/kitti_yolo/images/train --eval-list splits/eval_val.txt \
    --imgsz 192 640 -n 256 --out calib_D_256.npy
```

## 3. 컴파일 (D 먼저 — 작아서 빨리 끝나고 실패 원인 찾기 쉽다)

```bash
bash hailo/03_hailo_compile.sh models/best-p34_640x192_op13_fp32.onnx \
  '/model.20/cv2.0/cv2.0.2/Conv,/model.20/cv3.0/cv3.0.2/Conv,/model.20/cv2.1/cv2.1.2/Conv,/model.20/cv3.1/cv3.1.2/Conv' \
  calib_D_256.npy hailo10h
bash hailo/03_hailo_compile.sh models/best-p34_1280x384_op13_fp32.onnx \
  '/model.20/cv2.0/cv2.0.2/Conv,/model.20/cv3.0/cv3.0.2/Conv,/model.20/cv2.1/cv2.1.2/Conv,/model.20/cv3.1/cv3.1.2/Conv' \
  calib_B_256.npy hailo10h
```
end node 이름은 1단계 출력을 그대로 복사한다 (위는 예상값).

## 4. 돌려줄 것

`hailo_out/<모델명>/` 폴더의 다음 파일:
- `pipeline.log` (전체 로그 — 실패해도 이것만 있으면 원인을 찾을 수 있다)
- `*.hef` 크기, `profile.html`, `model_script.alls`

## 실패 시 흔한 원인

| 증상 | 조치 |
|---|---|
| parse 에서 Unsupported (C2PSA 의 MatMul/Softmax 등) | 로그 줄 그대로 보내기. COCO YOLO11n 640 은 통과했으므로 이름·opset 문제일 가능성이 크다 |
| optimize 에서 메모리 부족 | `.alls` 에 `model_optimization_config(calibration, batch_size=4)` 추가 |
| compile 에서 resources/utilization | 1280 은 context 가 늘 수 있다. D(640)만이라도 먼저 확보 |
| TF GPU 오류 | `CUDA_VISIBLE_DEVICES=''` 가 비어 있는지 (스크립트가 설정) |

## 다음 (HEF 가 나오면)

- optimization_level 2 로 다시 (GPU 가능 환경), 박스 출력층(cv2.*.2) 16비트 유지 시험 — 1차 분석에서 640x192 의 Car 손실 원인이 박스 경로였다
- RPi5 + Hailo-10H 실기기: HailoRT 추론 + CPU 후처리(DFL·디코드·NMS) → eval_val 재채점
