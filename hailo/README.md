# Hailo DFC 파이프라인 (본선 대비)

예선은 ONNX/ORT 로 끝나지만, 본선은 Raspberry Pi 5 + **Hailo-10H NPU** 다.
`.onnx` → `.hef` 가 안 나오면 본선에서 아무것도 못 한다. 그래서 예선 모델이
확정되기 전에 **컴파일이 통과하는지만** 먼저 뚫어둔다.

`scripts/quantize_ort.py` 와는 완전히 다른 툴체인이다. 혼동 주의:

| | ONNX Runtime PTQ | Hailo DFC |
|---|---|---|
| 입력 | FP32 ONNX | **FP32 ONNX** (양자화 전!) |
| 양자화 주체 | ORT | **DFC 내부** |
| 산출물 | INT8 `.onnx` | `.har` → `.hef` |
| 용도 | 예선 데스크톱 지연 측정 | 본선 NPU 이식 |

> DFC 에는 **양자화되지 않은 부동소수점 모델**을 넣는다. INT8 ONNX 를 넣으면 안 된다.
> `quantize_ort.py` 결과물은 DFC 입력이 아니다.

## 환경

- **x86_64 Linux 전용.** Apple Silicon(ARM) 지원 계획 없음 — Hailo 공식 답변.
  맥 쓰는 팀원은 Colab(Ubuntu x86_64) 또는 x86 머신을 써야 한다.
- Hailo Developer Zone 가입 + EULA 동의만 하면 무료. 별도 라이선스 없음.
- **Hailo-10H 는 DFC v5.x 계열.** v3.x 는 Hailo-8 용이고, Model Zoo 2.x 는
  hailo10h 지원이 빠졌다. 버전 잘못 받으면 `--hw-arch hailo10h` 가 없다.
- venv 를 학습 환경과 **절대 합치지 않는다** (numpy/protobuf 충돌).

## 순서

```bash
# 1) Windows / 학습 env 에서: FP32 ONNX export + Hailo end node 자동 탐색
python hailo/01_export_onnx.py --weights runs/<실험>/weights/best.pt --imgsz 384 1280

# 2) 캘리브레이션 세트 (train 또는 holdout 에서만!)
python hailo/02_make_calib.py --image-dir <train 이미지 폴더> \
    --eval-list splits/eval_val.txt --out calib_256.npy -n 256

# 3) Linux(WSL) / DFC env 에서: parse → optimize → compile → profile
bash hailo/03_hailo_compile.sh onnx/<모델>.onnx '<01 이 출력한 end node 목록>' calib_256.npy hailo10h
```

## 왜 end node 를 잘라야 하나 (01 번 스크립트의 존재 이유)

Ultralytics 가 `nms=False` 로 뽑아도 ONNX 안에는 DFL softmax + 앵커 연산 +
Concat 이 그대로 남는다. 이걸 DFC 에 넣으면 미지원 연산이거나 정확도가 깨진다.
그래서 `hailo parser --end-node-names` 로 **Detect 디코드 직전의 raw Conv** 에서
그래프를 끊어야 한다.

문제는 그 노드 이름이 모델마다 다르다는 것. 특히 **P5 를 제거하면 Detect 가
3-scale → 2-scale 이 되어 end node 가 6개 → 4개로 바뀐다.** Model Zoo 의
yolov8s 설정을 그대로 못 쓴다. `01_export_onnx.py` 가 이걸 자동으로 찾아
`.endnodes.json` 으로 남긴다.

`scripts/head_nodes.py` 와 목적이 다르다 — 그쪽은 **ORT PTQ 제외 목록**(FP32 로
남길 노드), 이쪽은 **Hailo parse 절단 지점**이다. 둘 다 필요하다.

## 캘리브레이션 주의

- 전처리가 추론 전처리와 어긋나면 양자화 스케일이 틀어져 정확도가 이유 없이 떨어진다.
  `02_make_calib.py` 는 Ultralytics 와 동일한 letterbox(비율 유지, 중앙 패딩 114)를 쓴다.
- 출력은 **uint8 NHWC `(N, H, W, 3)`**, 0~255 그대로. 정규화는 `.alls` 의
  `normalization` 레이어가 NPU 안에서 처리한다.
- **공식 eval 1,000장은 캘리브레이션에도 쓰면 안 된다.** 라벨을 안 써도 분포 유출이다.
  `--eval-list splits/eval_val.txt` 를 주면 섞여 있을 때 즉시 중단한다.

## 알려진 함정

**RTX 50 시리즈(Blackwell) + DFC** — DFC 가 import 시 TensorFlow 로 GPU 테스트를
하는데 `CUDA_ERROR_INVALID_HANDLE` 로 죽는다. `03_hailo_compile.sh` 가
`export CUDA_VISIBLE_DEVICES=''` 로 CPU 강제해서 우회한다.

**WSL2 에서는 DFC 가 GPU 를 못 쓴다** (공식 문서 명시). 그래서 `.alls` 1차 시도는
`optimization_level=0` 으로 간다. `.hef` 가 일단 나오는 것을 확인한 뒤에 올릴 것.
optimization_level 3~4 / Adaround / QFT 는 GPU 없이는 매우 느리다.

**compile 에서 리소스 초과** — 로그에 `resources` / `utilization` 경고가 뜨면
해상도를 640×192 로 낮추거나 모델을 더 줄여야 한다. 이게 뜬다는 건 역설적으로
"경량화가 본선 통과 조건"이라는 증거이므로 보고서에 쓸 소재다.

## 보고서에 들어갈 것

- `.hef` 크기
- `hailo profiler` 의 예상 FPS / latency / 리소스 사용률
- 로그의 레이어별 양자화 방식 (어디가 INT8, 어디가 INT4/FP16)
- 참고 목표치: Model Zoo 기준 YOLO11n / Hailo-10H = **303 FPS**
