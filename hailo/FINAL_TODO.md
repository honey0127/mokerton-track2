# 본선 남은 작업 안내 (2026-11-21) — 보드 담당 / 공용 PC 담당

규칙은 저장소 루트 `DECISION_FINAL.md`. 측정을 시작하기 전에 한 번 읽어주세요 (eval_val 은 선택에 쓰지 않음).

## A. 보드 (Pi 5 + AI HAT+ 2) — 11/23~25

새로 생긴 파일
- `hailo/07_board_fast.py` : 비동기 처리량·지연·CPU·메모리 측정 (06 의 후속)
- `hailo/08_demo.py` : 라이브 시연 (카메라 / 영상 / KITTI 이미지)
- `hailo/hailo_rt.py` : 둘이 같이 쓰는 모듈 (빠른 디코드, HailoRT 비동기 backend, 자원 측정)
- `hailo/run_board_final.sh` : 아래 측정을 한 번에

PC 에서 확인한 것: 빠른 디코드가 06 의 디코드와 4개 모델 x conf 3개(0.001/0.25/0.9)에서 같은 검출을 낸다
(좌표 차이 3e-4 px 이하, 디코드 시간 1280x384 기준 약 10ms -> 0.6~1.6ms). 비동기 파이프라인·덤프·시연 녹화도
ONNX 로 돌려봤다. **HailoRT 비동기 API(`run_async`) 부분은 보드에서 처음 돈다.**

```bash
cd mokerton-track2 && git pull
export IMG=<holdout 이미지 폴더>        # 06 때 쓴 폴더
export GT=<KITTI training/label_2 폴더>  # ap 모드에만 필요

bash hailo/run_board_final.sh check     # 2분. 마지막 줄 '결과: 일치' 확인
bash hailo/run_board_final.sh sweep     # 15~20분
bash hailo/run_board_final.sh ap        # 10~15분
bash hailo/run_board_final.sh demo      # 1분
```

에러가 나면 그 화면 그대로(특히 Traceback 마지막 10줄) 보내주세요. 예상되는 곳:
- `wait_for_async_ready` / `run_async` / `get_async_queue_size` 인자 이름 (HailoRT 버전마다 조금 다름)
- `--out-format raw` 의 `quant_info` 필드 이름

끝나면 `hailo/results/board/final/` 폴더를 커밋·푸시 (`runs/` 의 예측 덤프는 올리지 않음).

**확인할 것 — 보드 holdout 값이 에뮬레이터와 다른 이유**: 11/19 의 `ap_board.txt` 를 만들 때 conf 를 얼마로
덤프했는지 (06 의 기본값은 0.25, 채점용은 0.001). `ap` 모드 결과(conf 0.001)가 에뮬레이터 값(A 96.50, B 96.03,
C 90.54, D 90.44)과 같으면 원인은 conf 였던 것.

### 시연 (카메라를 받으면)

```bash
# 모니터 연결한 보드에서. m 키로 B(정확도) <-> D(속도) 전환, q 종료
python3 hailo/08_demo.py --models hailo/hef/B_1280x384.hef,hailo/hef/D_640x192.hef --source picam --crop kitti --fullscreen
# 카메라 없을 때
python3 hailo/08_demo.py --models hailo/hef/B_1280x384.hef,hailo/hef/D_640x192.hef --source kitti \
    --images $IMG --split splits/holdout.txt --rate 10 --fullscreen
```
`--crop kitti` 는 카메라 화면 가운데 띠를 KITTI 비율(1242:375)로 잘라서 넣는다 (위아래 회색 여백 낭비 없음).
`--crop-y 0.6` 처럼 띠 위치를 내릴 수 있다. 발표 백업용으로 `--save demo.mp4` 로 30초 녹화해두기.

## B. 공용 PC (DFC 5.4.0) — 11/24~26, 정확도 개선 재양자화

지금 HEF 는 `optimization_level=0` (가장 가벼운 설정, 보정 이미지 실제 64장 사용). Hailo INT8 손실이
holdout 에서 A −0.72, B −1.14, C −1.87, D −1.23. 같은 보정 목록(256장)으로 아래 변형을 만든다.

| 이름 | `model_script.alls` 에서 바꿀 줄 | 기대 |
|---|---|---|
| Q1 | `model_optimization_flavor(optimization_level=1, compression_level=0)` + `model_optimization_config(calibration, calibset_size=256)` | equalization. CPU 로도 수십 분 |
| Q2 | Q1 + 박스 출력층(cv2.*.2 에 해당하는 conv) 16비트: `quantization_param([<층 이름들>], precision_mode=a16_w16)` | 예선 분석에서 640x192 Car 손실 원인이 박스 경로였다 |
| Q3 | `optimization_level=2` (CPU 에서 너무 오래 걸리면 생략) | finetune 포함 |

- 층 이름은 `hailo_out/<모델>/` 의 HAR 에서 확인 (`hailo har info` 또는 profile.html 의 출력층 목록).
  16비트 문법은 DFC 5.4 문서의 `quantization_param` / `precision_mode` 항목으로 확인.
- 모델당 순서: `04_emulate_eval.py --context quantized` 로 holdout 채점 → 현재보다 +0.30 이상이면 compile 해서
  HEF 를 `hailo/hef/<모델>_<Q이름>.hef` 로 올리기. 보드에서 07 로 FPS·holdout 을 다시 재서 규칙(DECISION_FINAL.md 2번)으로 판정.
- 우선순위: **D, B 먼저** (예선 주력), 시간 되면 A, C.

## C. 일정

| 날짜 | 보드 | 공용 PC | 정리 (경민) |
|---|---|---|---|
| 11/22 | — | — | 규칙 커밋, 코드 올리기 |
| 11/23~25 | check → sweep → ap → demo | Q1·Q2 (D, B 먼저) | 결과 받아 정리 |
| 11/26 | 새 HEF 측정 | Q3 / A·C | 보고서 초안 |
| 11/27 | — | — | 최종 결정 커밋 |
| 11/28 | 최종 HEF 로 시연 녹화 | — | Release·태그 |
| 11/30 | — | — | **보고서 PDF 메일 제출 23:59** |
| 12/1~3 | 리허설 | — | 발표 자료 |
