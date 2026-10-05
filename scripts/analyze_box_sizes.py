#!/usr/bin/env python3
"""KITTI GT 박스 크기 분포 분석 -> 탐지 헤드(P3/P4/P5) 유휴도 측정.

왜 이걸 먼저 돌리는가:
  YOLO 계열은 세 해상도의 탐지 헤드를 가진다. stride 8(P3)은 작은 객체,
  16(P4)은 중간, 32(P5)은 큰 객체를 담당한다. 데이터셋에 큰 객체가
  거의 없으면 P5 헤드는 파라미터와 연산만 먹고 거의 일을 안 한다.
  그걸 "거의 없다"고 추측하는 대신 **본인 train split 에서 직접 세는** 것이
  이 스크립트다. 아키텍처를 줄이는 결정의 근거가 여기서 나온다.

중요: 반드시 **train split 만** 넘긴다. 공식 eval 1,000장의 분포를 보고
아키텍처를 고르면 그 셋에 맞춘 설계가 되고, 사실상 평가셋 오버핏이다.

크기 -> 헤드 배정 기준:
  모델 입력 좌표계(letterbox 적용 후)에서 sqrt(bbox 면적)을 기준으로,
  FCOS/FPN 계열의 통상적인 경계인 64 / 128 px 를 쓴다.
    < 64      -> P3 (stride 8)
    64 ~ 128  -> P4 (stride 16)
    > 128     -> P5 (stride 32)
  경계값은 --bounds 로 바꿀 수 있다. 절대적인 규칙이 아니라 관례이므로
  보고서에는 경계값을 명시하고, 민감도를 보려면 몇 개 값으로 돌려볼 것.

사용 (results/box_sizes.md 의 수치를 재현하는 명령):
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

--image-dir 를 주면 이미지마다 실제 해상도를 읽어 letterbox 배율을 개별 계산한다
(KITTI 는 1242x375 / 1224x370 / 1238x374 / 1241x376 등이 섞여 있다).
"""
import argparse
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

CLASSES = ["Car", "Pedestrian", "Cyclist"]
DIFFICULTIES = ["easy", "moderate", "hard"]
MIN_HEIGHT = {"easy": 40, "moderate": 25, "hard": 25}
MAX_OCCLUSION = {"easy": 0, "moderate": 1, "hard": 2}
MAX_TRUNCATION = {"easy": 0.15, "moderate": 0.30, "hard": 0.50}


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(26)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"PNG 아님: {path}")
    return struct.unpack(">II", head[16:24])


def parse_label(line):
    p = line.split()
    if len(p) < 15:
        return None
    return {
        "type": p[0],
        "truncated": float(p[1]),
        "occluded": int(float(p[2])),
        "box": [float(p[4]), float(p[5]), float(p[6]), float(p[7])],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="KITTI GT 박스 크기 분포 -> 헤드 유휴도")
    ap.add_argument("--label-dir", required=True, help="KITTI 원본 label_2 디렉터리")
    ap.add_argument("--split", required=True, nargs="+",
                    help="분석할 id 목록 파일(여러 개 가능). "
                         "**eval_val.txt 는 절대 넘기지 말 것** — 평가셋 분포에 맞춰 "
                         "아키텍처를 고르면 사실상 평가셋 오버핏이다. "
                         "재현: --split splits/train.txt splits/holdout.txt")
    ap.add_argument("--image-dir", default=None,
                    help="이미지 폴더. 주면 파일마다 실제 해상도를 읽는다 "
                         "(없으면 --orig 값으로 가정)")
    ap.add_argument("--orig", nargs=2, type=int, default=[375, 1242], metavar=("H", "W"),
                    help="원본 해상도 가정값 (기본 375 1242)")
    ap.add_argument("--imgsz", nargs=2, type=int, default=[384, 1280], metavar=("H", "W"),
                    help="모델 입력 해상도 (기본 384 1280)")
    ap.add_argument("--bounds", nargs=2, type=float, default=[64, 128],
                    metavar=("P3P4", "P4P5"), help="헤드 경계 (기본 64 128)")
    ap.add_argument("--difficulty", default="moderate",
                    choices=DIFFICULTIES + ["all"],
                    help="어느 난이도 GT 를 셀지. 기본 moderate (공지 지정 지표)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    for sp in args.split:
        if "eval_val" in Path(sp).stem.lower():
            raise SystemExit(
                f"[중단] 공식 평가셋으로 보이는 파일이 들어왔습니다: {sp}\n"
                f"  아키텍처 결정은 train/holdout 으로만 하세요.")

    H_in, W_in = args.imgsz
    b1, b2 = args.bounds
    lbl_dir = Path(args.label_dir)

    ids, seen = [], set()
    for sp in args.split:
        for s in Path(sp).read_text().splitlines():
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                ids.append(s)
    if not ids:
        raise SystemExit(f"id 가 없습니다: {args.split}")
    print(f"split 파일 {len(args.split)}개 -> 중복 제거 후 id {len(ids):,}개")

    diffs = DIFFICULTIES if args.difficulty == "all" else [args.difficulty]
    img_dir = Path(args.image_dir) if args.image_dir else None

    # [difficulty][class][level] = count
    cnt = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    heights = defaultdict(lambda: defaultdict(list))
    n_img = 0
    scales = []

    for i in ids:
        f = lbl_dir / f"{i}.txt"
        if not f.exists():
            raise SystemExit(f"라벨 없음: {f}")
        n_img += 1

        if img_dir:
            cand = next((img_dir / f"{i}{e}" for e in (".png", ".jpg")
                         if (img_dir / f"{i}{e}").exists()), None)
            if cand is None:
                raise SystemExit(f"이미지 없음: {i}")
            W_o, H_o = png_size(cand) if cand.suffix == ".png" else (args.orig[1], args.orig[0])
        else:
            H_o, W_o = args.orig

        # letterbox 배율 (비율 유지)
        r = min(H_in / H_o, W_in / W_o)
        scales.append(r)

        for line in f.read_text().splitlines():
            o = parse_label(line.strip())
            if o is None or o["type"] not in CLASSES:
                continue
            x1, y1, x2, y2 = o["box"]
            h_o, w_o = y2 - y1, x2 - x1
            if h_o <= 0 or w_o <= 0:
                continue

            for d in diffs:
                if (o["occluded"] > MAX_OCCLUSION[d]
                        or o["truncated"] > MAX_TRUNCATION[d]
                        or h_o <= MIN_HEIGHT[d]):
                    continue
                # 모델 입력 좌표계로 변환
                size = ((h_o * r) * (w_o * r)) ** 0.5
                level = "P3" if size < b1 else ("P4" if size < b2 else "P5")
                cnt[d][o["type"]][level] += 1
                cnt[d]["(전체)"][level] += 1
                heights[d][o["type"]].append(h_o * r)
                heights[d]["(전체)"].append(h_o * r)

    avg_scale = sum(scales) / len(scales)
    print(f"분석 대상 {n_img:,}장  |  원본 -> {W_in}x{H_in} 평균 배율 {avg_scale:.3f}")
    print(f"헤드 경계: P3 < {b1:g} <= P4 < {b2:g} <= P5   (모델 입력 좌표계, sqrt(면적))")

    out = {"n_images": n_img, "imgsz": [H_in, W_in], "bounds": [b1, b2],
           "avg_letterbox_scale": round(avg_scale, 4), "result": {}}

    for d in diffs:
        print(f"\n=== {d} GT 기준 ===")
        print(f"  {'클래스':<12}{'P3':>9}{'P4':>9}{'P5':>9}{'합계':>9}"
              f"{'P5 비율':>10}{'중위 높이':>11}")
        rows = {}
        for c in CLASSES + ["(전체)"]:
            lv = cnt[d][c]
            tot = sum(lv.values())
            if tot == 0:
                print(f"  {c:<12}{'-':>9}{'-':>9}{'-':>9}{0:>9}")
                rows[c] = {"P3": 0, "P4": 0, "P5": 0, "total": 0}
                continue
            hs = sorted(heights[d][c])
            med = hs[len(hs) // 2]
            p5r = lv['P5'] / tot * 100
            mark = "  <<<" if c == "(전체)" else ""
            print(f"  {c:<12}{lv['P3']:>9,}{lv['P4']:>9,}{lv['P5']:>9,}{tot:>9,}"
                  f"{p5r:>9.2f}%{med:>10.1f}px{mark}")
            rows[c] = {"P3": lv["P3"], "P4": lv["P4"], "P5": lv["P5"],
                       "total": tot, "P5_pct": round(p5r, 3),
                       "median_height_px": round(med, 1)}
        out["result"][d] = rows

        t = rows["(전체)"]
        if t["total"]:
            p5 = t["P5_pct"]
            print()
            if p5 < 1.0:
                print(f"  판정: P5 담당 객체가 전체의 {p5:.2f}% — 사실상 유휴.")
                print(f"        P5 헤드 제거가 정당화된다. 파라미터·연산 절감 효과를")
                print(f"        실측해서 정확도 손실과 비교할 가치가 있다.")
            elif p5 < 5.0:
                print(f"  판정: P5 담당 객체가 {p5:.2f}% — 적지만 0은 아니다.")
                print(f"        제거 시 그 {p5:.2f}%에서 손실이 나므로, 클래스별로")
                print(f"        어디서 빠지는지 확인하고 결정할 것.")
            else:
                print(f"  판정: P5 담당 객체가 {p5:.2f}% — 무시할 수준이 아니다.")
                print(f"        제거보다 채널 축소(프루닝) 쪽이 안전하다.")

    # 해상도를 바꾸면 분포가 어떻게 움직이는지
    print(f"\n참고: 같은 데이터를 다른 해상도로 보면 경계가 이동합니다.")
    print(f"      --imgsz 192 640 으로 다시 돌려 비교해보세요 "
          f"(배율이 절반이면 모든 객체가 한 단계 작은 쪽으로 밀립니다).")

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\n기록: {p}  ← 보고서 근거 자료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
