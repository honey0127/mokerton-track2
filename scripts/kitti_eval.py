"""KITTI 2D Object Detection — Moderate AP40 평가 (공식 판정 엔진).

판정 엔진: third_party/kitti_object_eval_python
  OpenPCDet 에 포함된 공식 평가 포트(traveller59, numba). 공식 C++ devkit(evaluate_object.cpp)을
  줄 단위로 옮긴 코드다. 2026-10-06 합성 데이터 5세트(각 1000장)에서 공식 C++ 과
  9칸(3클래스 x 3난이도) 모두 소수점 둘째 자리까지 일치했다.
  이전 자체 구현(예측 우선 greedy 매칭)은 같은 데이터에서 칸별 최대 1.5점 차이가 나서 교체했다.

공식 규칙 요약 (보고서 '평가 방법' 절에 그대로 쓸 수 있다):
  1. 난이도 — Moderate: 높이>=25px, occlusion<=1, truncation<=0.30
  2. 이웃 클래스 — Car 판정의 Van, Pedestrian 판정의 Person_sitting 은 FN 도 FP 도 아니다
  3. 난이도 미달 GT 는 FN 이 아니고, 거기 맞은 예측도 FP 가 아니다
  4. 높이 25px 미만 예측은 FP 로 세지 않는다
  5. DontCare — '예측 박스 면적 대비 겹침'이 클래스 IoU 임계(Car 0.7 / Ped·Cyc 0.5)를 넘으면 버린다
  6. 매칭 — GT 마다, 임계 IoU 를 넘는 예측 중 '겹침이 가장 큰 것'을 고른다 (점수 순이 아니다)
  7. AP40 — TP 점수로 임계값 41개를 고르고, 임계값마다 TP/FP 를 다시 세어 1~40번째 정밀도 평균

  python scripts/kitti_eval.py --gt data/kitti_raw/training/label_2 \
      --pred runs/pred_best_1280x384_op13_fp32 --split splits/eval_val.txt --tag best_1280x384_op13_fp32
  python scripts/kitti_eval.py --selftest
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "third_party"))
from common import CLASSES, jsonl_append  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore", module="numba")
from kitti_object_eval_python import eval as E            # noqa: E402
from kitti_object_eval_python import kitti_common as KC   # noqa: E402

DIFFS = ("Easy", "Moderate", "Hard")
MIN_OVERLAP = np.array([[[0.7, 0.5, 0.5]] * 3])   # [1, metric(bbox/bev/3d), class] — 2D 는 metric 0만 쓴다


# ---------------------------------------------------------------- 입출력
def load_annos(folder, ids, what):
    folder = Path(folder)
    missing = [i for i in ids if not (folder / f"{i}.txt").exists()]
    if missing:
        sys.exit(f"[중단] {what} 파일 {len(missing)}개 없음 (예: {missing[:3]}) — {folder}\n"
                 f"예측이 0개인 이미지도 빈 파일이 있어야 한다.")
    return KC.get_label_annos(str(folder), [int(i) for i in ids])


def count_valid_gt(gt_annos, dt_annos, ci, di):
    return int(E._prepare_data(gt_annos, dt_annos, ci, di)[-1])


# ---------------------------------------------------------------- 평가
def evaluate(gt_dir, pred_dir, ids):
    gt = load_annos(gt_dir, ids, "GT")
    dt = load_annos(pred_dir, ids, "예측")
    ret = E.eval_class(gt, dt, [0, 1, 2], [0, 1, 2], 0, MIN_OVERLAP)
    ap = E.get_mAP_R40(ret["precision"])[:, :, 0]          # [class, difficulty]
    res = {}
    for ci, c in enumerate(CLASSES):
        for di, d in enumerate(DIFFS):
            n = count_valid_gt(gt, dt, ci, di)
            res[f"{c}/{d}"] = {"AP40": round(float(ap[ci, di]), 2) if n else float("nan"), "n_gt": n}
    missing = [c for c in CLASSES if res[f"{c}/Moderate"]["n_gt"] == 0]
    if missing:
        res["warning"] = (f"GT가 0인 클래스 {missing} 는 평균에서 제외했다. "
                          f"공식 1000장 평가셋에서는 3클래스 모두 있어야 정상이다.")
        print("[경고] " + res["warning"], file=sys.stderr)
    for d in DIFFS:
        vals = [res[f"{c}/{d}"]["AP40"] for c in CLASSES if not np.isnan(res[f"{c}/{d}"]["AP40"])]
        res[f"mAP/{d}"] = round(float(np.mean(vals)), 2) if vals else float("nan")
    return res


# ---------------------------------------------------------------- 규칙 자체검증
def _anno(lines, with_score=False):
    """[(type, trunc, occ, x1, y1, x2, y2[, score])] -> get_label_annos 와 같은 dict"""
    a = {"name": np.array([l[0] for l in lines], dtype=object).astype(str) if lines else np.array([]),
         "truncated": np.array([float(l[1]) for l in lines]),
         "occluded": np.array([int(l[2]) for l in lines]),
         "alpha": np.full(len(lines), -10.0),
         "bbox": np.array([[float(v) for v in l[3:7]] for l in lines]).reshape(-1, 4)}
    a["score"] = np.array([float(l[7]) for l in lines]) if with_score else np.zeros(len(lines))
    return a


def _stats(gts, dets, cls, diff="Moderate"):
    """한 장에 대해 공식 엔진의 (TP, FP, FN, 유효GT수) — 임계값 0"""
    g, d = _anno(gts), _anno(dets, with_score=True)
    ci, di = CLASSES.index(cls), DIFFS.index(diff)
    gl, dl, ig, idt, dc, _, nv = E._prepare_data([g], [d], ci, di)
    ov = E.image_box_overlap(d["bbox"], g["bbox"])                   # (n_det, n_gt)
    tp, fp, fn, _, _ = E.compute_statistics_jit(ov, gl[0], dl[0], ig[0], idt[0], dc[0], 0,
                                                MIN_OVERLAP[0, 0, ci], 0.0, True)
    return int(tp), int(fp), int(fn), int(nv)


def _ap(gts, dets, cls, diff="Moderate"):
    ci, di = CLASSES.index(cls), DIFFS.index(diff)
    ret = E.eval_class([_anno(gts)], [_anno(dets, True)], [ci], [di], 0, MIN_OVERLAP[:, :, [ci]])
    return float(E.get_mAP_R40(ret["precision"])[0, 0, 0])


def selftest():
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(("  OK   " if cond else "  FAIL ") + name)
        ok = ok and bool(cond)

    car = ("Car", 0.0, 0, 0, 0, 100, 100)
    det = lambda c, b, s=0.9: (c, -1, -1, *b, s)               # noqa: E731
    chk("완전 일치 -> TP1 FP0 FN0", _stats([car], [det("Car", (0, 0, 100, 100))], "Car")[:3] == (1, 0, 0))
    # AP40 은 GT 가 충분히 많아야 의미가 있다: 공식 알고리즘은 TP 점수로 '재현율 1/40 간격' 임계값을
    # 고르므로, GT 가 1~2개면 임계값이 0번 칸 하나뿐이라 AP40=0 이 된다 (공식 C++ 도 동일).
    many = [("Car", 0.0, 0, 10 * k, 0, 10 * k + 8, 100) for k in range(80)]
    hits = [det("Car", g[3:7], 0.5 + k / 200) for k, g in enumerate(many)]
    chk("GT 1개 완전 검출 -> AP40 0 (공식 알고리즘의 표본 수 한계)",
        _ap([car], [det("Car", (0, 0, 100, 100))], "Car") == 0.0)
    chk("GT 80개 완전 검출 -> AP40 100", abs(_ap(many, hits, "Car") - 100) < 1e-6)
    chk("GT 80개 중 40개 완전 검출 -> AP40 50", abs(_ap(many, hits[:40], "Car") - 50) < 1e-6)
    small = ("Car", 0.0, 0, 0, 0, 100, 20)
    chk("높이 20px GT 는 Moderate 유효GT 아님, 거기 맞은 예측도 FP 아님",
        _stats([small], [det("Car", (0, 0, 100, 20))], "Car") == (0, 0, 0, 0))
    occ2 = ("Car", 0.0, 2, 0, 0, 100, 100)
    chk("occlusion=2 -> Moderate 제외 / Hard 포함",
        _stats([occ2], [], "Car")[3] == 0 and _stats([occ2], [], "Car", "Hard")[3] == 1)
    chk("truncation=0.40 -> Moderate 제외", _stats([("Car", 0.4, 0, 0, 0, 100, 100)], [], "Car")[3] == 0)
    chk("Van 위의 Car 예측은 FP 아님",
        _stats([("Van", 0.0, 0, 0, 0, 100, 100)], [det("Car", (0, 0, 100, 100))], "Car") == (0, 0, 0, 0))
    chk("Person_sitting 위의 Pedestrian 예측은 FP 아님",
        _stats([("Person_sitting", 0.0, 0, 0, 0, 60, 120)], [det("Pedestrian", (0, 0, 60, 120))],
               "Pedestrian") == (0, 0, 0, 0))
    dc = ("DontCare", -1, -1, 0, 0, 200, 200)
    chk("DontCare 안에 완전히 들어간 예측은 버림", _stats([dc], [det("Car", (0, 0, 100, 100))], "Car")[1] == 0)
    # 예측 면적의 60%만 DontCare 와 겹침: Car(임계 0.7)는 FP, Pedestrian(임계 0.5)은 버림
    part = (140, 0, 240, 100)
    chk("DontCare 겹침 0.6 -> Car 는 FP (클래스별 임계 0.7)", _stats([dc], [det("Car", part)], "Car")[1] == 1)
    chk("DontCare 겹침 0.6 -> Pedestrian 은 버림 (임계 0.5)",
        _stats([dc], [det("Pedestrian", part)], "Pedestrian")[1] == 0)
    chk("Car IoU 0.68 -> FP1 FN1", _stats([car], [det("Car", (0, 0, 100, 68))], "Car")[:3] == (0, 1, 1))
    ped = ("Pedestrian", 0.0, 0, 0, 0, 100, 100)
    chk("Pedestrian IoU 0.68 -> TP (임계 0.5)",
        _stats([ped], [det("Pedestrian", (0, 0, 100, 68))], "Pedestrian")[0] == 1)
    chk("중복 예측 -> TP1 FP1",
        _stats([car], [det("Car", (0, 0, 100, 100)), det("Car", (0, 0, 100, 100), 0.8)], "Car")[:2] == (1, 1))
    chk("높이 25px 미만 예측은 GT 없어도 FP 아님", _stats([car], [det("Car", (300, 0, 400, 20))], "Car")[1] == 0)
    # 공식 매칭은 점수가 아니라 겹침 기준: 점수 높은 IoU 0.75 예측이 FP, 점수 낮은 IoU 0.95 예측이 TP
    chk("매칭은 겹침 최대 우선 (점수 순 아님)",
        _stats([car], [det("Car", (0, 0, 100, 75), 0.9), det("Car", (0, 0, 100, 95), 0.5)], "Car")[:2] == (1, 1))
    print("[selftest]", "통과" if ok else "실패")
    return ok


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt"); ap.add_argument("--pred")
    ap.add_argument("--split", default="splits/eval_val.txt")
    ap.add_argument("--tag", default="", help="모델 이름. 결과 파일명과 results.jsonl 기록에 쓴다")
    ap.add_argument("--out", default=None, help="기본: benchmarks/ap40_<tag>.json")
    ap.add_argument("--jsonl", default="benchmarks/results.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not (a.gt and a.pred):
        sys.exit("--gt 와 --pred 가 필요하다")

    ids = Path(a.split).read_text().split()
    t0 = time.time()
    res = evaluate(a.gt, a.pred, ids)
    res.update(tag=a.tag, n_images=len(ids), split=a.split,
               engine="kitti_object_eval_python (official devkit port)")

    print(f"{'':12s}" + "".join(f"{d:>10s}" for d in DIFFS))
    for c in CLASSES:
        print(f"{c:12s}" + "".join(f"{res[f'{c}/{d}']['AP40']:>10.2f}" for d in DIFFS)
              + f"   (Moderate GT {res[f'{c}/Moderate']['n_gt']})")
    print(f"{'mAP':12s}" + "".join(f"{res[f'mAP/{d}']:>10.2f}" for d in DIFFS))
    print(f"\n>>> 공식 점수 (Moderate AP40, 3클래스 평균): {res['mAP/Moderate']}   [{time.time() - t0:.1f}s]")

    out = Path(a.out or f"benchmarks/ap40_{a.tag or 'untagged'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    jsonl_append(a.jsonl, {
        "kind": "ap40", "model": a.tag, "split": a.split, "n_images": len(ids),
        "mAP_moderate": res["mAP/Moderate"],
        **{f"{c}_moderate": res[f"{c}/Moderate"]["AP40"] for c in CLASSES},
        "mAP_easy": res["mAP/Easy"], "mAP_hard": res["mAP/Hard"],
        "engine": "official-port", "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})


if __name__ == "__main__":
    main()
