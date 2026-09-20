"""KITTI 2D Object Detection — Moderate AP40 평가.

공식 eval.cpp 의 판정 규칙을 재현한다. Ultralytics 의 val() 로는 이 수치를 낼 수 없다.

규칙 (놓치면 AP가 부당하게 낮게 나온다):
  1. 난이도 필터 — Moderate: 높이>=25px, occlusion<=1, truncation<=0.30
  2. 이웃 클래스 무시 — Car 판정에서 Van, Pedestrian 판정에서 Person_sitting 은
     GT로 세지도 않고, 거기 맞은 예측을 FP로 세지도 않는다.
  3. 난이도 미달 GT — FN 으로 세지 않는다. 거기 맞은 예측도 FP가 아니다.
  4. DontCare 영역 — 예측 박스 면적 대비 교집합(IoA)>=0.5 면 그 예측을 버린다.
  5. 높이 미달 예측 — FP 로 세지 않고 버린다.
  6. IoU 임계값 — Car 0.70, Pedestrian 0.50, Cyclist 0.50
  7. AP40 — recall 1/40..40/40 의 40개 지점에서 보간 precision 평균

  python scripts/kitti_eval.py --gt data/kitti_raw/training/label_2 \
      --pred runs/pred_eval_val --split splits/eval_val.txt
  python scripts/kitti_eval.py --selftest
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import CLASSES, NEIGHBOR, IOU_THR, DIFFICULTY  # noqa: E402

import numpy as np

DC_IOA = 0.5   # DontCare 판정 임계


def read_kitti(path, with_score=False):
    """KITTI 라벨/예측 파일 -> list of dict"""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for ln in p.read_text().splitlines():
        f = ln.split()
        if len(f) < 15:
            continue
        d = {"type": f[0], "truncation": float(f[1]), "occlusion": int(float(f[2])),
             "bbox": np.array([float(v) for v in f[4:8]], dtype=np.float64)}
        if with_score:
            d["score"] = float(f[15]) if len(f) > 15 else 1.0
        out.append(d)
    return out


def iou_mat(a, b):
    """a:(N,4) b:(M,4) xyxy -> IoU (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a.T[:, :, None]
    bx1, by1, bx2, by2 = b.T[:, None, :]
    iw = np.minimum(ax2, bx2) - np.maximum(ax1, bx1)
    ih = np.minimum(ay2, by2) - np.maximum(ay1, by1)
    inter = iw.clip(0) * ih.clip(0)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / np.maximum(ua, 1e-9)


def ioa_mat(det, dc):
    """예측 박스 면적 대비 교집합 비율 (N_det, N_dc)"""
    if len(det) == 0 or len(dc) == 0:
        return np.zeros((len(det), len(dc)))
    ax1, ay1, ax2, ay2 = det.T[:, :, None]
    bx1, by1, bx2, by2 = dc.T[:, None, :]
    iw = np.minimum(ax2, bx2) - np.maximum(ax1, bx1)
    ih = np.minimum(ay2, by2) - np.maximum(ay1, by1)
    inter = (iw.clip(0) * ih.clip(0))
    area = np.maximum((ax2 - ax1) * (ay2 - ay1), 1e-9)
    return inter / area


def split_gt(gts, cls, diff):
    """GT를 (평가대상, 무시, DontCare) 로 분류"""
    min_h, max_occ, max_tr = DIFFICULTY[diff]
    valid, ignored, dc = [], [], []
    for g in gts:
        if g["type"] == "DontCare":
            dc.append(g["bbox"])
            continue
        h = g["bbox"][3] - g["bbox"][1]
        too_hard = (h < min_h) or (g["occlusion"] > max_occ) or (g["truncation"] > max_tr)
        if g["type"] == cls:
            (ignored if too_hard else valid).append(g["bbox"])
        elif NEIGHBOR.get(cls) == g["type"]:
            ignored.append(g["bbox"])          # 이웃 클래스는 항상 무시
    f = lambda L: np.array(L, dtype=np.float64).reshape(-1, 4)
    return f(valid), f(ignored), f(dc)


def eval_image(gts, dets, cls, diff, thr):
    """한 장 처리 -> (scores, is_tp, n_gt)"""
    valid, ignored, dc = split_gt(gts, cls, diff)
    min_h = DIFFICULTY[diff][0]

    d = [x for x in dets if x["type"] == cls]
    d.sort(key=lambda x: -x["score"])
    db = np.array([x["bbox"] for x in d], dtype=np.float64).reshape(-1, 4)
    ds = np.array([x["score"] for x in d], dtype=np.float64)

    iou_v = iou_mat(db, valid)
    iou_i = iou_mat(db, ignored)
    ioa_d = ioa_mat(db, dc)

    taken = np.zeros(len(valid), dtype=bool)
    scores, tps = [], []
    for k in range(len(d)):
        # 1) 평가대상 GT 매칭 (미할당 중 최대 IoU)
        if len(valid):
            cand = np.where(~taken, iou_v[k], -1.0)
            j = int(cand.argmax())
            if cand[j] >= thr:
                taken[j] = True
                scores.append(ds[k]); tps.append(1)
                continue
        # 2) 무시 GT(난이도 미달 / 이웃 클래스)에 맞으면 버린다
        if len(ignored) and iou_i[k].max() >= thr:
            continue
        # 3) 높이 미달 예측은 FP로 세지 않는다
        if (db[k][3] - db[k][1]) < min_h:
            continue
        # 4) DontCare 영역에 걸친 예측은 버린다
        if len(dc) and ioa_d[k].max() >= DC_IOA:
            continue
        scores.append(ds[k]); tps.append(0)
    return scores, tps, int(len(valid))


def ap40(scores, tps, n_gt):
    """40-point 보간 AP (KITTI 2017 이후 공식 방식)"""
    if n_gt == 0:
        return float("nan")
    if not scores:
        return 0.0
    o = np.argsort(-np.asarray(scores, dtype=np.float64))
    tp = np.asarray(tps, dtype=np.float64)[o]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    # 단조 감소 보간
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    total = 0.0
    for i in range(1, 41):
        r = i / 40.0
        m = rec >= r
        total += prec[m].max() if m.any() else 0.0
    return total / 40.0


def evaluate(gt_dir, pred_dir, ids, difficulties=("Easy", "Moderate", "Hard")):
    res = {}
    for cls in CLASSES:
        thr = IOU_THR[cls]
        for diff in difficulties:
            S, T, G = [], [], 0
            for i in ids:
                g = read_kitti(Path(gt_dir) / f"{i}.txt")
                p = read_kitti(Path(pred_dir) / f"{i}.txt", with_score=True)
                s, t, n = eval_image(g, p, cls, diff, thr)
                S += s; T += t; G += n
            res[f"{cls}/{diff}"] = {"AP40": round(100 * ap40(S, T, G), 2),
                                    "n_gt": G, "n_det": len(S)}
    missing = [c for c in CLASSES if res[f"{c}/Moderate"]["n_gt"] == 0]
    if missing:
        res["warning"] = (f"GT가 0인 클래스 {missing} 는 mAP 평균에서 제외되었다. "
                          f"공식 1000장 평가셋에서는 3클래스 모두 존재해야 정상이다.")
        print("[경고] " + res["warning"], file=sys.stderr)
    for diff in difficulties:
        vals = [res[f"{c}/{diff}"]["AP40"] for c in CLASSES
                if not np.isnan(res[f"{c}/{diff}"]["AP40"])]
        res[f"mAP/{diff}"] = round(float(np.mean(vals)), 2) if vals else float("nan")
        res[f"mAP/{diff}_classes"] = len(vals)
    return res


def selftest():
    """규칙별 단위 검증"""
    ok = True
    def chk(name, cond):
        nonlocal ok
        print(("  OK   " if cond else "  FAIL ") + name); ok = ok and cond

    gt_easy = {"type": "Car", "truncation": 0.0, "occlusion": 0,
               "bbox": np.array([0., 0., 100., 100.])}
    perfect = {"type": "Car", "bbox": np.array([0., 0., 100., 100.]), "score": 0.9}

    s, t, n = eval_image([gt_easy], [perfect], "Car", "Moderate", 0.7)
    chk("완전 일치 -> TP 1, n_gt 1", t == [1] and n == 1)
    chk("AP40 만점", abs(ap40(s, t, n) - 1.0) < 1e-9)

    # 난이도 미달 GT(높이 20px)는 n_gt 에서 빠지고, 거기 맞은 예측도 FP가 아니다
    small_gt = dict(gt_easy, bbox=np.array([0., 0., 100., 20.]))
    small_det = dict(perfect, bbox=np.array([0., 0., 100., 20.]))
    s, t, n = eval_image([small_gt], [small_det], "Car", "Moderate", 0.7)
    chk("높이 20px GT는 Moderate n_gt 제외", n == 0)
    chk("높이 20px 예측은 FP 아님", t == [])

    # occlusion 2 는 Moderate 제외, Hard 포함
    occ = dict(gt_easy, occlusion=2)
    chk("occlusion=2 Moderate 제외", eval_image([occ], [], "Car", "Moderate", 0.7)[2] == 0)
    chk("occlusion=2 Hard 포함", eval_image([occ], [], "Car", "Hard", 0.7)[2] == 1)

    # truncation 0.4 는 Moderate 제외
    tr = dict(gt_easy, truncation=0.4)
    chk("truncation=0.40 Moderate 제외", eval_image([tr], [], "Car", "Moderate", 0.7)[2] == 0)

    # Van 위의 Car 예측은 FP 가 아니다
    van = {"type": "Van", "truncation": 0.0, "occlusion": 0, "bbox": np.array([0., 0., 100., 100.])}
    s, t, n = eval_image([van], [perfect], "Car", "Moderate", 0.7)
    chk("Van 위 Car 예측은 FP 아님", t == [] and n == 0)

    # Person_sitting 위의 Pedestrian 예측은 FP 가 아니다
    ps = {"type": "Person_sitting", "truncation": 0.0, "occlusion": 0,
          "bbox": np.array([0., 0., 60., 120.])}
    pd = {"type": "Pedestrian", "bbox": np.array([0., 0., 60., 120.]), "score": 0.8}
    chk("Person_sitting 위 Pedestrian 예측은 FP 아님",
        eval_image([ps], [pd], "Pedestrian", "Moderate", 0.5)[1] == [])

    # DontCare 영역에 걸친 예측은 버린다
    dc = {"type": "DontCare", "truncation": 0.0, "occlusion": 0,
          "bbox": np.array([0., 0., 200., 200.])}
    chk("DontCare 영역 예측 제거", eval_image([dc], [perfect], "Car", "Moderate", 0.7)[1] == [])

    # Car IoU 0.7 경계: IoU 0.68 은 FP
    loose = dict(perfect, bbox=np.array([0., 0., 100., 68.]))
    s, t, n = eval_image([gt_easy], [loose], "Car", "Moderate", 0.7)
    chk("Car IoU<0.70 은 FP", t == [0] and n == 1)
    # Pedestrian IoU 0.5 기준에서는 같은 겹침이 TP
    gp = {"type": "Pedestrian", "truncation": 0.0, "occlusion": 0,
          "bbox": np.array([0., 0., 100., 100.])}
    dp = {"type": "Pedestrian", "bbox": np.array([0., 0., 100., 68.]), "score": 0.9}
    chk("Pedestrian IoU>=0.50 은 TP", eval_image([gp], [dp], "Pedestrian", "Moderate", 0.5)[1] == [1])

    # 중복 예측 1개는 FP
    s, t, n = eval_image([gt_easy], [perfect, dict(perfect, score=0.8)], "Car", "Moderate", 0.7)
    chk("중복 예측은 FP", t == [1, 0])

    # AP40: GT 2개 중 1개만 완벽 검출 -> 0.5
    g2 = [gt_easy, dict(gt_easy, bbox=np.array([200., 0., 300., 100.]))]
    s, t, n = eval_image(g2, [perfect], "Car", "Moderate", 0.7)
    chk("절반 검출 AP40=0.50", abs(ap40(s, t, n) - 0.5) < 1e-9)

    print("[selftest]", "통과" if ok else "실패")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt"); ap.add_argument("--pred")
    ap.add_argument("--split", default="splits/eval_val.txt")
    ap.add_argument("--out", default="benchmarks/ap40.json")
    ap.add_argument("--tag", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not (a.gt and a.pred):
        sys.exit("--gt 와 --pred 가 필요하다")

    ids = Path(a.split).read_text().split()
    res = evaluate(a.gt, a.pred, ids)
    res["tag"] = a.tag
    res["n_images"] = len(ids)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n>>> 공식 점수 (Moderate mAP40, 3클래스 평균): {res['mAP/Moderate']}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
