"""고정 평가셋 생성/검증 + leakage 없는 train split 생성.

이 스크립트가 학습/평가 split의 **단일 진입점**이다.
다른 곳에서 절대 임의로 split을 만들지 않는다 (Data Leakage 적발 시 실격).

  python scripts/make_split.py --verify splits/eval_val.txt   # 주최측 파일 검증
  python scripts/make_split.py --emit                          # 규칙대로 재생성
  python scripts/make_split.py --kitti-root data/kitti_raw --write-splits
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

N_KITTI = 7481  # KITTI training split 총 장수: 000000 ~ 007480


def canonical_eval_ids():
    """09/09 공지 첨부 eval_val.txt의 생성 규칙.

    전체 7481장을 stride 7.5로 균일 샘플링한 1000개.
      [0] + [floor(k*7.5)+1 for k in 0..997] + [7480]
    """
    ids = [0] + [k * 15 // 2 + 1 for k in range(998)] + [N_KITTI - 1]
    assert len(ids) == len(set(ids)) == 1000
    return ids


def verify(path):
    raw = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    ids = [int(x) for x in raw]
    ok = True
    def chk(cond, msg):
        nonlocal ok
        print(("  OK   " if cond else "  FAIL ") + msg)
        ok = ok and cond
    print(f"[검증] {path}")
    chk(len(ids) == 1000, f"총 개수 1000 (실제 {len(ids)})")
    chk(len(set(ids)) == len(ids), f"중복 없음 (고유 {len(set(ids))})")
    chk(all(0 <= i < N_KITTI for i in ids), f"범위 0~{N_KITTI-1} (min {min(ids)}, max {max(ids)})")
    chk(all(len(x) == 6 for x in raw), "모든 항목이 6자리 제로패딩")
    chk(sorted(ids) == ids, "오름차순 정렬")
    chk(set(ids) == set(canonical_eval_ids()), "공지 규칙(stride 7.5 균일 샘플링)과 일치")
    print("[검증]", "통과" if ok else "실패")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", metavar="FILE")
    ap.add_argument("--emit", action="store_true", help="규칙대로 eval_val.txt 재생성")
    ap.add_argument("--out", default="splits/eval_val.txt")
    ap.add_argument("--kitti-root", help="KITTI training/ 상위 경로")
    ap.add_argument("--write-splits", action="store_true",
                    help="eval_val 제외한 train.txt / holdout.txt 생성")
    ap.add_argument("--holdout", type=int, default=500,
                    help="자체 검증용 홀드아웃 장수 (eval_val과 무관, 학습 제외)")
    ap.add_argument("--guard", type=int, default=0,
                    help=("eval 인덱스 +-N 프레임을 학습에서 함께 제외. "
                          "eval_val이 stride 7.5로 촘촘해 비용이 매우 크다 "
                          "(0:6481장 / 1:4486장 / 2:2492장 / 3:498장). 기본 0 권장."))
    a = ap.parse_args()

    if a.verify:
        sys.exit(0 if verify(a.verify) else 1)

    if a.emit:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(f"{i:06d}" for i in canonical_eval_ids()) + "\n")
        print(f"[생성] {a.out} (1000장)")

    if a.write_splits:
        eval_ids = set(int(x) for x in Path(a.out).read_text().split())
        assert len(eval_ids) == 1000, "eval_val.txt가 1000장이 아니다"

        # 인접 프레임 가드: eval 인덱스 주변을 학습에서 뺀다.
        # KITTI는 연속 주행 프레임이라 ±1~3 프레임이 학습에 남으면 낙관적 결과가 나온다.
        banned = set()
        for i in eval_ids:
            for d in range(-a.guard, a.guard + 1):
                if 0 <= i + d < N_KITTI:
                    banned.add(i + d)

        pool = [i for i in range(N_KITTI) if i not in banned]
        if len(pool) < a.holdout * 2:
            sys.exit(f"[중단] guard={a.guard}는 학습풀을 {len(pool)}장까지 줄인다. "
                     f"--guard 를 낮춰라 (0 권장).")
        if a.guard:
            print(f"[경고] guard={a.guard}로 학습풀이 {N_KITTI-1000} -> {len(pool)}장으로 줄었다. "
                  f"공지는 eval 인덱스 본체만 금지하므로 guard=0도 규정 준수다.")
        holdout = pool[:: max(1, len(pool) // a.holdout)][: a.holdout]
        hset = set(holdout)
        train = [i for i in pool if i not in hset]

        d = Path(a.out).parent
        (d / "train.txt").write_text("\n".join(f"{i:06d}" for i in train) + "\n")
        (d / "holdout.txt").write_text("\n".join(f"{i:06d}" for i in sorted(hset)) + "\n")

        assert not (set(train) & eval_ids), "LEAKAGE: train이 eval_val과 겹친다"
        assert not (hset & eval_ids), "LEAKAGE: holdout이 eval_val과 겹친다"
        print(f"[생성] train.txt {len(train)}장 / holdout.txt {len(hset)}장 / eval_val 1000장")
        print(f"       가드(+-{a.guard} 프레임)로 추가 제외: {len(banned) - 1000}장")
        print("       leakage 검사 통과")


if __name__ == "__main__":
    main()
