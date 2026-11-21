#!/usr/bin/env python3
"""본선 결과 정리: 서버(예선 ONNX Runtime) vs 엣지(Hailo-10H 보드) 표 + DECISION_FINAL.md 규칙 판정.

  python hailo/summarize_final.py            # -> hailo/results/board/final/SUMMARY.md

읽는 것 (없는 항목은 '-' 로 두고 넘어간다)
  서버  benchmarks/ap40_holdout_*.json, benchmarks/eval_val_cloud/*.json (Xeon VNNI 채점)
        benchmarks/m1pro/bench_key15_m1pro.jsonl (M1 Pro 4스레드 지연·RSS)
  엣지  hailo/results/<A~D>/ap40_*_holdout_quantized.json (에뮬레이터)
        hailo/results/board/npu_fps_rep3.txt, e2e/*.json, ap_board.txt (11/19, 06 방식)
        hailo/results/board/ap40/ap40_board_evalval_*.json (11/19)
        hailo/results/board/final/fast_*.jsonl, best_*.jsonl, ap40_fast_holdout_*.json (07)

HEF 변형(예: D_640x192_Q1.hef)도 fast_/best_/ap40_fast_ 파일이 있으면 같은 구조 줄 아래에 붙는다.
"""
from __future__ import annotations
import json, re, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
B = REPO / "hailo" / "results" / "board"
F = B / "final"

ARCH = {  # 문자 -> (예선 ONNX 이름, 해상도, 설명)
    "A": ("best_1280x384", "1280x384", "P5 있음"),
    "B": ("best-p34_1280x384", "1280x384", "P5 제거"),
    "C": ("best_640x192", "640x192", "P5 있음"),
    "D": ("best-p34_640x192", "640x192", "P5 제거"),
}
INT8 = "int8-mixed-percentile"


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def jl(p):
    try:
        return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    except Exception:
        return []


def mod(d):
    if not d:
        return None
    if "mAP_moderate" in d:
        return d["mAP_moderate"]
    for k in ("mAP/Moderate", "Overall/Moderate", "mAP"):
        if k in d:
            v = d[k]
            return v["AP40"] if isinstance(v, dict) else v
    cls = [d.get(f"{c}/Moderate", {}).get("AP40") for c in ("Car", "Pedestrian", "Cyclist")]
    return round(sum(cls) / 3, 2) if all(v is not None for v in cls) else None


def f(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def parse_txt_scores(path):
    """ap_board.txt 같은 '== <이름>' 블록 + '공식 점수 ...: 96.85' 줄."""
    out, cur = {}, None
    try:
        for ln in Path(path).read_text().splitlines():
            m = re.match(r"== (\S+)", ln)
            if m:
                cur = m.group(1)
            m = re.search(r"공식 점수.*?:\s*([\d.]+)", ln)
            if m and cur:
                out[cur] = float(m.group(1))
    except Exception:
        pass
    return out


def npu_fps():
    vals = {}
    try:
        for ln in (B / "npu_fps_rep3.txt").read_text().splitlines():
            m = re.match(r"(\S+) r\d \S+ fps: ([\d.]+)", ln)
            if m:
                vals.setdefault(m.group(1), []).append(float(m.group(2)))
    except Exception:
        pass
    return {k: statistics.median(v) for k, v in vals.items()}


def m1_bench():
    out = {}
    for r in jl(REPO / "benchmarks" / "m1pro" / "bench_key15_m1pro.jsonl"):
        out[r["model"].replace("_op13", "").replace(".onnx", "")] = r
    return out


def best_fast(stem):
    """07 결과: best_*.jsonl(같은 조합 3회) 가 있으면 그 중앙값, 없으면 sweep 의 최고 FPS 1회."""
    reps = [r for r in jl(F / f"best_{stem}.jsonl") if r.get("mode") == "async"]
    if reps:
        fps = statistics.median(r["fps"] for r in reps)
        r0 = sorted(reps, key=lambda r: r["fps"])[len(reps) // 2]
        return {**r0, "fps": fps, "n_rep": len(reps)}
    rows = [r for r in jl(F / f"fast_{stem}.jsonl") if r.get("mode") == "async"]
    if not rows:
        return None
    r = max(rows, key=lambda r: r["fps"])
    return {**r, "n_rep": 1}


def sync_fast(stem):
    rows = [r for r in jl(F / f"fast_{stem}.jsonl") if r.get("mode") == "sync"]
    return rows[-1] if rows else None


def hef_stems():
    stems = {f"{k}_{v[1]}" for k, v in ARCH.items()}
    for p in list(F.glob("fast_*.jsonl")) + list(F.glob("ap40_fast_holdout_*.json")) + list(F.glob("best_*.jsonl")):
        m = re.match(r"(?:fast_|best_|ap40_fast_holdout_)(.+?)\.jsonl?$", p.name)
        if m:
            stems.add(m.group(1))
    return sorted(stems)


def main():
    F.mkdir(parents=True, exist_ok=True)
    ab = parse_txt_scores(B / "ap_board.txt")
    npu = npu_fps()
    m1 = m1_bench()
    rows = {}
    for stem in hef_stems():
        L = stem[0]
        onnx, wh, desc = ARCH[L]
        e2e06 = jload(B / "e2e" / f"{L}_{wh}.json") if stem == f"{L}_{wh}" else None
        fast = best_fast(stem)
        rows[stem] = {
            "L": L, "stem": stem, "wh": wh, "desc": desc,
            "srv_fp32_hold": mod(jload(REPO / "benchmarks" / f"ap40_holdout_{onnx}_op13_fp32.json")),
            "srv_int8_hold": mod(jload(REPO / "benchmarks" / f"ap40_holdout_{onnx}_op13_{INT8}.json")),
            "srv_fp32_ev": mod(jload(REPO / "benchmarks" / "eval_val_cloud" / f"ap40_eval_val_{onnx}_op13_fp32.json")),
            "srv_int8_ev": mod(jload(REPO / "benchmarks" / "eval_val_cloud" / f"ap40_eval_val_{onnx}_op13_{INT8}.json")),
            "m1_fp32": m1.get(f"{onnx}_fp32"), "m1_int8": m1.get(f"{onnx}_{INT8}"),
            "emu_hold": mod(jload(REPO / "hailo" / "results" / L / f"ap40_{L}_holdout_quantized.json")) if stem == f"{L}_{wh}" else None,
            "brd_hold_06": ab.get(stem),
            "brd_hold_07": mod(jload(F / f"ap40_fast_holdout_{stem}.json")),
            "brd_ev": mod(jload(B / "ap40" / f"ap40_board_evalval_{stem}.json")),
            "npu_fps": npu.get(stem), "e2e06": e2e06, "fast": fast, "sync": sync_fast(stem),
        }

    md = ["# 본선 결과 요약 (자동 생성: hailo/summarize_final.py)", ""]
    md += ["## 1. 정확도 — Moderate AP40", "",
           "| 모델 | 서버 FP32 holdout | 서버 INT8 holdout | Hailo 에뮬 holdout | 보드 holdout (07) | 보드 holdout (11/19) "
           "| 서버 FP32 eval_val | 서버 INT8 eval_val | 보드 eval_val |",
           "|---|---|---|---|---|---|---|---|---|"]
    for s, r in rows.items():
        md.append(f"| {s} ({r['desc']}) | {f(r['srv_fp32_hold'])} | {f(r['srv_int8_hold'])} | {f(r['emu_hold'])} | "
                  f"**{f(r['brd_hold_07'])}** | {f(r['brd_hold_06'])} | {f(r['srv_fp32_ev'])} | {f(r['srv_int8_ev'])} | {f(r['brd_ev'])} |")
    md += ["", "서버 INT8 = ONNX Runtime int8-mixed-percentile (예선 주력 방식). 보드 = Hailo-10H INT8.",
           "보드 holdout (07) 은 07_board_fast.py --dump (conf 0.001) 채점. 규칙 판정은 이 열만 쓴다.", ""]

    md += ["## 2. 속도·자원", "",
           "| 모델 | M1 FP32 p50 ms | M1 INT8 p50 ms | M1 INT8 FPS | M1 INT8 RSS MB | 보드 NPU 단독 FPS | 보드 직렬 FPS (06) "
           "| 보드 직렬 FPS (07 sync) | **보드 파이프라인 FPS (07)** | 조합 b/i/형식 | 지연 p50 ms | CPU% | RSS MB | 온도 C |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, r in rows.items():
        a, b = r["m1_fp32"] or {}, r["m1_int8"] or {}
        fa, sy = r["fast"] or {}, r["sync"] or {}
        combo = f"{fa.get('batch')}/{fa.get('inflight')}/{fa.get('out_format')}" if fa else "-"
        if fa and fa.get("n_rep", 1) > 1:
            combo += f" (x{fa['n_rep']})"
        md.append(f"| {s} | {f(a.get('p50_ms'))} | {f(b.get('p50_ms'))} | {f(b.get('fps_p50'), 1)} | {f(b.get('peak_rss_MB'), 0)} | "
                  f"{f(r['npu_fps'], 1)} | {f((r['e2e06'] or {}).get('fps_serial'), 1)} | {f(sy.get('fps'), 1)} | "
                  f"**{f(fa.get('fps'), 1)}** | {combo} | {f(fa.get('lat_p50_ms'), 1)} | {f(fa.get('proc_cpu_pct_mean'), 0)} | "
                  f"{f(fa.get('rss_mb_max'), 0)} | {f(fa.get('temp_c_max'), 1)} |")
    md += ["", "M1 = Apple M1 Pro 4스레드, 예열 120초, 15라운드 p50 (예선 run_m1.sh). 보드 FPS 는 전처리~후처리 포함, "
           "입력은 메모리의 디코드된 이미지 (카메라와 같은 조건). CPU% 는 코어 1개 = 100.", ""]

    # ---- 규칙 판정
    md += ["## 3. DECISION_FINAL.md 규칙 판정", ""]

    def best_variant(L, wh):
        base = f"{L}_{wh}"
        cand = [r for s, r in rows.items() if r["L"] == L]
        b = rows.get(base)
        lines, chosen = [], b
        if b is None:
            return None, lines
        for r in cand:
            if r is b:
                continue
            if None in (r["brd_hold_07"], b["brd_hold_07"]) or not (r["fast"] and b["fast"]):
                lines.append(f"- {r['stem']}: 데이터 부족 — 보류"); continue
            d = r["brd_hold_07"] - b["brd_hold_07"]; x = r["fast"]["fps"] / b["fast"]["fps"]
            ok = d >= 0.30 and x >= 0.95
            lines.append(f"- {r['stem']} vs {base}: holdout {d:+.2f} (기준 +0.30), FPS {x:.3f}배 (기준 0.95) -> {'통과' if ok else '탈락'}")
            if ok and (chosen is b or r["brd_hold_07"] > chosen["brd_hold_07"] + 0.10 or
                       (abs(r["brd_hold_07"] - chosen["brd_hold_07"]) <= 0.10 and r["fast"]["fps"] > chosen["fast"]["fps"])):
                chosen = r
        return chosen, lines

    final = {}
    for res, (old, new) in (("1280x384", ("A", "B")), ("640x192", ("C", "D"))):
        co, lo = best_variant(old, res)
        cn, ln_ = best_variant(new, res)
        md += [f"### {res}", ""] + lo + ln_
        if not co or not cn or None in (co["brd_hold_07"], cn["brd_hold_07"]) or not (co["fast"] and cn["fast"]):
            md += [f"- {new} vs {old}: 데이터 부족 — 보류 (07 sweep·ap 결과 필요)", ""]
            final[res] = None; continue
        d = cn["brd_hold_07"] - co["brd_hold_07"]; x = cn["fast"]["fps"] / co["fast"]["fps"]
        ok = d >= -0.5 and x >= 1.10
        win = cn if ok else co
        final[res] = win["stem"]
        md += [f"- {cn['stem']} vs {co['stem']}: holdout {d:+.2f} (기준 -0.50 이상), FPS {x:.3f}배 (기준 1.10 이상) "
               f"-> {'대체 통과' if ok else '대체 탈락'}",
               f"- **{res} 주력: {win['stem']}**", ""]
    md += ["eval_val 은 판정에 쓰지 않았다.", ""]
    (F / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n-> {F / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
