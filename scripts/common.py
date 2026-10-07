"""공용 유틸 — 전처리/후처리는 반드시 이 모듈 하나만 쓴다.

캘리브레이션·벤치마크·추론이 서로 다른 전처리를 쓰면 PTQ 스케일이 통째로 어긋나므로
(예선 트러블슈팅 4-2 참조) letterbox 구현을 여기 한 곳에만 둔다.
"""
from __future__ import annotations
import json, os
from pathlib import Path

import numpy as np

# KITTI 공식 2D 평가 대상 3클래스
CLASSES = ["Car", "Pedestrian", "Cyclist"]
# 평가 시 FP로 세지 않는 이웃 클래스 (공식 eval.cpp의 neighboring class)
NEIGHBOR = {"Car": "Van", "Pedestrian": "Person_sitting"}
# 클래스별 2D IoU 임계값 (KITTI Object Detection Evaluation 2012)
IOU_THR = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
# 난이도 기준: (최소 bbox 높이 px, 최대 occlusion, 최대 truncation)
DIFFICULTY = {
    "Easy":     (40, 0, 0.15),
    "Moderate": (25, 1, 0.30),
    "Hard":     (25, 2, 0.50),
}


def parse_imgsz(s):
    """'1280x384' | '1280,384' | '640' -> (W, H). 32의 배수 강제 검증."""
    if isinstance(s, (tuple, list)):
        w, h = int(s[0]), int(s[1])
    else:
        t = str(s).lower().replace("*", "x").replace(",", "x")
        parts = [p for p in t.split("x") if p]
        w = int(parts[0])
        h = int(parts[1]) if len(parts) > 1 else int(parts[0])
    if w % 32 or h % 32:
        raise ValueError(f"해상도는 32의 배수여야 한다: {w}x{h}")
    if w > 1280:
        raise ValueError(f"공지상 가로 해상도 상한은 1280이다: {w}")
    return w, h


def letterbox(img, dst_w, dst_h, color=114):
    """종횡비 유지 리사이즈 + 중앙 패딩. (out, scale, pad_x, pad_y) 반환.

    보간은 항상 INTER_LINEAR — Ultralytics predict 의 LetterBox, 그리고 학습(augment) 경로의
    리사이즈와 같게 맞춘다. 축소 시 INTER_AREA 를 쓰면 640x192 에서 Ultralytics 결과와
    박스·점수가 미세하게 달라진다 (2026-10-06 확인: 1280x384 는 원래도 일치).
    """
    import cv2
    h, w = img.shape[:2]
    r = min(dst_w / w, dst_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (dst_w - nw) / 2, (dst_h - nh) / 2
    l, t = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
    b, rr = dst_h - nh - t, dst_w - nw - l
    img = cv2.copyMakeBorder(img, t, b, l, rr, cv2.BORDER_CONSTANT, value=(color,) * 3)
    return img, r, l, t


def preprocess(path, dst_w, dst_h):
    """이미지 경로 -> (NCHW float32 [0,1] RGB, scale, pad_x, pad_y, 원본 hw)"""
    import cv2
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(path)
    oh, ow = im.shape[:2]
    lb, r, px, py = letterbox(im, dst_w, dst_h)
    x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), r, px, py, (oh, ow)


def nms(boxes, scores, thr):
    """단일 클래스 NMS. boxes: (N,4) xyxy."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


def decode_yolo(out, r, px, py, oh, ow, conf_thr=0.001, iou_thr=0.65, max_det=300):
    """YOLO raw head [1, 4+nc, A] -> [(cls, score, x1,y1,x2,y2)] (원본 이미지 좌표).

    nms=False로 export한 출력 전제. 행 0~3 = cx,cy,w,h (입력 픽셀), 행 4~ = 클래스 점수.
    """
    p = out[0] if out.ndim == 3 else out           # (4+nc, A)
    if p.shape[0] > p.shape[1]:                     # (A, 4+nc)로 나온 경우
        p = p.T
    nc = p.shape[0] - 4
    box, cls = p[:4], p[4:]
    conf = cls.max(0)
    cid = cls.argmax(0)
    m = conf > conf_thr
    if not m.any():
        return []
    box, conf, cid = box[:, m], conf[m], cid[m]
    cx, cy, w, h = box
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    # letterbox 역변환
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - px) / r
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - py) / r
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, ow)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, oh)

    res = []
    for c in range(nc):
        s = cid == c
        if not s.any():
            continue
        b, sc = xyxy[s], conf[s]
        for i in nms(b, sc, iou_thr):
            res.append((c, float(sc[i]), *[float(v) for v in b[i]]))
    res.sort(key=lambda t: -t[1])
    return res[:max_det]


def model_input_wh(sess):
    """ONNX Runtime 세션의 고정 입력 크기 -> (W, H). 동적 축이면 None."""
    shp = sess.get_inputs()[0].shape          # [1, 3, H, W]
    h, w = shp[2], shp[3]
    return (w, h) if isinstance(w, int) and isinstance(h, int) else None


def jsonl_append(path, rec):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
