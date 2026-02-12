def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0
def nms(boxes, iou_thresh=0.45):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
    kept = []

    while boxes:
        best = boxes.pop(0)
        kept.append(best)

        boxes = [
            b for b in boxes
            if iou(best['coordinates'], b['coordinates']) < iou_thresh
        ]

    return kept
# containment ratio is the ratio of the inner area to the outer area (Containmnet Supression)
def containment_ratio(inner, outer):
    x1, y1, x2, y2 = inner
    ox1, oy1, ox2, oy2 = outer

    ix1 = max(x1, ox1)
    iy1 = max(y1, oy1)
    ix2 = min(x2, ox2)
    iy2 = min(y2, oy2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    inner_area = (x2 - x1) * (y2 - y1)

    return inter / inner_area if inner_area > 0 else 0
def area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)
def dominance_suppression(boxes, overlap_thresh=0.4, area_ratio=1.8):
    kept = []

    boxes = sorted(boxes, key=lambda x: x["conf"], reverse=True)

    for a in (boxes):
        suppress = False
        for b in kept:
            cont = containment_ratio(a["coordinates"], b["coordinates"])
            
            if cont > overlap_thresh:
                if area(b["coordinates"]) > area_ratio * area(a["coordinates"]):
                    suppress = False
                    break
        if not suppress:
            kept.append(a)

    return kept