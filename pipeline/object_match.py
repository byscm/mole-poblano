# How alike two objects from two photos are, by their qualities (object_tags.py), without looking at where they are:
#   look    DINOv3 fingerprints (cosine)          color   dominant colors (Lab, lightness counts less)
#   height  measured height                        size    larger floor side (boxes are rough: counts less)
# quality() -> (score 0..1, parts). auto_landmarks() finds pairs that are alike AND keep the same distances
# between them in both photos (RANSAC), to align two photos with no hand-picked references.
import json, math, os
import numpy as np

WEIGHTS = dict(look=0.35, color=0.25, height=0.25, size=0.15)


def load(folder):
    m = json.load(open(os.path.join(folder, "map.json"), encoding="utf-8"))
    d = np.load(os.path.join(folder, "object_tags.npz"))
    emb = {int(i): e.astype(np.float32) for i, e in zip(d["ids"], d["emb"])}
    return {o["id"]: dict(o=o, t=o.get("tags"), emb=emb.get(o["id"])) for o in m["objects"] if o.get("tags")}


def color_distance(ca, cb):
    if not ca or not cb:
        return 99.0
    w = np.array([0.5, 1, 1])
    def one_way(x, y):
        return sum(c["share"] * min(np.linalg.norm((np.array(c["lab"]) - np.array(d["lab"])) * w) for d in y) for c in x) / max(sum(c["share"] for c in x), 1e-9)
    return (one_way(ca, cb) + one_way(cb, ca)) / 2


def quality(a, b):
    ta, tb = a["t"], b["t"]
    cos = float(a["emb"] @ b["emb"]) if a["emb"] is not None and b["emb"] is not None else 0.0
    parts = dict(
        look=float(np.clip((cos - 0.35) / 0.4, 0, 1)),
        color=float(math.exp(-color_distance(ta["colors"], tb["colors"]) / 15)),
        height=float(math.exp(-((ta["height_m"] - tb["height_m"]) / max(0.12, 0.15 * max(ta["height_m"], tb["height_m"]))) ** 2)),
        size=float(math.exp(-(math.log(max(ta["floor_sides_m"][0], 0.05) / max(tb["floor_sides_m"][0], 0.05)) / 0.5) ** 2)),
    )
    parts["look_cos"] = round(cos, 3)
    score = sum(WEIGHTS[k] * parts[k] for k in WEIGHTS)
    return score, parts


def anchor(x):
    """Floor point that is the same physical spot in both photos. An object completed to its known size: the centre
    of the whole object (each photo may see a different half of it). Otherwise: median of its own points."""
    o, t = x["o"], x["t"]
    if o.get("completed_from_known_size"):
        from draw_plan import footprint
        return np.array(footprint(o)[0], float)
    return np.array(t["anchor_m"], float) if t.get("anchor_m") else None


def heading(x):
    """Direction of the long side, only for objects completed to a known size (their outline is reliable)."""
    o = x["o"]
    if not o.get("completed_from_known_size"):
        return None
    g = math.radians(o.get("yaw_from_appearance", o["yaw_deg"]))
    return g if o["width_m"] >= o["depth_m"] else g + math.pi / 2


def auto_landmarks(A, B, min_quality=0.6, min_margin=0.15, agree=0.5, min_spread=1.0):
    """Pairs (idA, idB) alike in quality whose anchors fit ONE turn + shift between the photos.
    Only DISTINCTIVE pairs propose an alignment (much more alike than the next candidate, both ways: the 12 equal
    chairs cannot, the dining table or the bonsai can); every alike pair can then agree with it."""
    Q = {(ia, ib): quality(a, b)[0] for ia, a in A.items() for ib, b in B.items()
         if anchor(a) is not None and anchor(b) is not None}
    if not Q:
        return [], None, []
    def margin(ia, ib):
        row = sorted((q for (x, y), q in Q.items() if x == ia and y != ib), reverse=True)
        col = sorted((q for (x, y), q in Q.items() if y == ib and x != ia), reverse=True)
        return Q[(ia, ib)] - max(row[:1] + col[:1] + [0.0])
    keys = [k for k, q in Q.items() if q >= min_quality]
    marg = {k: margin(*k) for k in keys}
    distinct = [k for k in keys if marg[k] >= min_margin]
    pa = {k: anchor(A[k[0]]) for k in keys}
    pb = {k: anchor(B[k[1]]) for k in keys}
    hyps = []
    for i, k1 in enumerate(distinct):  # two distinctive pairs: turn from the line between them
        for k2 in distinct[i + 1:]:
            if k1[0] == k2[0] or k1[1] == k2[1]:
                continue
            da, db = pa[k2] - pa[k1], pb[k2] - pb[k1]
            if np.linalg.norm(da) < min_spread or abs(np.linalg.norm(da) - np.linalg.norm(db)) > agree:
                continue
            th = math.atan2(da[1], da[0]) - math.atan2(db[1], db[0])
            hyps.append((th, (pa[k1] + pa[k2]) / 2, (pb[k1] + pb[k2]) / 2))
    for k in distinct:  # one distinctive pair with a reliable outline: its own heading, both ways round
        ha, hb = heading(A[k[0]]), heading(B[k[1]])
        if ha is not None and hb is not None:
            for flip in (0.0, math.pi):
                hyps.append((ha - hb + flip, pa[k], pb[k]))
    best = (0, [], None)
    for th, ca, cb in hyps:
            R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            t = ca - cb @ R.T
            inl, usedA, usedB = [], set(), set()
            for k in sorted(keys, key=lambda k: -Q[k]):
                if k[0] in usedA or k[1] in usedB:
                    continue
                if np.linalg.norm(pb[k] @ R.T + t - pa[k]) < agree:
                    inl.append(k); usedA.add(k[0]); usedB.add(k[1])
            score = sum(max(marg[k], 0) + 0.05 for k in inl)
            if score > best[0]:
                best = (score, inl, (R, t))
    return best[1], best[2], [dict(A=k[0], B=k[1], quality=round(Q[k], 2), margin=round(marg[k], 2)) for k in distinct]
