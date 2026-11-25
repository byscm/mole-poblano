# Overlays our room plan (room_layout.py) and SpatialLM's (run_spatiallm.py) on the floor points, and says how far
# apart each of our walls is from the closest SpatialLM wall.
# Usage: python -I compare_layouts.py <scene_folder>     Writes: layout_comparison.png, layout_comparison.json
import json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draw_plan import footprint, font
from room_layout import to_floor_points

OURS, THEIRS, OPENING = (35, 35, 45), (30, 110, 230), (230, 120, 20)


def seg_info(a, b):
    a, b = np.array(a, float), np.array(b, float)
    v = b - a
    return a, b, v / max(np.linalg.norm(v), 1e-9), np.linalg.norm(v)


def main():
    folder = os.path.abspath(sys.argv[1])
    m = json.load(open(os.path.join(folder, "map.json"), encoding="utf-8"))
    s = json.load(open(os.path.join(folder, "spatiallm.json")))
    ours = m.get("room", {}).get("walls", [])
    theirs = [w for w in s["walls"] if np.linalg.norm(np.subtract(w["end_m"], w["start_m"])) > 0.05]

    # how our walls match theirs: angle and perpendicular offset to the best overlapping SpatialLM wall
    report = []
    for w in ours:
        a, b, v, L = seg_info(w["start_m"], w["end_m"])
        mid = (a + b) / 2
        best = None
        for t in theirs:
            ta, tb, tv, tL = seg_info(t["start_m"], t["end_m"])
            ang = math.degrees(math.acos(min(1, abs(v @ tv))))
            if ang > 20:
                continue
            tn = np.array([-tv[1], tv[0]])
            off = abs((mid - ta) @ tn)
            along = (mid - ta) @ tv
            gap = max(0, -along, along - tL)
            score = off + gap
            if best is None or score < best[0]:
                best = (score, t["id"], round(ang, 1), round(off, 2), round(t_len := tL, 2))
        report.append(dict(ours=w["id"], ours_length_m=w["length_m"],
                           spatiallm=best[1] if best else None, angle_deg=best[2] if best else None,
                           offset_m=best[3] if best else None, spatiallm_length_m=best[4] if best else None))
        print(f"  {w['id']} ({w['length_m']:.2f} m): " + (f"closest SpatialLM wall {best[1]} ({best[4]:.2f} m), "
                                                          f"{best[2]:.0f} deg apart, {best[3]:.2f} m off" if best else "no matching SpatialLM wall"))
    json.dump(dict(walls=report, spatiallm_doors=s["doors"], spatiallm_windows=s["windows"]),
              open(os.path.join(folder, "layout_comparison.json"), "w"), indent=1)

    # drawing
    W, H = m["W"], m["H"]
    P = np.load(os.path.join(folder, "moge_points.npz"))["points"].reshape(H, W, 3).astype(np.float64)
    if np.nanmedian(P[..., 2]) < 0:
        P = P * np.array([1, -1, -1])
    Q = to_floor_points(m, P).reshape(-1, 3)
    rgb = np.array(Image.open(os.path.join(folder, "photo.png")).convert("RGB")).reshape(-1, 3)
    ok = np.isfinite(Q).all(1) & (Q[:, 2] < 2.0) & (np.linalg.norm(Q[:, :2], axis=1) < 12)
    Q, rgb = Q[ok], rgb[ok]
    pts = [np.zeros(2)] + [np.array(p) for w in ours + theirs for p in (w["start_m"], w["end_m"])]
    pts = np.array(pts)
    lo, hi = pts.min(0) - 0.8, pts.max(0) + 0.8
    S, PANEL = 1100, 430
    sc = S / max(hi - lo); c0 = (lo + hi) / 2
    px = lambda p: (S / 2 + (p[0] - c0[0]) * sc, S / 2 - (p[1] - c0[1]) * sc)
    im = Image.new("RGB", (S + PANEL, S), (250, 250, 248)); d = ImageDraw.Draw(im, "RGBA")
    for k in np.arange(math.floor(lo.min()) - 1, math.ceil(hi.max()) + 1.01, 0.5):
        tone = (222, 222, 222) if abs(k - round(k)) < 1e-6 else (238, 238, 238)
        d.line([px((k, lo[1] - 5)), px((k, hi[1] + 5))], fill=tone); d.line([px((lo[0] - 5, k)), px((hi[0] + 5, k))], fill=tone)
    order = np.argsort(Q[:, 2])  # floor points first, higher ones on top
    for q, c in zip(Q[order][::7], rgb[order][::7]):
        x, y = px(q)
        d.point((x, y), fill=tuple(int(v) for v in c) + (110,))
    for o in m["objects"]:
        _, corners, _, _ = footprint(o)
        d.polygon([px(p) for p in corners], fill=(150, 150, 150, 110), outline=(90, 90, 90))
        d.text(px(o["center_m"][:2]), o["name"], font=font(13), fill=(60, 60, 60), anchor="mm")
    for b in s["boxes"]:
        g = math.radians(b["angle_deg"]); cx, cy = b["center_m"][:2]; sx, sy = b["size_m"][:2]
        cs = [(cx + math.cos(g) * u - math.sin(g) * v, cy + math.sin(g) * u + math.cos(g) * v)
              for u, v in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))]
        d.polygon([px(p) for p in cs], outline=THEIRS + (200,), width=2)
        d.text(px((cx, cy)), b["name"], font=font(13), fill=THEIRS, anchor="mm")
    for t in theirs:
        d.line([px(t["start_m"]), px(t["end_m"])], fill=THEIRS + (220,), width=5)
    for w in ours:
        d.line([px(w["start_m"]), px(w["end_m"])], fill=OURS, width=7)
        mid = (np.array(w["start_m"]) + np.array(w["end_m"])) / 2 - np.array(w["room_side"]) * 0.3
        d.text(px(mid), w["id"], font=font(16), fill=OURS, anchor="mm")
    by_id = {t["id"]: t for t in s["walls"]}
    for kind, items in (("door", s["doors"]), ("window", s["windows"])):
        for o in items:
            t = by_id.get(o["wall"])
            if not t:
                continue
            _, _, v, _ = seg_info(t["start_m"], t["end_m"])
            c = np.array(o["center_m"][:2]); h = v * o["width_m"] / 2
            d.line([px(c - h), px(c + h)], fill=OPENING, width=11 if kind == "door" else 6)
            d.text(px(c + np.array([-v[1], v[0]]) * 0.35), kind, font=font(13), fill=OPENING, anchor="mm")
    x, y = px((0, 0)); d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(200, 40, 40))
    d.text((x + 12, y), "camera", font=font(15), fill=(200, 40, 40), anchor="lm")
    d.rectangle([30, S - 40, 30 + sc, S - 32], fill=(20, 20, 20)); d.text((30 + sc / 2, S - 46), "1 m", font=font(16), fill=(20, 20, 20), anchor="md")

    d.rectangle([S, 0, S + PANEL, S], fill=(255, 255, 255))
    y = 20
    d.text((S + 20, y), "Two room plans, one photo", font=font(22), fill=(20, 20, 20)); y += 40
    for col, t in ((OURS, "ours (SAM walls + MoGe points)"), (THEIRS, "SpatialLM (reads the point cloud)"), (OPENING, "SpatialLM doors / windows")):
        d.rectangle([S + 20, y + 4, S + 50, y + 12], fill=col); d.text((S + 60, y), t, font=font(15, False), fill=(40, 40, 40)); y += 26
    y += 14
    d.text((S + 20, y), "Our walls vs the closest SpatialLM wall", font=font(16), fill=(20, 20, 20)); y += 28
    for r in report:
        t = (f"{r['ours']} {r['ours_length_m']:.2f} m  ->  {r['spatiallm']} {r['spatiallm_length_m']:.2f} m,  "
             f"{r['angle_deg']:.0f} deg, {r['offset_m']:.2f} m off") if r["spatiallm"] else f"{r['ours']}: none"
        d.text((S + 20, y), t, font=font(14, False), fill=(40, 40, 40)); y += 22
    y += 14
    d.text((S + 20, y), f"SpatialLM: {len(theirs)} wall pieces, {len(s['doors'])} door, {len(s['windows'])} windows", font=font(14, False), fill=(40, 40, 40)); y += 22
    d.text((S + 20, y), "objects: " + ", ".join(b["name"] for b in s["boxes"]), font=font(14, False), fill=(40, 40, 40)); y += 22
    d.text((S + 20, y), f"ceiling: ours {m['room'].get('ceiling_height_m')} m, SpatialLM {theirs[0]['height_m']:.2f} m" if theirs else "",
           font=font(14, False), fill=(40, 40, 40))
    im.save(os.path.join(folder, "layout_comparison.png"))
    print("comparison done")


if __name__ == "__main__":
    main()
