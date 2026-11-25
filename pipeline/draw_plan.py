# Top-down plan, for measuring: each object as a rectangle with its size, an arrow towards where it faces,
# distances to its 2 nearest neighbours, a 50 cm grid and a 1 m ruler. It also puts the number of each
# object on symbolic_overlay.png when _symbolic_overlay_raw.png exists.
# Usage: python -I draw_plan.py <map_folder>
# Writes: plan.png (and symbolic_overlay.png with numbers)
import colorsys, json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size, bold=True):
    for f in (("arialbd.ttf",) if bold else ()) + ("arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def color(k, n):
    return tuple(int(v * 255) for v in colorsys.hsv_to_rgb(k / max(1, n), 0.65, 0.95))


def footprint(o):
    """Corners of the rectangle on the floor (same convention as build_symbolic: the measured part is the visible face)."""
    g = math.radians(o.get("yaw_from_appearance", o["yaw_deg"]))
    a, f = max(o["width_m"], 0.05), max(o["depth_m"], 0.05)
    front = np.array([math.sin(g), -math.cos(g)])
    side = np.array([math.cos(g), math.sin(g)])
    p = np.array(o["center_m"][:2], float)
    c = p + p / max(np.linalg.norm(p), 1e-9) * f / 2  # measured = visible face; the body extends away from the camera
    corners = [c + side * sa * a / 2 + front * sf * f / 2 for sa, sf in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return c, corners, front, g


def arrow(d, p0, p1, col, w):
    d.line([p0, p1], fill=col, width=w)
    v = np.array(p1, float) - np.array(p0, float)
    L = np.linalg.norm(v)
    if L < 1:
        return
    v /= L
    n = np.array([-v[1], v[0]])
    t = 4 * w
    d.polygon([tuple(p1), tuple(np.array(p1) - v * t * 2 + n * t), tuple(np.array(p1) - v * t * 2 - n * t)], fill=col)


def main():
    folder = sys.argv[1]
    scene_map = json.load(open(os.path.join(folder, "map.json"), encoding="utf-8"))
    objs = scene_map["objects"]
    n = len(objs)
    fp = {o["id"]: footprint(o) for o in objs}

    # framing: all footprints, their arrows and the camera
    pts = [np.zeros(2)]
    for o in objs:
        c, corners, fr, _ = fp[o["id"]]
        pts += corners + [np.array(o["center_m"][:2]) + fr * max(0.4, 0.6 * max(o["width_m"], o["depth_m"]))]
    room = scene_map.get("room") or {}
    walls = room.get("walls", [])
    for w in walls:
        pts += [np.array(w["start_m"]), np.array(w["end_m"])]
    pts = np.array(pts)
    lo, hi = pts.min(0) - 0.6, pts.max(0) + 0.6
    S, PANEL = 1100, 420
    scale_px = S / max(hi - lo)
    c0 = (lo + hi) / 2
    px = lambda p: (S / 2 + (p[0] - c0[0]) * scale_px, S / 2 - (p[1] - c0[1]) * scale_px)

    im = Image.new("RGB", (S + PANEL, S), (252, 252, 250))
    d = ImageDraw.Draw(im, "RGBA")
    # grid: 50 cm (10 cm if the scene is small); every metre stronger
    step = 0.1 if max(hi - lo) < 3 else 0.5
    for k in np.arange(math.floor(lo.min()) - 1, math.ceil(hi.max()) + 1.01, step):
        tone = (215, 215, 215) if abs(k - round(k)) < 1e-6 else (236, 236, 236)
        d.line([px((k, lo[1] - 5)), px((k, hi[1] + 5))], fill=tone)
        d.line([px((lo[0] - 5, k)), px((hi[0] + 5, k))], fill=tone)
    d.rectangle([S, 0, S + PANEL, S], fill=(255, 255, 255))
    labels = []

    # room: visible floor, walls (thick dark lines with length) and object -> wall distances (dashed)
    if room.get("floor", {}).get("outline_m"):
        d.polygon([px(p) for p in room["floor"]["outline_m"]], fill=(120, 200, 140, 45))
    f_wall = font(16)
    wall_labels = []
    def dashed(p0, p1, col, width, gap_col=(252, 252, 250)):
        P0, P1 = np.array(px(p0)), np.array(px(p1)); L = max(np.linalg.norm(P1 - P0), 1e-9)
        d.line([tuple(P0), tuple(P1)], fill=gap_col, width=width)
        for t in np.arange(0, L, 14):
            d.line([tuple(P0 + (P1 - P0) * t / L), tuple(P0 + (P1 - P0) * min(t + 7, L) / L)], fill=col, width=width)

    WALL_INK = {"photo": (45, 45, 55), "photo + SpatialLM": (45, 45, 55), "SpatialLM": (30, 110, 230)}
    for w in walls:
        ink = WALL_INK.get(w.get("source", "photo"), (45, 45, 55))
        if w.get("source") == "SpatialLM":  # not seen in the photo at all: dashed
            dashed(w["start_m"], w["end_m"], ink, 6)
        else:
            d.line([px(w["start_m"]), px(w["end_m"])], fill=ink, width=7)
            for p0, p1 in w.get("assumed_parts_m", []):  # the stretch not seen in the photo
                dashed(p0, p1, ink, 5)
        if w["length_m"] < 0.3:
            continue
        side = np.array(w["room_side"], float)
        mid = (np.array(w["start_m"]) + np.array(w["end_m"])) / 2 - side * 0.28
        tag = " *" if w.get("confirmed_by") else ""
        wall_labels.append((px(mid), f"{w['id']}  {w['length_m']:.2f} m{tag}", ink))
    by_id = {w["id"]: w for w in walls}
    for op in room.get("openings", []):  # windows light blue, doors orange, drawn on their wall
        w = by_id.get(op["wall"])
        if not w:
            continue
        a, b = np.array(w["start_m"]), np.array(w["end_m"]); v = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        t = (np.array(op["center_m"]) - a) @ v
        c = a + v * t; h = v * op["width_m"] / 2
        col = (120, 200, 245) if op["kind"] == "window" else (230, 120, 20)
        d.line([px(c - h), px(c + h)], fill=col, width=4 if op["kind"] == "window" else 9)
        labels.append((*px(c - np.array(w["room_side"]) * 0.55),
                       f"{op['kind']} {op['width_m']:.2f} m, {op['bottom_m']:.2f}-{op['bottom_m'] + op['height_m']:.2f} m", col))
    wall_ends = {w["id"]: w for w in walls}
    for o in objs:
        for wd in o.get("wall_distances", []):
            A, B = np.array(px(wd["from_m"])), np.array(px(wd["to_m"]))
            L = np.linalg.norm(B - A)
            for t in np.arange(0, L, 10):  # dashed
                d.line([tuple(A + (B - A) * t / max(L, 1e-9)), tuple(A + (B - A) * min(t + 5, L) / max(L, 1e-9))],
                       fill=(200, 90, 20), width=2)
            if L > 1:
                labels.append(((A[0] + B[0]) / 2, (A[1] + B[1]) / 2, f"{wd['distance_m']:.2f} m", (200, 90, 20)))

    # distances: 2 nearest neighbours of each object
    dist = {}
    for p in scene_map["distances"]:
        dist[(p["a"], p["b"])] = dist[(p["b"], p["a"])] = p["distance_m"]
    pairs = set()
    for o in objs:
        near = sorted((dd, j) for (a, j), dd in dist.items() if a == o["id"])[:2]
        pairs |= {tuple(sorted((o["id"], j))) for _, j in near}
    cen = {o["id"]: o["center_m"][:2] for o in objs}
    f_dist = font(17)
    for i, j in sorted(pairs):
        A, B = px(cen[i]), px(cen[j])
        d.line([A, B], fill=(40, 70, 160, 200), width=2)
        labels.append(((A[0] + B[0]) / 2, (A[1] + B[1]) / 2, f"{dist[(i, j)]:.2f} m"))

    # objects
    f_num = font(26)
    for k, o in enumerate(objs):
        col = color(k, n)
        c, corners, fr, g = fp[o["id"]]
        d.polygon([px(p) for p in corners], fill=col + (170,), outline=(40, 40, 40), width=2)
        p0 = np.array(o["center_m"][:2])
        L = max(0.4, 0.6 * max(o["width_m"], o["depth_m"]))
        arrow(d, px(p0), px(p0 + fr * L), (20, 20, 20), 3)
        d.ellipse([px(p0)[0] - 4, px(p0)[1] - 4, px(p0)[0] + 4, px(p0)[1] + 4], fill=(40, 70, 160))
    for k, o in enumerate(objs):  # numbers on top of everything
        x, y = px(fp[o["id"]][0])
        r = 17
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255), outline=(20, 20, 20), width=2)
        d.text((x, y), str(o["id"]), font=f_num, fill=(20, 20, 20), anchor="mm")
    taken = [(px(fp[o["id"]][0])[0] - 19, px(fp[o["id"]][0])[1] - 19, px(fp[o["id"]][0])[0] + 19, px(fp[o["id"]][0])[1] + 19)
             for o in objs]
    overlaps = lambda b: any(b[0] < q[2] and q[0] < b[2] and b[1] < q[3] and q[1] < b[3] for q in taken)
    for (x, y), t, ink in wall_labels:
        bb = d.textbbox((x, y), t, font=f_wall, anchor="mm")
        d.rounded_rectangle([bb[0] - 4, bb[1] - 3, bb[2] + 4, bb[3] + 3], 4, fill=ink)
        d.text((x, y), t, font=f_wall, fill=(255, 255, 255), anchor="mm")
        taken.append((bb[0] - 4, bb[1] - 3, bb[2] + 4, bb[3] + 3))
    for lab in labels:  # move the label if it covers a number or another label
        x0, y0, t = lab[:3]
        ink = lab[3] if len(lab) > 3 else (40, 70, 160)
        for dx, dy in [(0, 0), (0, 20), (0, -20), (30, 0), (-30, 0), (0, 40), (0, -40), (40, 25), (-40, -25)]:
            x, y = x0 + dx, y0 + dy
            bb = d.textbbox((x, y), t, font=f_dist, anchor="mm")
            bb = (bb[0] - 5, bb[1] - 4, bb[2] + 5, bb[3] + 4)
            if not overlaps(bb):
                break
        taken.append(bb)
        bb = d.textbbox((x, y), t, font=f_dist, anchor="mm")
        d.rounded_rectangle([bb[0] - 4, bb[1] - 3, bb[2] + 4, bb[3] + 3], 4, fill=(255, 255, 255, 230), outline=ink)
        d.text((x, y), t, font=f_dist, fill=ink, anchor="mm")

    # camera
    x, y = px((0, 0))
    ey = np.array(scene_map["cv_axes_in_floor"][2][:2], float)  # where the camera looks (on the floor)
    if np.linalg.norm(ey) < 1e-6:
        ey = np.array([0, 1.0])
    ey /= np.linalg.norm(ey)
    arrow(d, (x, y), px(ey * 0.6), (30, 80, 200), 3)
    d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(30, 80, 200))
    d.text((x + 14, y + 4), f"camera ({scene_map['camera_height_m']:.2f} m high)", font=font(16), fill=(30, 80, 200), anchor="lm")
    fb = scene_map.get("fused_from")
    if fb:  # second photo of a fused map
        c2 = np.array(fb["camera_B_m"], float)
        look = np.array(fb.get("camera_B_look", [0, 0]), float)
        x2, y2 = px(c2)
        if np.linalg.norm(look) > 1e-6:
            arrow(d, (x2, y2), px(c2 + look / np.linalg.norm(look) * 0.6), (200, 60, 160), 3)
        d.ellipse([x2 - 9, y2 - 9, x2 + 9, y2 + 9], fill=(200, 60, 160))
        d.text((x2 + 14, y2 + 4), f"camera 2 ({fb['B']})", font=font(16), fill=(200, 60, 160), anchor="lm")

    # 1 m ruler
    rx, ry = 30, S - 40
    d.rectangle([rx, ry, rx + scale_px, ry + 8], fill=(20, 20, 20))
    d.text((rx + scale_px / 2, ry - 6), "1 m", font=font(18), fill=(20, 20, 20), anchor="md")
    d.text((20, 16), f"Top view  |  grid {step*100:.0f} cm  |  arrow = facing direction", font=font(18), fill=(20, 20, 20))

    # side panel with the list
    y = 20
    d.text((S + 20, y), f"{n} objects", font=font(24), fill=(20, 20, 20)); y += 40
    scale_txt = (f"scale corrected x{scene_map['scale_applied']['factor']} ({scene_map['scale_applied'].get('source', 'references')})"
                 if "scale_applied" in scene_map else "scale not corrected")
    d.text((S + 20, y), f"lens {scene_map['fov_deg']:.0f} degrees ({scene_map['fov_source']})", font=font(15, False), fill=(90, 90, 90)); y += 20
    d.text((S + 20, y), scale_txt, font=font(15, False), fill=(90, 90, 90)); y += 20
    if room:
        ceil = room.get("ceiling_height_m")
        d.text((S + 20, y), f"ceiling {ceil:.2f} m high" if ceil else "ceiling not seen", font=font(15, False), fill=(90, 90, 90)); y += 20
        d.text((S + 20, y), f"{len(walls)} wall pieces, visible floor {room['floor']['visible_area_m2']} m2",
               font=font(15, False), fill=(90, 90, 90)); y += 20
        if room.get("merged_with"):
            d.text((S + 20, y), "walls: black = photo (* = SpatialLM agrees),", font=font(15, False), fill=(90, 90, 90)); y += 18
            d.text((S + 20, y), "blue dashed = SpatialLM only, dashed = not seen", font=font(15, False), fill=(90, 90, 90)); y += 20
        for sz in room.get("size_m", [])[:2]:
            d.text((S + 20, y), f"room {sz['distance_m']:.2f} m between {sz['between'][0]} and {sz['between'][1]}",
                   font=font(15, False), fill=(90, 90, 90)); y += 20
    y += 10
    for k, o in enumerate(objs):
        col = color(k, n)
        d.rectangle([S + 20, y + 2, S + 38, y + 20], fill=col, outline=(40, 40, 40))
        d.text((S + 46, y), f"{o['id']}. {o['name'][:34]}", font=font(17), fill=(20, 20, 20))
        y += 22
        d.text((S + 46, y), f"{o['width_m']:.2f} x {o['depth_m']:.2f} x {o['height_m']:.2f} m (width x depth x height)",
               font=font(14, False), fill=(70, 70, 70))
        y += 18
        d.text((S + 46, y), f"{o['camera_distance_m']:.2f} m from the camera   faces {o.get('yaw_from_appearance', o['yaw_deg']):.0f} degrees",
               font=font(14, False), fill=(70, 70, 70))
        y += 28
        if y > S - 60:
            d.text((S + 20, y), "... (the rest is in map.json)", font=font(14, False), fill=(70, 70, 70))
            break
    im.save(os.path.join(folder, "plan.png"))

    # numbers on the overlay over the photo
    path = os.path.join(folder, "_symbolic_overlay_raw.png")
    if os.path.exists(path):
        f = Image.open(path).convert("RGB")
        path = os.path.join(folder, "symbolic_overlay.png")
        df = ImageDraw.Draw(f)
        s = max(f.size) / 1280
        fn = font(int(30 * s))
        r = int(20 * s)
        for o in objs:
            x0, y0, x1, y1 = o["image_box"]
            x, yy = (x0 + x1) / 2, max(r + 2, min(f.size[1] - r - 2, (y0 + y1) / 2))
            df.ellipse([x - r, yy - r, x + r, yy + r], fill=(255, 255, 255), outline=(20, 20, 20), width=max(1, int(2 * s)))
            df.text((x, yy), str(o["id"]), font=fn, fill=(20, 20, 20), anchor="mm")
        f.save(path)
    print("plan done")


if __name__ == "__main__":
    main()
