# PREVIEW before making 3D models: every object of a map is a see-through box of its measured size with a "card"
# inside it showing the object's own photo crop; the card always turns to face the camera (only about the vertical
# axis), so each object reads as a flat picture of itself standing where it will go. Approve this, then make models.
# Usage: blender -b --factory-startup -P billboard_preview.py -- <map.json> <out_folder> [orbit_seconds]
#   map.json: a scene map (detect_objects.py / fuse_photos.py). Each object needs center_m, width_m, depth_m, height_m,
#   yaw (yaw_from_appearance or yaw_deg) and a background-free crop: "crop" (relative to the map's folder or to
#   "source_folder" of that object, or absolute).
# Writes: preview.blend, cards_over_<photo>.png (photo | cards from that photo's camera; every photo of a merged map), cards_view.png (bird's-eye),
#         cards_top.png, orbit_frames/f_####.png (encode with ffmpeg) when orbit_seconds > 0
import bpy, colorsys, json, math, os, sys
import numpy as np
from mathutils import Matrix, Vector

a = sys.argv[sys.argv.index("--") + 1:]
MAP, OUT = a[0], a[1]
ORBIT = float(a[2]) if len(a) > 2 else 0.0
os.makedirs(OUT, exist_ok=True)
FOLDER = os.path.dirname(os.path.abspath(MAP))
m = json.load(open(MAP, encoding="utf-8"))
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene


def material(name, rgb, alpha=1.0):
    mt = bpy.data.materials.new(name); mt.use_nodes = True
    b = mt.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1); b.inputs["Roughness"].default_value = 0.9
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha
        if hasattr(mt, "surface_render_method"): mt.surface_render_method = "BLENDED"
    return mt


def card_material(name, path):
    """Unlit picture with its own transparency (the crop's alpha)."""
    mt = bpy.data.materials.new(name); mt.use_nodes = True
    nt = mt.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tx = nt.nodes.new("ShaderNodeTexImage"); tx.image = bpy.data.images.load(path); tx.extension = "CLIP"
    em = nt.nodes.new("ShaderNodeEmission"); tr = nt.nodes.new("ShaderNodeBsdfTransparent"); mx = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(tx.outputs["Color"], em.inputs["Color"])
    nt.links.new(tx.outputs["Alpha"], mx.inputs["Fac"])
    nt.links.new(tr.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs["Surface"])
    if hasattr(mt, "surface_render_method"): mt.surface_render_method = "DITHERED"
    return mt


def trimmed_crop(o):
    """The object's crop cut to its visible pixels; returns (png path, width/height)."""
    p = o.get("crop", "")
    for base in (o.get("source_folder"), FOLDER):
        if p and not os.path.isabs(p) and base and os.path.exists(os.path.join(base, p)):
            p = os.path.join(base, p); break
    if not p or not os.path.exists(p):
        return None, 1.0
    im = bpy.data.images.load(p)
    w, h = im.size
    px = np.array(im.pixels[:]).reshape(h, w, 4)
    ys, xs = np.where(px[..., 3] > 0.1)
    if len(xs) == 0:
        return None, 1.0
    cut = px[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    hh, ww = cut.shape[:2]
    t = bpy.data.images.new(f"card_{o['id']}", ww, hh, alpha=True); t.pixels = cut.ravel()
    q = os.path.join(OUT, "cards", f"object_{o['id']}.png"); os.makedirs(os.path.dirname(q), exist_ok=True)
    t.filepath_raw = q; t.file_format = "PNG"; t.save()
    return q, ww / hh


def wire_box(name, c, g, w, d, h, z0, mat):
    """Thin edges of the box (12 bars)."""
    R = Matrix.Rotation(g, 4, "Z")
    pts = [Vector((sx * w / 2, sy * d / 2, sz * h)) for sz in (0, 1) for sy in (-1, 1) for sx in (-1, 1)]
    pts = [Vector((c.x, c.y, z0)) + (R @ p) for p in pts]
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    objs = []
    for i, j in edges:
        v = pts[j] - pts[i]
        bpy.ops.mesh.primitive_cylinder_add(radius=0.008, depth=v.length, vertices=6, location=(pts[i] + pts[j]) / 2)
        e = bpy.context.object; e.rotation_euler = v.to_track_quat("Z", "Y").to_euler(); e.data.materials.append(mat); objs.append(e)
    bpy.ops.object.select_all(action="DESELECT")
    for e in objs: e.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]; bpy.ops.object.join()
    objs[0].name = name
    return objs[0]


# camera that every card turns to (moved for each render)
cd = bpy.data.cameras.new("view_cam"); cam = bpy.data.objects.new("view_cam", cd); sc.collection.objects.link(cam); sc.camera = cam

objs = m["objects"]
n = max(1, len(objs))
cards, report = [], []
for k, o in enumerate(objs):
    rgb = colorsys.hsv_to_rgb(k / n, 0.7, 0.9)
    g = math.radians(o.get("yaw_from_appearance", o["yaw_deg"]))
    w, d, h = max(o["width_m"], 0.05), max(o["depth_m"], 0.05), max(o["height_m"], 0.05)
    z0 = o.get("base_z_m", 0.0)
    p = Vector((o["center_m"][0], o["center_m"][1], 0))
    c = p + p.normalized() * (d / 2) if "body_center_m" not in o else Vector((*o["body_center_m"], 0))
    box = wire_box(f"{o['id']:02d}_{o['name']}_box", c, g, w, d, h, z0, material(f"edge_{k}", rgb))
    png, ratio = trimmed_crop(o)
    if png is None:
        report.append(dict(id=o["id"], name=o["name"], card=False)); continue
    ch = o.get("card_height_m", h); cw = ch * ratio  # card as tall as what the photo saw, picture not stretched
    cz = o.get("card_base_z_m", z0)
    me = bpy.data.meshes.new(f"card_{o['id']}")
    me.from_pydata([(-cw / 2, 0, 0), (cw / 2, 0, 0), (cw / 2, 0, ch), (-cw / 2, 0, ch)], [], [(0, 1, 2, 3)])
    me.uv_layers.new(name="UV")
    for li, uv in zip(me.loops, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        me.uv_layers["UV"].data[li.index].uv = uv
    card = bpy.data.objects.new(f"{o['id']:02d}_{o['name']}_card", me); sc.collection.objects.link(card)
    card.location = (c.x, c.y, cz); card.data.materials.append(card_material(f"card_{o['id']}", png))
    cn = card.constraints.new("LOCKED_TRACK"); cn.target = cam; cn.track_axis = "TRACK_NEGATIVE_Y"; cn.lock_axis = "LOCK_Z"
    cards.append(card)
    report.append(dict(id=o["id"], name=o["name"], card=True, size_m=[round(w, 2), round(d, 2), round(h, 2)],
                       card_width_m=round(cw, 2), xy_m=[round(c.x, 2), round(c.y, 2)], base_z_m=z0))

# floor with a 1 m grid, and the room walls when the map has them
pts = np.array([r["xy_m"] for r in report if "xy_m" in r] + [[0, 0]])
lo, hi = np.percentile(pts, 5, 0) - 1.0, np.percentile(pts, 95, 0) + 1.0  # a far stray object must not shrink the view
ctr = (lo + hi) / 2
bpy.ops.mesh.primitive_plane_add(size=1, location=(ctr[0], ctr[1], 0)); fl = bpy.context.object
fl.scale = (hi[0] - lo[0] + 4, hi[1] - lo[1] + 4, 1); fl.data.materials.append(material("floor", (0.82, 0.82, 0.8)))
gm = material("grid", (0.55, 0.55, 0.55))
for x in np.arange(math.floor(lo[0]), hi[0] + 1, 1.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, ctr[1], 0.002)); r = bpy.context.object; r.scale = (0.01, hi[1] - lo[1] + 2, 0.001); r.data.materials.append(gm)
for y in np.arange(math.floor(lo[1]), hi[1] + 1, 1.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(ctr[0], y, 0.002)); r = bpy.context.object; r.scale = (hi[0] - lo[0] + 2, 0.01, 0.001); r.data.materials.append(gm)
wm = material("wall", (0.9, 0.9, 0.88), 0.25)
for wl in m.get("room", {}).get("walls", []):
    s, e = Vector((*wl["start_m"], 0)), Vector((*wl["end_m"], 0)); L = (e - s).length
    bpy.ops.mesh.primitive_cube_add(size=1, location=(s + e) / 2 + Vector((0, 0, wl["height_m"] / 2)))
    r = bpy.context.object; r.scale = (L, 0.03, wl["height_m"]); r.rotation_euler = (0, 0, math.atan2(e.y - s.y, e.x - s.x)); r.data.materials.append(wm)

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sun.data.energy = 3.0
sun.rotation_euler = (math.radians(35), 0, math.radians(-30)); sc.collection.objects.link(sun)
sc.world = bpy.data.worlds.new("world"); sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.95, 0.95, 0.95, 1)
sc.render.engine = "BLENDER_EEVEE"
sc.render.image_settings.file_format = "PNG"


def shoot(path, loc, look, lens=24, res=(1600, 1000), ortho=None):
    cam.location = loc
    cam.rotation_euler = (Vector(look) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    cd.type = "ORTHO" if ortho else "PERSP"
    if ortho: cd.ortho_scale = ortho
    cd.lens = lens
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.filepath = path; bpy.ops.render.render(write_still=True)


# 1) from each photo's own camera, over that photo: each card should cover its object
def photo_overlay(pm, folder, tag, place=Matrix.Identity(4), h_cam=None):
    ex, ey, ez = (Vector(v) for v in pm["cv_axes_in_floor"])
    W, H = pm["W"], pm["H"]
    cd.type = "PERSP"; cd.sensor_fit = "HORIZONTAL" if W >= H else "VERTICAL"; cd.sensor_width = cd.sensor_height = 36.0
    cd.lens = pm["fx_px"] / max(W, H) * 36.0
    cam.matrix_world = place @ Matrix.Translation((0, 0, h_cam or pm["camera_height_m"])) @ Matrix((ex, -ey, -ez)).transposed().to_4x4()
    bpy.context.view_layer.update()
    fl.hide_render = True
    sc.render.film_transparent = True; sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.render.filepath = os.path.join(OUT, f"_cards_{tag}.png"); bpy.ops.render.render(write_still=True)
    ph = np.array(bpy.data.images.load(os.path.join(folder, "photo.png")).pixels[:]).reshape(H, W, 4)
    r3 = np.array(bpy.data.images.load(sc.render.filepath).pixels[:]).reshape(H, W, 4)
    al = r3[..., 3:4] * 0.85
    ov = ph.copy(); ov[..., :3] = r3[..., :3] * al + ph[..., :3] * 0.45 * (1 - al)  # photo darkened so the cards stand out
    both = np.concatenate([ph, ov], 1)  # photo | cards over it
    im = bpy.data.images.new("ov_" + tag, 2 * W, H, alpha=True); im.pixels = both.ravel()
    im.filepath_raw = os.path.join(OUT, f"cards_over_{tag}.png"); im.file_format = "PNG"; im.save()
    fl.hide_render = False; sc.render.film_transparent = False


if m.get("cameras"):
    for cm in m["cameras"]:
        pm = json.load(open(os.path.join(cm["folder"], "map.json"), encoding="utf-8"))
        place = Matrix.Translation((*cm["xy_m"], 0)) @ Matrix.Rotation(math.radians(cm["turn_deg"]), 4, "Z")
        photo_overlay(pm, cm["folder"], cm["photo"], place, pm["camera_height_m"] * cm.get("shrink", 1.0))
elif "cv_axes_in_floor" in m and os.path.exists(os.path.join(FOLDER, "photo.png")):
    photo_overlay(m, FOLDER, "photo")

span = float(max(hi - lo))
shoot(os.path.join(OUT, "cards_view.png"), (ctr[0] + span * 0.3, lo[1] - span * 0.25, span * 0.5), (ctr[0], ctr[1], 0.5), lens=20)
shoot(os.path.join(OUT, "cards_top.png"), (ctr[0], ctr[1], 15), (ctr[0], ctr[1] + 1e-4, 0), res=(1200, 1200), ortho=span)

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "preview.blend"))
json.dump(report, open(os.path.join(OUT, "cards.json"), "w", encoding="utf-8"), indent=1)

# 2) orbit: the camera circles the room at eye-over height; every card keeps turning to face it
if ORBIT > 0:
    fd = os.path.join(OUT, "orbit_frames"); os.makedirs(fd, exist_ok=True)
    N = int(ORBIT * 24); R = span * 0.55
    sc.render.resolution_x, sc.render.resolution_y = 1280, 720; cd.type = "PERSP"; cd.lens = 22
    for f in range(N):
        t = 2 * math.pi * f / N
        loc = (ctr[0] + R * math.sin(t), ctr[1] - R * math.cos(t), 2.6)
        cam.location = loc
        cam.rotation_euler = (Vector((ctr[0], ctr[1], 0.6)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        sc.render.filepath = os.path.join(fd, f"f_{f + 1:04d}.png"); bpy.ops.render.render(write_still=True)
print("DONE", sum(r["card"] for r in report), "cards of", len(report))
