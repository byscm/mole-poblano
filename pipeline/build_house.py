# Builds a space seen from SEVERAL photos (merge_photos / inventory) in Blender: every object's model scaled to its
# box, turned and placed; a floor under everything, the walls two photos agree on, a flat ceiling; each photo's camera.
# Renders the models over every photo, from above and in perspective, an orbit, and exports one GLB.
# Usage: blender -b --factory-startup -P build_house.py -- <inventory_map.json> <models_folder> <out_folder> [shell.json] [orbit_s]
#   inventory_map.json: objects {name, kind?, body_center_m, yaw_deg, width_m, depth_m, height_m, base_z_m} + cameras
#   models_folder/<kind>.glb, kind = name without a trailing _<n>
#   shell.json (optional): {"walls": [[[x0,y0],[x1,y1]], ...], "ceiling_m": 2.6, "floor_hex": "...", "ceiling_hex": "..."}
import bpy, json, math, os, re, sys
import numpy as np
from mathutils import Matrix, Vector
from mathutils.geometry import convex_hull_2d

a = sys.argv[sys.argv.index("--") + 1:]
MAP, MODELS, OUT = a[:3]
SHELL = json.load(open(a[3], encoding="utf-8")) if len(a) > 3 and a[3] != "-" else {}
ORBIT = float(a[4]) if len(a) > 4 else 0
m = json.load(open(MAP, encoding="utf-8"))
os.makedirs(OUT, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene


def hexrgb(h):
    return tuple((int(h[i:i + 2], 16) / 255) ** 2.2 for i in (0, 2, 4))


def material(name, rgb, rough=0.6, metal=0.0):
    mt = bpy.data.materials.new(name); mt.use_nodes = True
    b = mt.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1); b.inputs["Roughness"].default_value = rough; b.inputs["Metallic"].default_value = metal
    return mt


def vertex_colour_material(x):
    if x.data.color_attributes and not x.data.materials:
        mt = bpy.data.materials.new(x.name); mt.use_nodes = True
        ca = mt.node_tree.nodes.new("ShaderNodeVertexColor"); ca.layer_name = x.data.color_attributes[0].name
        mt.node_tree.links.new(ca.outputs[0], mt.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
        x.data.materials.append(mt)


def place_model(o, path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [x for x in bpy.data.objects if x not in before]
    meshes = [x for x in new if x.type == "MESH"]
    for x in meshes:  # bake the glTF transforms into the vertices
        mw = x.matrix_world.copy(); x.parent = None; x.data.transform(mw); x.matrix_world = Matrix.Identity(4)
    for x in new:
        if x.type != "MESH": bpy.data.objects.remove(x)
    vs = np.array([v.co[:] for x in meshes for v in x.data.vertices])
    mn, mx = vs.min(0), vs.max(0)
    w, d, h = o["width_m"], o["depth_m"], o["height_m"]
    # models face -Y; long furniture is turned so its long side runs along X (= the box's width axis)
    align = math.pi / 2 if (w >= d) != ((mx[0] - mn[0]) >= (mx[1] - mn[1])) and abs(w - d) > 0.15 * max(w, d) else 0.0
    A = Matrix.Rotation(align, 4, "Z") @ Matrix.Translation((-(mn[0] + mx[0]) / 2, -(mn[1] + mx[1]) / 2, -mn[2]))
    ext = np.abs(np.array(A.to_3x3()) @ (mx - mn))
    S = Matrix.Diagonal((w / ext[0], d / ext[1], h / ext[2], 1))
    c = o["body_center_m"]
    T = Matrix.Translation((c[0], c[1], o.get("base_z_m", 0.0))) @ Matrix.Rotation(math.radians(o["yaw_deg"]), 4, "Z") @ S @ A
    for x in meshes:
        x.data.transform(T); vertex_colour_material(x)
    if len(meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for x in meshes: x.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]; bpy.ops.object.join()
    meshes[0].name = "obj_" + o["name"]
    return meshes[0]


def image_plane(o, img):
    """Flat things (a table runner): a plane with the object's cut-out picture lying on its box."""
    mt = bpy.data.materials.new("img_" + o["name"]); mt.use_nodes = True; nt = mt.node_tree
    b = nt.nodes["Principled BSDF"]; tx = nt.nodes.new("ShaderNodeTexImage"); tx.image = bpy.data.images.load(img)
    nt.links.new(tx.outputs["Color"], b.inputs["Base Color"]); nt.links.new(tx.outputs["Alpha"], b.inputs["Alpha"])
    b.inputs["Roughness"].default_value = 0.9
    bpy.ops.mesh.primitive_plane_add(size=1); ob = bpy.context.object; ob.name = "obj_" + o["name"]
    c = o["body_center_m"]
    ob.scale = (o["width_m"], o["depth_m"], 1); ob.rotation_euler = (0, 0, math.radians(o["yaw_deg"]))
    ob.location = (c[0], c[1], o.get("base_z_m", 0.0) + max(o["height_m"], 0.004)); ob.data.materials.append(mt)


def glass_top(o, thick=0.012):
    """A glass slab on top of the model (TRELLIS drops clear glass: a coffee table keeps only its legs)."""
    mt = bpy.data.materials.new("glass"); mt.use_nodes = True; b = mt.node_tree.nodes["Principled BSDF"]
    b.inputs["Transmission Weight"].default_value = 1.0; b.inputs["Roughness"].default_value = 0.02; b.inputs["IOR"].default_value = 1.5
    bpy.ops.mesh.primitive_cube_add(size=1); ob = bpy.context.object; ob.name = "glass_" + o["name"]
    c = o["body_center_m"]; ob.scale = (o["width_m"], o["depth_m"], thick)
    ob.rotation_euler = (0, 0, math.radians(o["yaw_deg"])); ob.location = (c[0], c[1], o.get("base_z_m", 0.0) + o["height_m"] - thick / 2)
    ob.data.materials.append(mt)


def wall_plane(o, img):
    """Flat things on a wall (a painting, a blind): an upright plane with the cut-out picture, its front along the yaw."""
    mt = bpy.data.materials.new("img_" + o["name"]); mt.use_nodes = True; nt = mt.node_tree
    b = nt.nodes["Principled BSDF"]; tx = nt.nodes.new("ShaderNodeTexImage"); tx.image = bpy.data.images.load(img)
    nt.links.new(tx.outputs["Color"], b.inputs["Base Color"]); nt.links.new(tx.outputs["Alpha"], b.inputs["Alpha"])
    b.inputs["Roughness"].default_value = 0.8
    bpy.ops.mesh.primitive_plane_add(size=1); ob = bpy.context.object; ob.name = "obj_" + o["name"]
    c = o["body_center_m"]; g = math.radians(o["yaw_deg"])
    ob.scale = (o["width_m"], o["height_m"], 1); ob.rotation_euler = (math.pi / 2, 0, g)  # plane normal ends up along -Y turned by yaw
    ob.location = (c[0], c[1], o.get("base_z_m", 0.0) + o["height_m"] / 2); ob.data.materials.append(mt)


PART_MATS = {}


def part_material(key):
    if key in PART_MATS: return PART_MATS[key]
    if key == "glass":
        mt = material("p_glass", (0.9, 0.95, 0.95), 0.02); b = mt.node_tree.nodes["Principled BSDF"]
        b.inputs["Transmission Weight"].default_value = 1.0; b.inputs["IOR"].default_value = 1.5
    elif key == "light":  # a recessed spot: glowing disc
        mt = material("p_light", (1, 1, 1), 0.5); b = mt.node_tree.nodes["Principled BSDF"]
        b.inputs["Emission Color"].default_value = (1, 0.95, 0.85, 1); b.inputs["Emission Strength"].default_value = 30
    elif key == "mirror":
        mt = material("p_mirror", (0.92, 0.92, 0.92), 0.02, 1.0)
    elif key == "b8913a":
        mt = material("p_gold", hexrgb(key), 0.3, 1.0)
    else:
        mt = material("p_" + key, hexrgb(key), 0.25 if key == "f2f0ec" else 0.6)
    PART_MATS[key] = mt; return mt


def build_parts(path):
    """Hand-built parts (parts.json): boxes and discs with a yaw, boxes optionally pitched about their width axis."""
    for p in json.load(open(path, encoding="utf-8")):
        if p["kind"] == "box":
            bpy.ops.mesh.primitive_cube_add(size=1); ob = bpy.context.object; ob.scale = p["size"]
            ob.rotation_euler = (0, math.radians(p.get("pitch", 0.0)), 0); ob.rotation_mode = "XYZ"
            ob.rotation_euler.rotate(Matrix.Rotation(math.radians(p["yaw"]), 3, "Z"))
        else:  # disc: a thin cylinder lying on the wall, its axis along a
            bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=p["radius"], depth=p["thick"]); ob = bpy.context.object
            if p.get("axis") != "z":  # "z": lying flat (a ceiling spot), else upright on a wall
                ob.rotation_euler = (0, math.pi / 2, 0); ob.rotation_euler.rotate(Matrix.Rotation(math.radians(p["yaw"]), 3, "Z"))
        ob.location = p["center"]; ob.name = "part_" + p["name"]; ob.data.materials.append(part_material(p["mat"]))


def fix_look(ob, kind):
    """Per-kind fixes of what TRELLIS gets wrong: colour shift, see-through glass, a dark slab where glass should be."""
    t = SHELL.get("tint", {}).get(kind); gl = SHELL.get("glassy", {}).get(kind); cut = SHELL.get("cut_top", {}).get(kind)
    if cut:
        import bmesh
        bm = bmesh.new(); bm.from_mesh(ob.data)
        zs = [v.co.z for v in bm.verts]; top = max(zs); zmin = min(zs)
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z > top - cut * (top - zmin)], context="VERTS")
        bm.to_mesh(ob.data); bm.free()
    for mt in ob.data.materials:
        if not mt or not mt.use_nodes: continue
        nt = mt.node_tree; b = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if b is None: continue
        if t and b.inputs["Base Color"].links:
            src = b.inputs["Base Color"].links[0].from_socket
            hs = nt.nodes.new("ShaderNodeHueSaturation")
            hs.inputs["Hue"].default_value, hs.inputs["Saturation"].default_value, hs.inputs["Value"].default_value = t
            nt.links.new(src, hs.inputs["Color"]); nt.links.new(hs.outputs["Color"], b.inputs["Base Color"])
        if gl:
            b.inputs["Transmission Weight"].default_value = gl[0]; b.inputs["Roughness"].default_value = gl[1]
            if b.inputs["Roughness"].links: nt.links.remove(b.inputs["Roughness"].links[0])
            b.inputs["IOR"].default_value = 1.5


PLANES, GLASS = set(SHELL.get("as_image_plane", [])), set(SHELL.get("glass_top", []))
WALL_PLANES, SKIP = set(SHELL.get("wall_planes", [])), set(SHELL.get("skip_models", []))
if SHELL.get("parts"): build_parts(SHELL["parts"])
CUTS = SHELL.get("cut_folder", "")
placed, missing = [], []
for o in m["objects"]:
    kind = o.get("kind") or re.sub(r"_\d+$", "", o["name"])
    if kind in SKIP:
        placed.append(o["name"]); continue  # built by hand (parts.json)
    if kind in PLANES:
        image_plane(o, os.path.join(CUTS, kind + ".png")); placed.append(o["name"]); continue
    if kind in WALL_PLANES:
        wall_plane(o, os.path.join(CUTS, kind + ".png")); placed.append(o["name"]); continue
    if kind in GLASS:
        glass_top(o)
    p = os.path.join(MODELS, kind + ".glb")
    if not os.path.exists(p):
        missing.append(o["name"]); continue
    fix_look(place_model(o, p), kind); placed.append(o["name"])
print(len(placed), "placed; missing models:", missing or "none")

# floor: the hull of every object box and camera, plus a margin
pts = []
for o in m["objects"]:
    g = math.radians(o["yaw_deg"]); c = Vector(o["body_center_m"])
    ax, ay = Vector((math.cos(g), math.sin(g))), Vector((-math.sin(g), math.cos(g)))
    pts += [c + ax * sx * o["width_m"] / 2 + ay * sy * o["depth_m"] / 2 for sx in (-1, 1) for sy in (-1, 1)]
pts += [Vector(c["xy_m"]) for c in m.get("cameras", [])]
pts += [Vector(p) for w in SHELL.get("walls", []) for p in w]
hull = [pts[i] for i in convex_hull_2d(pts)]
cen = sum(hull, Vector((0, 0))) / len(hull)
hull = [cen + (p - cen) * (1 + 0.6 / max((p - cen).length, 0.1)) for p in hull]
if SHELL.get("outline"):  # a closed wall outline: the floor and ceiling follow it exactly
    hull = [Vector(q) for q in SHELL["outline"]]


def flat_poly(name, poly, z, mat, flip=False):
    me = bpy.data.meshes.new(name)
    vs = [(p.x, p.y, z) for p in poly]
    me.from_pydata(vs, [], [list(range(len(vs)))[::-1] if flip else list(range(len(vs)))])
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob); me.materials.append(mat)
    return ob


floor = flat_poly("floor", hull, 0.0, material("floor", hexrgb(SHELL.get("floor_hex", "1c1a19")), 0.15))
if SHELL.get("floor_texture"):  # a tiled picture of the real floor, repeated every floor_tile_m metres
    fm = floor.data.materials[0]; nt = fm.node_tree; b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping"); tx = nt.nodes.new("ShaderNodeTexImage")
    tx.image = bpy.data.images.load(SHELL["floor_texture"]); k = 1.0 / SHELL.get("floor_tile_m", 1.2)
    mp.inputs["Scale"].default_value = (k, k, k)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], tx.inputs["Vector"])
    nt.links.new(tx.outputs["Color"], b.inputs["Base Color"]); b.inputs["Roughness"].default_value = SHELL.get("floor_rough", 0.12)
H_CEIL = SHELL.get("ceiling_m", 2.6)
ceil = flat_poly("ceiling", hull, H_CEIL, material("ceiling", hexrgb(SHELL.get("ceiling_hex", "eeece6")), 0.8), flip=True)
wall_mat = material("wall", hexrgb(SHELL.get("wall_hex", "e4e2dc")), 0.8)
walls = []
for i, (p0, p1) in enumerate(SHELL.get("walls", [])):
    p0, p1 = Vector(p0), Vector(p1); L = (p1 - p0).length
    bpy.ops.mesh.primitive_cube_add(size=1)
    wo = bpy.context.object; wo.name = f"wall_{i}"
    mid = (p0 + p1) / 2
    wo.location = (mid.x, mid.y, H_CEIL / 2); wo.scale = (L, 0.1, H_CEIL); wo.rotation_euler = (0, 0, math.atan2(p1.y - p0.y, p1.x - p0.x))
    wo.data.materials.append(wall_mat); walls.append(wo)
shell = [floor, ceil] + walls

# light: soft sky + a sun, the ceiling lets light from the sky (scene is a cut-away model)
wd = bpy.data.worlds.new("w"); sc.world = wd; wd.use_nodes = True
wd.node_tree.nodes["Background"].inputs[0].default_value = (0.9, 0.9, 0.92, 1); wd.node_tree.nodes["Background"].inputs[1].default_value = 0.9
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sun.data.energy = 2.5; sun.data.angle = 0.3
sc.collection.objects.link(sun); sun.rotation_euler = (math.radians(35), 0, math.radians(30))
sc.render.engine = "CYCLES"
try:
    pr = bpy.context.preferences.addons["cycles"].preferences; pr.compute_device_type = "OPTIX"; pr.get_devices()
    for dv in pr.devices: dv.use = True
    sc.cycles.device = "GPU"
except Exception as e:
    print("GPU:", e)
sc.cycles.samples = 48; sc.cycles.use_denoising = True
cd = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cd); sc.collection.objects.link(cam); sc.camera = cam


def render(path, res):
    sc.render.resolution_x, sc.render.resolution_y = res; sc.render.filepath = path; bpy.ops.render.render(write_still=True)


def look(loc, at, lens=24, ortho=None):
    cd.type = "ORTHO" if ortho else "PERSP"
    if ortho: cd.ortho_scale = ortho
    cd.lens = lens; cd.sensor_fit = "AUTO"; cd.sensor_width = 36
    cam.location = loc; cam.rotation_euler = (Vector(at) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()


# 1) the models over every photo (from that photo's camera)
OVL = os.environ.get("OVERLAYS", "").split(",")
for cm in [c for c in m.get("cameras", []) if os.environ.get("ALL_OVERLAYS") or c["photo"].startswith("photo_") or c["photo"] in OVL]:  # the 46 extra photos only help alignment
    pm = json.load(open(os.path.join(cm["folder"], "map.json"), encoding="utf-8"))
    place = Matrix.Translation((*cm["xy_m"], 0)) @ Matrix.Rotation(math.radians(cm["turn_deg"]), 4, "Z")
    ex, ey, ez = (Vector(v) for v in pm["cv_axes_in_floor"])
    W, Hh = pm["W"], pm["H"]
    cd.type = "PERSP"; cd.sensor_fit = "HORIZONTAL" if W >= Hh else "VERTICAL"; cd.sensor_width = cd.sensor_height = 36.0
    cd.lens = pm["fx_px"] / max(W, Hh) * 36.0
    cam.matrix_world = place @ Matrix.Translation((0, 0, pm["camera_height_m"] * cm.get("shrink", 1.0))) @ Matrix((ex, -ey, -ez)).transposed().to_4x4()
    bpy.context.view_layer.update()
    for s in shell: s.hide_render = True
    sc.render.film_transparent = True
    tmp = os.path.join(OUT, f"_models_{cm['photo']}.png"); render(tmp, (W, Hh))
    ph = np.array(bpy.data.images.load(os.path.join(cm["folder"], "photo.png")).pixels[:]).reshape(Hh, W, 4)
    r3 = np.array(bpy.data.images.load(tmp).pixels[:]).reshape(Hh, W, 4)
    al = r3[..., 3:4]
    ov = ph.copy(); ov[..., :3] = r3[..., :3] * al + ph[..., :3] * 0.5 * (1 - al)
    both = np.concatenate([ph, ov], 1)
    im = bpy.data.images.new("ov", 2 * W, Hh, alpha=True); im.pixels = both.ravel()
    im.filepath_raw = os.path.join(OUT, f"models_over_{cm['photo']}.png"); im.file_format = "PNG"; im.save()
    for s in shell: s.hide_render = False
    sc.render.film_transparent = False

# 2) from above and 3) a perspective view (ceiling hidden: looking in from above)
P = np.array([[p.x, p.y] for p in hull]); lo, hi = P.min(0), P.max(0); ctr = (lo + hi) / 2; span = float(max(hi - lo))
ceil.hide_render = True
look((ctr[0], ctr[1], 15), (ctr[0], ctr[1] + 1e-4, 0), ortho=span * 1.02); render(os.path.join(OUT, "top.png"), (1400, 1400))
look((ctr[0] + span * 0.35, lo[1] - span * 0.2, span * 0.55), (ctr[0], ctr[1], 0.5), lens=20); render(os.path.join(OUT, "view.png"), (1600, 1000))
look((ctr[0] - span * 0.35, hi[1] + span * 0.2, span * 0.55), (ctr[0], ctr[1], 0.5), lens=20); render(os.path.join(OUT, "view_back.png"), (1600, 1000))
ceil.hide_render = False

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "house.blend"))
bpy.ops.object.select_all(action="DESELECT")
for o in sc.objects:
    if o.type == "MESH": o.select_set(True)
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, "house.glb"), export_format="GLB", use_selection=True)

# 4) orbit with the ceiling hidden
if ORBIT > 0:
    ceil.hide_render = True
    fd = os.path.join(OUT, "orbit_frames"); os.makedirs(fd, exist_ok=True)
    N = int(ORBIT * 24); R = span * 0.6; sc.cycles.samples = 24
    for f in range(N):
        t = 2 * math.pi * f / N
        cpos = Vector((ctr[0] + R * math.sin(t), ctr[1] - R * math.cos(t)))
        for wo in [o for o in sc.objects if o.name.startswith("wall_")]:  # dollhouse cut-away: hide the walls on the camera's side of the room
            d = Vector((wo.location.x - ctr[0], wo.location.y - ctr[1])); n = Vector((-math.sin(wo.rotation_euler.z), math.cos(wo.rotation_euler.z)))
            if d.dot(n) < 0: n = -n  # outward normal
            wo.hide_render = n.dot((cpos - Vector((wo.location.x, wo.location.y))).normalized()) > 0.25
        look((cpos.x, cpos.y, span * 0.45), (ctr[0], ctr[1], 0.4), lens=22)
        render(os.path.join(fd, f"{f:04d}.png"), (1280, 720))
json.dump(dict(placed=placed, missing=missing), open(os.path.join(OUT, "placed.json"), "w"), indent=1)
print("DONE")
