# Builds the furnished room in Blender from a layout (scene_layout.py): every model scaled to its real size, turned and
# placed; walls and the visible floor from the map; the photo's camera. Renders it over the photo, from above and in
# perspective, and exports the whole room as one GLB.
# Usage: blender -b --factory-startup -P build_room.py -- <layout.json> <map_folder> <out_folder>
# Writes: room.blend, room.glb, models_only_<view>.png, over_<view>.png, side_by_side_<view>.png (photo_A + layout extra_views),
#         top.png, view.png, placed.json
import bpy, bmesh, json, math, os, sys
import numpy as np
from mathutils import Matrix, Vector

LAYOUT, MAPDIR, OUT = sys.argv[sys.argv.index("--") + 1:][:3]
lay = json.load(open(LAYOUT, encoding="utf-8"))
smap = json.load(open(os.path.join(MAPDIR, "map.json"), encoding="utf-8"))
os.makedirs(OUT, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene


def vertex_colour_material(x):
    if x.data.color_attributes and not x.data.materials:
        m = bpy.data.materials.new(x.name); m.use_nodes = True
        ca = m.node_tree.nodes.new("ShaderNodeVertexColor"); ca.layer_name = x.data.color_attributes[0].name
        m.node_tree.links.new(ca.outputs[0], m.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
        x.data.materials.append(m)


def flat_material(name, rgb, alpha=1.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1); b.inputs["Roughness"].default_value = 0.9
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha
    return m


report = []
for it in lay["items"]:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=it["model"])
    new = [x for x in bpy.data.objects if x not in before]
    meshes = [x for x in new if x.type == "MESH"]
    for x in meshes:  # bake the glTF transforms into the vertices
        mw = x.matrix_world.copy(); x.parent = None; x.data.transform(mw); x.matrix_world = Matrix.Identity(4)
    for x in new:
        if x.type != "MESH":
            bpy.data.objects.remove(x)
    vs = np.array([v.co[:] for x in meshes for v in x.data.vertices])
    mn, mx = vs.min(0), vs.max(0)
    # models face -Y; long furniture is turned so its long side runs along X (= the map's width axis)
    align = 0.0
    if it.get("size_m") and it["size_m"][0] >= it["size_m"][1] and (mx[1] - mn[1]) > (mx[0] - mn[0]):
        align = math.pi / 2
    A = Matrix.Rotation(align, 4, "Z") @ Matrix.Translation((-(mn[0] + mx[0]) / 2, -(mn[1] + mx[1]) / 2, -mn[2]))
    ext = np.abs(np.array(A.to_3x3()) @ (mx - mn))
    if it.get("size_m"):
        S = Matrix.Diagonal((it["size_m"][0] / ext[0], it["size_m"][1] / ext[1], it["size_m"][2] / ext[2], 1))
        got = it["size_m"]
    else:
        s = it["height_m"] / ext[2]
        S = Matrix.Scale(s, 4); got = [float(ext[0] * s), float(ext[1] * s), it["height_m"]]
    T = (Matrix.Translation((it["xy_m"][0], it["xy_m"][1], it["base_z_m"])) @ Matrix.Rotation(math.radians(it["yaw_deg"]), 4, "Z")
         @ S @ A)
    for x in meshes:
        x.data.transform(T); x.name = it["label"]; vertex_colour_material(x)
    if len(meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for x in meshes: x.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]; bpy.ops.object.join()
    report.append(dict(label=it["label"], size_m=[round(v, 2) for v in got], xy_m=it["xy_m"], yaw_deg=it["yaw_deg"],
                       base_z_m=it["base_z_m"], sure=it["sure"], note=it["note"]))
    print(f"{it['label']:18s} {got[0]:.2f} x {got[1]:.2f} x {got[2]:.2f} m  [{it['sure']}]")

def image_material(name, path, tile_m, rough=0.6):
    """Image texture repeated every tile_m metres in world space (box projection on object coords; the shell meshes
    are built in world coordinates with an identity transform)."""
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]; b.inputs["Roughness"].default_value = rough
    tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / tile_m, 1 / tile_m, 1 / tile_m)
    tx = nt.nodes.new("ShaderNodeTexImage"); tx.image = bpy.data.images.load(path); tx.projection = "BOX"; tx.projection_blend = 0.15
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], tx.inputs["Vector"])
    nt.links.new(tx.outputs["Color"], b.inputs["Base Color"])
    return m


def glass_material():
    m = bpy.data.materials.new("window_glass"); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.55, 0.65, 0.7, 1); b.inputs["Roughness"].default_value = 0.05
    b.inputs["Alpha"].default_value = 0.12
    if hasattr(m, "surface_render_method"):
        m.surface_render_method = "BLENDED"
    elif hasattr(m, "blend_method"):
        m.blend_method = "BLEND"
    return m


def slab(name, a, b, n, d0, d1, z0, z1, mat, shadow=True):
    """Box between the wall line a-b offset d0..d1 along its normal n, from height z0 to z1 (world coordinates)."""
    me = bpy.data.meshes.new(name); bm = bmesh.new()
    pts = [a + n * d0, b + n * d0, b + n * d1, a + n * d1]
    vs = [bm.verts.new((p.x, p.y, z)) for z in (z0, z1) for p in pts]
    for f in ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)):
        bm.faces.new([vs[i] for i in f])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(name, me); sc.collection.objects.link(o); o.data.materials.append(mat)
    if not shadow and hasattr(o, "visible_shadow"):
        o.visible_shadow = False
    return o


def hex_material(name, hx, rough=0.5, metal=0.0):
    m = flat_material(name, tuple(((int(hx[i:i + 2], 16) / 255) ** 2.2) for i in (0, 2, 4)))
    b = m.node_tree.nodes["Principled BSDF"]; b.inputs["Roughness"].default_value = rough; b.inputs["Metallic"].default_value = metal
    return m


def build_ceiling(cfg, outline, C):
    """Louvered pergola roof repeated over the whole floor outline: tilted blades every pitch_m along one direction,
    dark beams round the edge and across (cross_beams_at: points their lines pass through, across the blades)."""
    poly = np.array(outline, float)
    u = np.array(cfg["louver_dir"], float); u /= np.linalg.norm(u); n = np.array([-u[1], u[0]])
    edges = list(zip(poly, np.roll(poly, -1, 0)))

    def clip(o, d):
        """Part (t0, t1) of the line o + t d inside the convex outline."""
        t0, t1 = -1e9, 1e9
        cen = poly.mean(0)
        for p, q in edges:
            e = q - p; nn = np.array([-e[1], e[0]])
            if nn @ (cen - p) < 0: nn = -nn  # points inside
            den, num = nn @ d, nn @ (o - p)
            if abs(den) < 1e-9:
                if num < 0: return None
                continue
            t = -num / den
            if den > 0: t0 = max(t0, t)
            else: t1 = min(t1, t)
        return (t0, t1) if t1 > t0 else None

    def box(bm, a, b, side, half_w, z, half_h, tilt=0.0):
        """Box from a to b (2D), half_w wide along 'side', half_h high, cross-section turned by tilt about its length."""
        ct, st = math.cos(tilt), math.sin(tilt)
        corners = [(sw * half_w * ct - sh * half_h * st, sw * half_w * st + sh * half_h * ct) for sw, sh in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        vs = [bm.verts.new((*(p + side * cw), z + cz)) for p in (a, b) for cw, cz in corners]
        for f in ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)):
            bm.faces.new([vs[i] for i in f])

    def obj(name, bm, mat):
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
        o = bpy.data.objects.new(name, me); sc.collection.objects.link(o); o.data.materials.append(mat)
        return o

    zl = C + cfg.get("louver_above_m", 0.06)
    pitch, w, th = cfg["louver_pitch_m"], cfg["louver_width_m"], cfg.get("louver_thick_m", 0.012)
    s_all = poly @ n
    bm = bmesh.new(); count = 0
    for s in np.arange(s_all.min() + pitch / 2, s_all.max(), pitch):
        o = n * s
        seg = clip(o, u)
        if seg:
            box(bm, o + u * seg[0], o + u * seg[1], n, w / 2, zl, th / 2, math.radians(cfg.get("louver_tilt_deg", 35))); count += 1
    obj("ceiling_louvers", bm, hex_material("louver", cfg.get("louver_hex", "d8d8d2"), 0.35, 0.6))
    bw, bh = cfg.get("beam_width_m", 0.14), cfg.get("beam_height_m", 0.22)
    bm = bmesh.new(); cen = poly.mean(0)
    for p, q in edges:  # perimeter beams, just inside the outline
        e = (q - p) / np.linalg.norm(q - p); nn = np.array([-e[1], e[0]])
        if nn @ (cen - p) < 0: nn = -nn
        box(bm, p, q, nn, bw / 2, zl, bh / 2)
    for c in cfg.get("cross_beams_at", []):
        c = np.array(c, float); seg = clip(c, n)
        if seg: box(bm, c + n * seg[0], c + n * seg[1], u, bw / 2, zl, bh / 2)
    obj("ceiling_beams", bm, hex_material("beam", cfg.get("beam_hex", "2b2c2e"), 0.5))
    print(f"ceiling: {count} louvers every {pitch} m, {len(cfg.get('cross_beams_at', []))} cross beams")


shell = lay.get("shell")
if shell:
    # the room: textured floor, and on each wall a plaster parapet with a dark wood baseboard and windows above it
    floor_mat = image_material("floor", shell["floor_texture"], shell["floor_tile_m"], 0.45)
    wall_mat = image_material("wall", shell["wall_texture"], shell["wall_tile_m"], 0.85)
    base_mat = flat_material("baseboard", (0.16, 0.09, 0.05))
    frame_mat = flat_material("window_frame", (0.03, 0.03, 0.03))
    glass = glass_material()
    me = bpy.data.meshes.new("floor_room"); bm = bmesh.new()
    bm.faces.new([bm.verts.new((x, y, 0.003)) for x, y in shell["floor_outline_m"]]); bm.to_mesh(me); bm.free()
    fo = bpy.data.objects.new("floor_room", me); sc.collection.objects.link(fo); fo.data.materials.append(floor_mat)
    inner = Vector((*np.mean(np.array(shell["floor_outline_m"]), 0), 0))
    P, C, bay = shell["parapet_m"], shell["ceiling_m"], shell.get("window_bay_m", 1.2)
    for w in shell["walls"]:
        a, b = Vector((*w["start_m"], 0)), Vector((*w["end_m"], 0))
        u = (b - a).normalized(); n = Vector((-u.y, u.x, 0))
        if (inner - a).dot(n) < 0:
            n = -n  # n points into the room
        L = (b - a).length
        slab(f"wall_{w['id']}_parapet", a, b, n, -0.15, 0.0, 0.0, P, wall_mat)
        slab(f"wall_{w['id']}_baseboard", a, b, n, 0.0, 0.012, 0.0, 0.08, base_mat)
        slab(f"wall_{w['id']}_sill", a, b, n, -0.15, 0.02, P, P + 0.04, frame_mat)
        slab(f"wall_{w['id']}_head", a, b, n, -0.12, -0.04, C - 0.08, C, frame_mat)
        slab(f"wall_{w['id']}_glass", a, b, n, -0.085, -0.075, P + 0.04, C - 0.08, glass, shadow=False)
        k = max(1, round(L / bay))
        for j in range(k + 1):
            q = a + u * (L * j / k)
            slab(f"wall_{w['id']}_mullion{j}", q - u * 0.03, q + u * 0.03, n, -0.12, -0.04, P + 0.04, C - 0.08, frame_mat)
        slab(f"wall_{w['id']}_transom", a, b, n, -0.12, -0.04, 2.2, 2.25, frame_mat)
    if shell.get("ceiling"):
        build_ceiling(shell["ceiling"], shell["floor_outline_m"], C)
else:
    # walls (thin slabs on the map lines) and the floor the photos see
    wall_mat = flat_material("wall", (0.75, 0.73, 0.7))
    for w in lay["walls"]:
        a, b = Vector((*w["start_m"], 0)), Vector((*w["end_m"], 0))
        L = (b - a).length
        bpy.ops.mesh.primitive_cube_add(size=1, location=((a + b) / 2) + Vector((0, 0, w["height_m"] / 2)))
        o = bpy.context.object; o.name = "wall_" + w["id"]
        o.scale = (L, 0.06, w["height_m"]); o.rotation_euler = (0, 0, math.atan2(b.y - a.y, b.x - a.x))
        o.data.materials.append(wall_mat)
    if len(lay["floor_outline_m"]) >= 3:
        me = bpy.data.meshes.new("visible_floor"); bm = bmesh.new()
        vv = [bm.verts.new((x, y, 0.002)) for x, y in lay["floor_outline_m"]]
        bm.faces.new(vv); bm.to_mesh(me); bm.free()
        fo = bpy.data.objects.new("visible_floor", me); sc.collection.objects.link(fo)
        fo.data.materials.append(flat_material("visible_floor", (0.62, 0.6, 0.56)))
bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
ground = bpy.context.object; ground.name = "ground"; ground.data.materials.append(flat_material("ground", (0.42, 0.42, 0.4)))

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sun.data.energy = 3.5
sun.rotation_euler = (math.radians(40), 0, math.radians(-30)); sc.collection.objects.link(sun)
sc.world = bpy.data.worlds.new("world"); sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.8, 0.85, 0.95, 1)
sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.8
sc.render.engine = "CYCLES"
try:
    pr = bpy.context.preferences.addons["cycles"].preferences
    pr.compute_device_type = "OPTIX"
    pr.refresh_devices() if hasattr(pr, "refresh_devices") else pr.get_devices()
    for d in pr.devices: d.use = d.type == "OPTIX"
    sc.cycles.device = "GPU"
except Exception as e:
    print("GPU:", e)
sc.cycles.samples = 64; sc.cycles.use_denoising = True

# 1) each photo's camera (photo A, plus extra photos joined to the room), models only, over the photo
room_bits = [o for o in sc.objects if o.name.startswith(("wall_", "visible_floor", "floor_room", "ground", "ceiling_"))]


def over_photo(pmap, mapdir, tag, place=Matrix.Identity(4)):
    """Render the models from the camera of photo pmap (placed in the room by 'place') over that photo."""
    ex, ey, ez = (Vector(v) for v in pmap["cv_axes_in_floor"])
    W, H = pmap["W"], pmap["H"]
    cd = bpy.data.cameras.new("cam_" + tag); cd.sensor_fit = "HORIZONTAL" if W >= H else "VERTICAL"
    cd.sensor_width = cd.sensor_height = 36.0; cd.lens = pmap["fx_px"] / max(W, H) * 36.0
    pc = bpy.data.objects.new("cam_" + tag, cd); sc.collection.objects.link(pc)
    loc, rot, _ = (place @ Matrix.Translation((0, 0, pmap["camera_height_m"])) @ Matrix((ex, -ey, -ez)).transposed().to_4x4()).decompose()
    pc.matrix_world = Matrix.LocRotScale(loc, rot, None)  # a photo joined with a scale moves the camera, it doesn't zoom it
    for o in room_bits: o.hide_render = True
    ground.hide_render = False; ground.is_shadow_catcher = True
    sc.camera = pc; sc.render.film_transparent = True
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.render.filepath = os.path.join(OUT, f"models_only_{tag}.png"); bpy.ops.render.render(write_still=True)
    photo = bpy.data.images.load(os.path.join(mapdir, "photo.png"))
    r3 = bpy.data.images.load(sc.render.filepath)
    pf = np.array(photo.pixels[:]).reshape(H, W, 4); p3 = np.array(r3.pixels[:]).reshape(H, W, 4)
    a = p3[..., 3:4]
    over = pf.copy(); over[..., :3] = p3[..., :3] * a + pf[..., :3] * (1 - a)
    alone = p3.copy(); alone[..., :3] = p3[..., :3] * a + 0.92 * (1 - a); alone[..., 3] = 1
    for name, arr in ((f"over_{tag}.png", over), (f"side_by_side_{tag}.png", np.concatenate([pf, alone], 1))):
        h, w = arr.shape[:2]
        im = bpy.data.images.new(name, w, h, alpha=True); im.pixels = arr.ravel()
        im.filepath_raw = os.path.join(OUT, name); im.file_format = "PNG"; im.save()
    for o in room_bits: o.hide_render = False
    ground.is_shadow_catcher = False; sc.render.film_transparent = False


over_photo(smap, MAPDIR, "photo_A")
for v in lay.get("extra_views", []):
    vmap = json.load(open(os.path.join(v["map_folder"], "map.json"), encoding="utf-8"))
    place = Matrix.Translation((*v["t_m"], 0)) @ Matrix.Rotation(math.radians(v["turn_deg"]), 4, "Z") @ Matrix.Scale(v.get("scale", 1.0), 4)
    over_photo(vmap, v["map_folder"], v["name"], place)

# 2) from above (orthographic) and 3) a perspective view from behind-above camera A
pts = np.array([it["xy_m"] for it in lay["items"]] + [p for w in (shell["walls"] if shell else lay["walls"]) for p in (w["start_m"], w["end_m"])])
lo, hi = pts.min(0) - 1.0, pts.max(0) + 1.0
ctr = (lo + hi) / 2
tc = bpy.data.cameras.new("top"); tc.type = "ORTHO"; tc.ortho_scale = float(max(hi - lo))
to = bpy.data.objects.new("top", tc); sc.collection.objects.link(to); to.location = (ctr[0], ctr[1], 12)
ceiling = [o for o in sc.objects if o.name.startswith("ceiling_")]
for o in ceiling: o.hide_render = True  # the plan and the bird's-eye view look in from above
sc.camera = to; sc.render.resolution_x = sc.render.resolution_y = 1200
sc.render.filepath = os.path.join(OUT, "top.png"); bpy.ops.render.render(write_still=True)
vc = bpy.data.cameras.new("view"); vc.lens = 24
vo = bpy.data.objects.new("view", vc); sc.collection.objects.link(vo)
vo.location = (ctr[0] + 3.5, lo[1] - 1.5, 4.5)
d = Vector((ctr[0], ctr[1], 0.6)) - vo.location
vo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
sc.camera = vo; sc.render.resolution_x, sc.render.resolution_y = 1600, 1000
sc.render.filepath = os.path.join(OUT, "view.png"); bpy.ops.render.render(write_still=True)
for o in ceiling: o.hide_render = False
if ceiling:  # 4) the whole room (ceiling too) from camera A, to compare with the photo
    ex, ey, ez = (Vector(v) for v in smap["cv_axes_in_floor"])
    ic = bpy.data.cameras.new("inside"); ic.sensor_fit = "VERTICAL" if smap["H"] > smap["W"] else "HORIZONTAL"
    ic.sensor_width = ic.sensor_height = 36.0; ic.lens = smap["fx_px"] / max(smap["W"], smap["H"]) * 36.0
    io = bpy.data.objects.new("inside", ic); sc.collection.objects.link(io)
    io.matrix_world = Matrix.Translation((0, 0, smap["camera_height_m"])) @ Matrix((ex, -ey, -ez)).transposed().to_4x4()
    sc.camera = io; sc.render.resolution_x, sc.render.resolution_y = smap["W"], smap["H"]
    sc.render.filepath = os.path.join(OUT, "inside_photo_A.png"); bpy.ops.render.render(write_still=True)

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "room.blend"))
bpy.ops.object.select_all(action="DESELECT")
for o in sc.objects:
    if o.type == "MESH" and o.name not in ("ground",): o.select_set(True)
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, "room.glb"), export_format="GLB", use_selection=True)
json.dump(report, open(os.path.join(OUT, "placed.json"), "w", encoding="utf-8"), indent=1)
print("DONE")
