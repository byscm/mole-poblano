# SYMBOLIC scene of the map: each object is a box with its size (width x depth x height), its number
# and an arrow towards where it faces. Useful to measure distances and see the layout without generating models.
# Usage: blender -b --factory-startup -P build_symbolic.py -- <map_folder>
# Writes: symbolic_scene.blend, _symbolic_render.png, _symbolic_overlay_raw.png (draw_plan.py makes symbolic_overlay.png)

import bpy, json, math, os, sys, colorsys
import numpy as np
from mathutils import Matrix, Vector

FOLDER = sys.argv[sys.argv.index("--") + 1]
scene_map = json.load(open(os.path.join(FOLDER, "map.json"), encoding="utf-8"))
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene


def material(name, rgb, alpha=1.0, emission=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Alpha"].default_value = alpha
    if emission:
        b.inputs["Emission Color"].default_value = (*rgb, 1)
        b.inputs["Emission Strength"].default_value = emission
    return m


n = max(1, len(scene_map["objects"]))
for k, o in enumerate(scene_map["objects"]):
    rgb = colorsys.hsv_to_rgb(k / n, 0.65, 0.95)
    g = math.radians(o.get("yaw_from_appearance", o["yaw_deg"]))
    a, f, h = max(o["width_m"], 0.05), max(o["depth_m"], 0.05), max(o["height_m"], 0.05)
    front = Vector((math.sin(g), -math.cos(g), 0))
    p = Vector((o["center_m"][0], o["center_m"][1], 0))
    c = p + p.normalized() * (f / 2)  # the measured part is the visible face; the body extends away from the camera
    z0 = o.get("base_z_m", 0.0)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(c.x, c.y, z0 + h / 2))
    box = bpy.context.object
    box.name = f"{o['id']:02d}_{o['name']}"
    box.scale = (a, f, h)
    box.rotation_euler = (0, 0, g)
    box.data.materials.append(material(box.name, rgb, 0.85))
    # arrow on the floor: where it faces
    L = max(0.25, 0.6 * max(a, f))
    base = c + front * (f / 2)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=L, location=base + front * (L / 2) + Vector((0, 0, 0.02)),
                                        rotation=(math.pi / 2, 0, g + math.pi))
    bpy.context.object.data.materials.append(material("arrow", (0.05, 0.05, 0.05)))
    bpy.ops.mesh.primitive_cone_add(radius1=0.06, depth=0.12, location=base + front * (L + 0.06) + Vector((0, 0, 0.02)),
                                    rotation=(math.pi / 2, 0, g + math.pi))
    bpy.context.object.data.materials.append(material("arrow", (0.05, 0.05, 0.05)))
    # number on top
    bpy.ops.object.text_add(location=(c.x, c.y, z0 + h + 0.08))
    t = bpy.context.object
    t.data.body = str(o["id"])
    t.data.size = 0.35
    t.data.align_x = t.data.align_y = "CENTER"
    t.data.extrude = 0.01
    t.data.materials.append(material("text", (0.02, 0.02, 0.02)))
    t.constraints.new("TRACK_TO")  # faces the camera (assigned later)
    t.hide_render = True  # in the images draw_plan.py draws the number; in the .blend it is visible

# distances: each object with its 2 nearest neighbours (line on the floor + label in metres).
# Only seen from above; in the .blend they are in the "distances" collection.
col_d = bpy.data.collections.new("distances"); sc.collection.children.link(col_d)
cen = {o["id"]: Vector((o["center_m"][0], o["center_m"][1], 0.03)) for o in scene_map["objects"]}
dist = {}
for p in scene_map["distances"]:
    dist[(p["a"], p["b"])] = dist[(p["b"], p["a"])] = p["distance_m"]
pairs = set()
for i in cen:
    near = sorted((d, j) for (a, j), d in dist.items() if a == i)[:2]
    pairs |= {tuple(sorted((i, j))) for _, j in near}
mat_d = material("distance_line", (0.1, 0.1, 0.1))
for i, j in sorted(pairs):
    a, b = cen[i], cen[j]
    d = b - a
    bpy.ops.mesh.primitive_cylinder_add(radius=0.012, depth=d.length, location=(a + b) / 2)
    ln = bpy.context.object
    ln.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    ln.data.materials.append(mat_d)
    bpy.ops.object.text_add(location=(a + b) / 2 + Vector((0, 0, 0.05)))
    tx = bpy.context.object
    tx.data.body = f"{dist[(i, j)]:.2f} m"
    tx.data.size = 0.2
    tx.data.align_x = tx.data.align_y = "CENTER"
    tx.data.materials.append(material("distance_text", (0.0, 0.15, 0.6)))
    for ob in (ln, tx):
        for c0 in ob.users_collection:
            c0.objects.unlink(ob)
        col_d.objects.link(ob)
        ob.hide_render = True

# floor with a 50 cm grid (catches shadows in the overlay over the photo)
bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
floor = bpy.context.object
floor.is_shadow_catcher = True
for gx in np.arange(-15, 15.01, 0.5):
    for axis in (0, 1):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(gx, 7.5, 0.001) if axis == 0 else (0, gx + 7.5, 0.001))
        r = bpy.context.object
        r.scale = (0.006, 30, 0.001) if axis == 0 else (30, 0.006, 0.001)
        r.data.materials.append(material("grid", (0.2, 0.2, 0.2), 0.25))
        r.name = "grid"
        r.hide_render = True  # only seen from above
        r["grid"] = True

# camera of the photo
ex, ey, ez = (Vector(v) for v in scene_map["cv_axes_in_floor"])
R = Matrix((ex, -ey, -ez)).transposed()
W, H = scene_map["W"], scene_map["H"]
cd = bpy.data.cameras.new("photo_camera")
cd.sensor_fit = "HORIZONTAL" if W >= H else "VERTICAL"
cd.sensor_width = cd.sensor_height = 36.0
cd.lens = scene_map["fx_px"] / max(W, H) * 36.0
cam = bpy.data.objects.new("photo_camera", cd)
sc.collection.objects.link(cam)
cam.matrix_world = Matrix.Translation((0, 0, scene_map["camera_height_m"])) @ R.to_4x4()
bpy.ops.object.empty_add(location=(0, 0, scene_map["camera_height_m"]))
cam_marker = bpy.context.object; cam_marker.name = "PHOTO_CAMERA"
for ob in sc.objects:
    if ob.type == "FONT" and ob.constraints:
        ob.constraints[0].target = cam
        ob.constraints[0].track_axis = "TRACK_Z"
        ob.constraints[0].up_axis = "UP_Y"

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.0
sun.rotation_euler = (math.radians(35), 0, math.radians(-30))
sc.collection.objects.link(sun)
sc.world = bpy.data.worlds.new("world")
sc.world.color = (0.9, 0.9, 0.9)

sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items] else "CYCLES"
if sc.render.engine == "CYCLES":
    sc.cycles.samples = 32
sc.render.film_transparent = True
sc.render.resolution_x, sc.render.resolution_y = W, H
sc.camera = cam
sc.render.filepath = os.path.join(FOLDER, "_symbolic_render.png")
bpy.ops.render.render(write_still=True)

photo = np.array(bpy.data.images.load(os.path.join(FOLDER, "photo.png")).pixels[:]).reshape(H, W, 4)
r3 = np.array(bpy.data.images.load(sc.render.filepath).pixels[:]).reshape(H, W, 4)
a = r3[..., 3:4] * 0.8
out = photo.copy(); out[..., :3] = r3[..., :3] * a + photo[..., :3] * (1 - a)
im = bpy.data.images.new("overlay", W, H, alpha=True); im.pixels = out.ravel()
im.filepath_raw = os.path.join(FOLDER, "_symbolic_overlay_raw.png"); im.file_format = "PNG"; im.save()

# the top-down plan is drawn by draw_plan.py (sharper and with measurements)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(FOLDER, "symbolic_scene.blend"))
print("DONE")
