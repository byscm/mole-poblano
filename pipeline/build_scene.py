# Builds the map scene (map.json) in Blender: each object with its generated (or gallery) model,
# scaled to the measured height, turned to where it faces and put in its place; camera equal to the photo's.
# Renders over the original photo (floor as shadow catcher) to check that everything lands in place.
# Usage: blender -b --factory-startup -P build_scene.py -- <map_folder>
# Writes: scene.blend, render_3d.png, overlay.png, comparison.png

import bpy, json, math, os, sys
import numpy as np
from mathutils import Matrix, Vector

SCENE = sys.argv[sys.argv.index("--") + 1]
scene_map = json.load(open(os.path.join(SCENE, "map.json"), encoding="utf-8"))
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

for o in scene_map["objects"]:
    path = o.get("model") or os.path.join(SCENE, "generated", f"object_{o['id']}.glb")
    if not os.path.exists(path):
        print("no model:", o["id"], o["name"])
        continue
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [x for x in bpy.data.objects if x not in before]
    meshes = [x for x in new if x.type == "MESH"]
    for x in meshes:  # bake transforms
        mw = x.matrix_world.copy()
        x.parent = None
        x.data.transform(mw)
        x.matrix_world = Matrix.Identity(4)
    for x in new:
        if x.type != "MESH":
            bpy.data.objects.remove(x)
    # pose found by comparing silhouettes (fit_poses.py): remove the fake base, lay down if needed, rotate
    pose_path = os.path.join(SCENE, "generated", f"object_{o['id']}_pose.json")
    pose = json.load(open(pose_path)) if os.path.exists(pose_path) else None
    if pose and pose.get("base_cut") is not None:
        import bmesh
        for x in meshes:
            bm = bmesh.new(); bm.from_mesh(x.data)
            bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < pose["base_cut"]], context="VERTS")
            bm.to_mesh(x.data); bm.free()
    if pose:
        for x in meshes:
            x.data.transform(Matrix.Rotation(math.radians(pose["yaw"]), 4, "Z") @ Matrix.Rotation(math.radians(pose["pitch"]), 4, "X"))
        to_cam = -Vector((o["center_m"][0], o["center_m"][1], 0)).normalized()
        g = math.atan2(to_cam.y, to_cam.x) + math.pi / 2  # turns the view from -Y towards where the camera is
    else:
        g = math.radians(o["yaw_deg"])
    vs = np.array([v.co[:] for x in meshes for v in x.data.vertices])
    mn, mx = vs.min(0), vs.max(0)
    # gallery model: already has its verified real size; generated: scaled to the measured height
    s = 1.0 if o.get("model") else o["height_m"] / max(mx[2] - mn[2], 1e-6)
    depth = (mx[1] - mn[1]) * s
    front = Vector((math.sin(g), -math.cos(g), 0))  # R(g) @ (0,-1): where it faces
    # what was measured is the face the camera sees: the object's center is half a depth behind it
    p = Vector((o["center_m"][0], o["center_m"][1], 0))
    c = p + p.normalized() * (depth / 2)  # the visible face was measured; the body extends away from the camera
    T = (Matrix.Translation((c.x, c.y, o.get("base_z_m", 0.0))) @ Matrix.Rotation(g, 4, "Z") @ Matrix.Scale(s, 4)
         @ Matrix.Translation((-(mn[0] + mx[0]) / 2, -(mn[1] + mx[1]) / 2, -mn[2])))
    for x in meshes:
        x.data.transform(T)
        x.name = f"{o['id']}_{o['name']}"
        # make TRELLIS vertex colors show up in Cycles
        if x.data.color_attributes and not x.data.materials:
            m = bpy.data.materials.new(x.name); m.use_nodes = True
            nt = m.node_tree
            ca = nt.nodes.new("ShaderNodeVertexColor"); ca.layer_name = x.data.color_attributes[0].name
            nt.links.new(ca.outputs[0], nt.nodes["Principled BSDF"].inputs["Base Color"])
            x.data.materials.append(m)
    print(f"object {o['id']} {o['name']}: scale {s:.3f}  size {(mx-mn)[0]*s:.2f} x {(mx-mn)[1]*s:.2f} x {o['height_m']:.2f} m")

# floor: shadow catcher (over the photo only the shadows show)
bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
floor = bpy.context.object
floor.is_shadow_catcher = True

# photo camera
ex, ey, ez = (Vector(v) for v in scene_map["cv_axes_in_floor"])
R = Matrix((ex, -ey, -ez)).transposed()
cd = bpy.data.cameras.new("cam")
W, H = scene_map["W"], scene_map["H"]
cd.sensor_fit = "HORIZONTAL" if W >= H else "VERTICAL"
cd.sensor_width = cd.sensor_height = 36.0
cd.lens = scene_map["fx_px"] / max(W, H) * 36.0
cam = bpy.data.objects.new("cam", cd)
sc.collection.objects.link(cam)
cam.matrix_world = Matrix.Translation((0, 0, scene_map["camera_height_m"])) @ R.to_4x4()
sc.camera = cam

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.5
sun.rotation_euler = (math.radians(45), 0, math.radians(-30))
sc.collection.objects.link(sun)
sc.world = bpy.data.worlds.new("world"); sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.8, 0.85, 0.95, 1)
sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.8

sc.render.engine = "CYCLES"
try:
    pr = bpy.context.preferences.addons["cycles"].preferences
    pr.compute_device_type = "OPTIX"
    pr.refresh_devices() if hasattr(pr, "refresh_devices") else pr.get_devices()
    for d in pr.devices:
        d.use = d.type == "OPTIX"
    sc.cycles.device = "GPU"
except Exception as e:
    print("GPU:", e)
sc.cycles.samples = 96
sc.cycles.use_denoising = True
sc.render.film_transparent = True
sc.render.resolution_x, sc.render.resolution_y = W, H
sc.render.filepath = os.path.join(SCENE, "render_3d.png")
bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(SCENE, "scene.blend"))

# over the photo and side by side
photo = bpy.data.images.load(os.path.join(SCENE, "photo.png"))
r3 = bpy.data.images.load(os.path.join(SCENE, "render_3d.png"))
pf = np.array(photo.pixels[:]).reshape(H, W, 4)
p3 = np.array(r3.pixels[:]).reshape(H, W, 4)
a = p3[..., 3:4]
overlay = pf.copy(); overlay[..., :3] = p3[..., :3] * a + pf[..., :3] * (1 - a)
alone = p3.copy(); alone[..., :3] = p3[..., :3] * a + 0.92 * (1 - a); alone[..., 3] = 1
for name, arr in (("overlay.png", overlay), ("comparison.png", np.concatenate([pf, alone], 1))):
    h, w = arr.shape[:2]
    im = bpy.data.images.new(name, w, h, alpha=True)
    im.pixels = arr.ravel()
    im.filepath_raw = os.path.join(SCENE, name); im.file_format = "PNG"; im.save()
print("DONE")
