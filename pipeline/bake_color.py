# Gives a colourless detailed mesh (e.g. from Hi3D) the colour of the coarse textured mesh it was refined from:
# decimates the detailed mesh, unwraps it, and bakes the coarse mesh's base colour onto it (selected to active).
# Usage: blender -b --factory-startup --python bake_color.py -- <textured.glb> <detailed.glb> <out.glb> [faces=400000] [tex=2048]
import os, sys
import bpy
from mathutils import Vector

a = sys.argv[sys.argv.index("--") + 1:]
src, det, out = a[:3]; FACES = int(a[3]) if len(a) > 3 else 400000; TEX = int(a[4]) if len(a) > 4 else 2048
bpy.ops.wm.read_factory_settings(use_empty=True)


def load(path):
    before = set(bpy.data.objects); bpy.ops.import_scene.gltf(filepath=path)
    ms = [o for o in set(bpy.data.objects) - before if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in ms: o.select_set(True)
    bpy.context.view_layer.objects.active = ms[0]
    if len(ms) > 1: bpy.ops.object.join()
    o = bpy.context.view_layer.objects.active
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM"); bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return o


S, D = load(src), load(det)
bb = lambda o: (Vector([min((o.matrix_world @ Vector(c))[i] for c in o.bound_box) for i in range(3)]), Vector([max((o.matrix_world @ Vector(c))[i] for c in o.bound_box) for i in range(3)]))
(s0, s1), (d0, d1) = bb(S), bb(D)
print("bounds textured", tuple(round(v, 3) for v in s1 - s0), "detailed", tuple(round(v, 3) for v in d1 - d0))
n0 = len(D.data.polygons)
if n0 > FACES:
    m = D.modifiers.new("dec", "DECIMATE"); m.ratio = FACES / n0
    bpy.context.view_layer.objects.active = D; bpy.ops.object.modifier_apply(modifier="dec")
print("faces", n0, "->", len(D.data.polygons))
bpy.context.view_layer.objects.active = D; bpy.ops.object.select_all(action="DESELECT"); D.select_set(True)
bpy.ops.object.shade_smooth()
bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.003); bpy.ops.object.mode_set(mode="OBJECT")
img = bpy.data.images.new("baked_color", TEX, TEX)
mat = bpy.data.materials.new("baked"); mat.use_nodes = True; nt = mat.node_tree
tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = img
nt.links.new(tex.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
nt.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
D.data.materials.clear(); D.data.materials.append(mat); nt.nodes.active = tex
sc = bpy.context.scene; sc.render.engine = "CYCLES"; sc.cycles.samples = 4
p = bpy.context.preferences.addons["cycles"].preferences; p.compute_device_type = "OPTIX"; p.get_devices()
for dv in p.devices: dv.use = dv.type == "OPTIX"
sc.cycles.device = "GPU"
sc.render.bake.use_pass_direct = sc.render.bake.use_pass_indirect = False; sc.render.bake.use_pass_color = True
sc.render.bake.use_selected_to_active = True; sc.render.bake.cage_extrusion = 0.02 * max(d1 - d0); sc.render.bake.max_ray_distance = 0.05 * max(d1 - d0)
sc.render.bake.margin = 8
bpy.ops.object.select_all(action="DESELECT"); S.select_set(True); D.select_set(True); bpy.context.view_layer.objects.active = D
bpy.ops.object.bake(type="DIFFUSE")
img.pack()
bpy.ops.object.select_all(action="DESELECT"); D.select_set(True)
bpy.ops.export_scene.gltf(filepath=out, use_selection=True, export_format="GLB")
print("WROTE", out, os.path.getsize(out) // 1024, "KB")
