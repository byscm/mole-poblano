# Renders a GLB from 4 sides (turns of 0, 90, 180, 270 deg, camera 20 deg above) with its own colours, Eevee.
# Usage: blender -b --factory-startup -P render_views.py -- <in.glb> <out_prefix>
# Writes <out_prefix>_y000.png ... _y270.png (512 px, light grey background)
import bpy, sys, math
from mathutils import Vector

glb, prefix = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
ms = [o for o in bpy.context.scene.objects if o.type == "MESH"]
vs = [o.matrix_world @ v.co for o in ms for v in o.data.vertices]
mn = Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs)))
mx = Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs)))
c, r = (mn + mx) / 2, (mx - mn).length
pivot = bpy.data.objects.new("pivot", None); bpy.context.scene.collection.objects.link(pivot)
pivot.location = c
for o in ms:
    if o.parent is None:
        o.parent = pivot; o.matrix_parent_inverse = pivot.matrix_world.inverted()
for o in bpy.context.scene.objects:  # glTF roots
    if o.type != "MESH" and o is not pivot and o.parent is None:
        o.parent = pivot; o.matrix_parent_inverse = pivot.matrix_world.inverted()
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x = sc.render.resolution_y = 512
sc.render.film_transparent = False
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.82, 0.82, 0.8, 1); w.node_tree.nodes["Background"].inputs[1].default_value = 1.0
sun = bpy.data.lights.new("sun", "SUN"); sun.energy = 3.0
so = bpy.data.objects.new("sun", sun); sc.collection.objects.link(so); so.rotation_euler = (math.radians(50), 0, math.radians(30))
cam = bpy.data.cameras.new("cam"); cam.lens = 50
co = bpy.data.objects.new("cam", cam); sc.collection.objects.link(co); sc.camera = co
el = math.radians(20); d = r * 2.0
co.location = c + Vector((0, -d * math.cos(el), d * math.sin(el)))
co.rotation_euler = (math.pi / 2 - el, 0, 0)
for yaw in (0, 90, 180, 270):
    pivot.rotation_euler = (0, 0, math.radians(yaw))
    sc.render.filepath = f"{prefix}_y{yaw:03d}.png"
    bpy.ops.render.render(write_still=True)
