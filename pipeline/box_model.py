# Simple built-in counter as a GLB: a charcoal body with a matte black stone top that overhangs a little at front and ends.
# For objects the image-to-3D route can't model well (built-in, flat furniture). Sizes in metres, origin at floor centre.
# Usage: blender -b --factory-startup -P box_model.py -- <out.glb> [length depth height top_thickness overhang [body_hex top_hex]]
import bpy, sys
a = sys.argv[sys.argv.index("--") + 1:]
out = a[0]
L, D, H, T, O = (list(map(float, a[1:6])) + [3.0, 0.60, 0.95, 0.03, 0.02][len(a[1:6]):])[:5]
hexrgb = lambda h: tuple((int(h[i:i + 2], 16) / 255) ** 2.2 for i in (0, 2, 4))  # sRGB hex -> linear
BODY, TOP = (hexrgb(a[6]), hexrgb(a[7])) if len(a) > 7 else ((0.05, 0.05, 0.055), (0.012, 0.012, 0.012))
bpy.ops.wm.read_factory_settings(use_empty=True)


def mat(name, rgb, rough):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1); b.inputs["Roughness"].default_value = rough
    return m


def box(name, sx, sy, sz, z0, y0, m):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, y0, z0 + sz / 2))
    o = bpy.context.object; o.name = name; o.scale = (sx, sy, sz)
    bpy.ops.object.transform_apply(scale=True); o.data.materials.append(m)
    return o

# back of the counter against the wall at y = +D/2; front faces -y
box("body", L, D, H - T, 0, 0, mat("body", BODY, 0.8))
box("top", L + 2 * O, D + O, T, H - T, -O / 2, mat("top", TOP, 0.55))
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB")
print(f"{out}: {L} x {D} x {H} m")
