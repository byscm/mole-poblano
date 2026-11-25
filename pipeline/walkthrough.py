# Walk-through video of the furnished room (room.blend from build_room.py): a camera at eye height moves along a path
# and looks at a moving target; Eevee renders PNG frames (encode with ffmpeg).
# Usage: blender -b <room.blend> -P walkthrough.py -- <out_frames_folder> [first last [step [keys.json]]]   (frames 1..N, 24 fps)
#   keys.json: [[second, [camera x, y, z], [target x, y, z]], ...] replaces the room path below
import bpy, json, math, os, sys
from mathutils import Vector

a = sys.argv[sys.argv.index("--") + 1:]
OUT = a[0]
os.makedirs(OUT, exist_ok=True)
sc = bpy.context.scene
FPS = 24
# (second, camera position, where it looks) in room metres; camera A (photo 2) is at (0, 0), y = towards the kitchen
KEYS = [
    (0.0, (0.15, -0.35, 1.25), (0.0, 4.0, 0.9)),     # sitting on the loveseat, like photo 2
    (4.0, (0.9, 1.6, 1.6), (-0.3, 4.2, 0.8)),        # stand up, walk past the low table, look at the dining table
    (8.0, (0.7, 3.2, 1.6), (1.9, 6.2, 0.9)),         # between the table and the heater (0.7 m from both), turn to the island
    (12.0, (0.9, 7.2, 1.6), (3.2, 8.6, 1.0)),        # in the kitchen aisle, look at the back counter
    (15.0, (0.75, 5.5, 1.65), (-0.3, 7.6, 0.8)),      # step back and turn to the folding table by the windows
    (19.0, (0.6, 6.4, 1.8), (-1.6, 0.8, 0.7)),       # turn round: the whole room down to the lounge
    (23.0, (0.2, 3.2, 2.2), (-2.0, 0.6, 0.5)),       # walk back, looking at the L sofa and the armchair
]
if len(a) > 4:
    KEYS = [(k[0], tuple(k[1]), tuple(k[2])) for k in json.load(open(a[4], encoding="utf-8"))]
N = int(KEYS[-1][0] * FPS) + 1

for o in list(sc.objects):  # cameras/lights from build_room are not needed
    if o.type == "CAMERA":
        bpy.data.objects.remove(o)
cam = bpy.data.objects.new("walk_cam", bpy.data.cameras.new("walk_cam")); sc.collection.objects.link(cam)
cam.data.lens = 20; cam.data.clip_start = 0.05
tgt = bpy.data.objects.new("walk_target", None); sc.collection.objects.link(tgt)
c = cam.constraints.new("TRACK_TO"); c.target = tgt; c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
for t, p, q in KEYS:
    f = 1 + int(t * FPS)
    cam.location = p; cam.keyframe_insert("location", frame=f)
    tgt.location = q; tgt.keyframe_insert("location", frame=f)
sc.camera = cam
sc.frame_start, sc.frame_end = (int(a[1]), min(int(a[2]), N)) if len(a) > 2 else (1, N)
sc.render.fps = FPS
sc.frame_step = int(a[3]) if len(a) > 3 else 1  # e.g. 96 for a quick preview

# hide the photo-only helper and light it nicely
for o in sc.objects:
    if o.name == "ground":
        o.is_shadow_catcher = False
sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y = 1280, 720
sc.render.film_transparent = False
sc.render.image_settings.file_format = "PNG"
sc.render.filepath = os.path.join(OUT, "f_")
bpy.ops.render.render(animation=True)
print("DONE", sc.frame_start, sc.frame_end)
