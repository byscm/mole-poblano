# How well the silhouette of a GLB model, seen from the real camera, matches the crop from the photo.
# Module use: from silhouette import fit_pose;  fit_pose(glb, crop_rgba_png, elev) -> dict
# CLI: python -I silhouette.py <model.glb> <crop.png> [elev_deg]
import json, struct
import numpy as np
from PIL import Image

N = 96


def glb_vertices(path):
    """All GLB vertices (with node transforms applied), in Blender axes (z up)."""
    b = open(path, "rb").read()
    nj = struct.unpack_from("<I", b, 12)[0]
    js = json.loads(b[20:20 + nj])
    b0 = 20 + nj + 8
    def mat(n):
        if "matrix" in n:
            return np.array(n["matrix"], float).reshape(4, 4).T
        M = np.eye(4)
        if "scale" in n:
            M = np.diag(list(n["scale"]) + [1]) @ M
        if "rotation" in n:
            x, y, z, w = n["rotation"]
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            Rm = np.eye(4); Rm[:3, :3] = R
            M = Rm @ M
        if "translation" in n:
            T = np.eye(4); T[:3, 3] = n["translation"]
            M = T @ M
        return M
    nodes = js.get("nodes", [])
    pts = []
    def visit(i, P):
        n = nodes[i]
        M = P @ mat(n)
        if "mesh" in n:
            for pr in js["meshes"][n["mesh"]]["primitives"]:
                a = js["accessors"][pr["attributes"]["POSITION"]]
                bv = js["bufferViews"][a["bufferView"]]
                off = b0 + bv.get("byteOffset", 0) + a.get("byteOffset", 0)
                v = np.frombuffer(b, np.float32, a["count"] * 3, off).reshape(-1, 3)
                pts.append(v @ M[:3, :3].T + M[:3, 3])
        for c in n.get("children", []):
            visit(c, M)
    scene = js["scenes"][js.get("scene", 0)]
    for i in scene["nodes"]:
        visit(i, np.eye(4))
    v = np.concatenate(pts)
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)  # glTF (y up) -> Blender (z up)


def to_grid(xs, ys, w, h):
    """Rasterize points (already in 0..1) to an N x N mask, keeping the aspect ratio."""
    m = np.zeros((N, N), bool)
    L = max(w, h)
    ox, oy = (L - w) / 2 / L, (L - h) / 2 / L
    ix = np.clip(((xs * w / L) + ox) * (N - 1), 0, N - 1).astype(int)
    iy = np.clip(((ys * h / L) + oy) * (N - 1), 0, N - 1).astype(int)
    m[iy, ix] = True
    # close the gaps between points
    for _ in range(2):
        d = m.copy(); d[1:] |= m[:-1]; d[:-1] |= m[1:]; d[:, 1:] |= m[:, :-1]; d[:, :-1] |= m[:, 1:]; m = d
    for _ in range(1):
        e = m.copy(); e[1:] &= m[:-1]; e[:-1] &= m[1:]; e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]; m = e
    return m


def crop_silhouette(path):
    a = np.array(Image.open(path).convert("RGBA"))[..., 3] > 127
    ys, xs = np.nonzero(a)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    L = max(w, h)
    canvas = np.zeros((L, L), bool)
    canvas[(L - h) // 2:(L - h) // 2 + h, (L - w) // 2:(L - w) // 2 + w] = a
    return np.array(Image.fromarray(canvas.astype(np.uint8) * 255).resize((N, N), Image.NEAREST)) > 127, w / h


def fake_base(v):
    """TRELLIS sometimes adds a wide, thin 'floor' slab under the object.
    Returns the height to cut at (or None if there is no slab)."""
    z = v[:, 2]
    z0, z1 = z.min(), z.max()
    h = z1 - z0
    def width(a, b):
        s = v[(z >= a) & (z < b)]
        return np.ptp(s[:, :2], 0).max() if len(s) > 20 else 0.0
    above = width(z0 + 0.25 * h, z1)
    for f in np.arange(0.0, 0.30, 0.005):  # move up little by little while it is still slab
        if width(z0 + f * h, z0 + (f + 0.01) * h) < 1.3 * above:
            return None if f < 0.005 else float(z0 + (f + 0.005) * h)
    return None


def rotate(v, yaw, pitch):
    p, g = np.radians(pitch), np.radians(yaw)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(g), -np.sin(g), 0], [np.sin(g), np.cos(g), 0], [0, 0, 1]])
    return v @ (Rz @ Rx).T


def silhouette_view(v, elev):
    """Silhouette seen from -Y with the camera raised 'elev' degrees (orthographic projection)."""
    e = np.radians(elev)
    x = v[:, 0]
    y = v[:, 1] * np.sin(e) + v[:, 2] * np.cos(e)
    w, h = np.ptp(x), np.ptp(y)
    return to_grid((x - x.min()) / max(w, 1e-9), (y.max() - y) / max(h, 1e-9), w, h)


def fit_pose(glb, crop, elev=0.0, pitches=(0, 90, -90), step=15):
    """Find the yaw and whether the model lies down (pitch) so that, seen from the camera, it matches the crop.
    Soft preference for yaw=0 and pitch=0 (TRELLIS generates facing front what it sees in the image)."""
    v = glb_vertices(glb)
    if len(v) > 60000:
        v = v[np.random.default_rng(0).choice(len(v), 60000, replace=False)]
    cut = fake_base(v)
    if cut is not None:
        v = v[v[:, 2] >= cut]
    sr, _ = crop_silhouette(crop)
    best = None
    for p in pitches:
        # front and back give almost the same (mirrored) silhouette: only yaws from -75 to 75 are searched,
        # because TRELLIS generates facing front what it sees in the image
        for y in [g % 360 for g in range(-75, 76, step)]:
            sm = silhouette_view(rotate(v, y, p), elev)
            iou = float((sm & sr).sum() / max(1, (sm | sr).sum()))
            yw = min(y, 360 - y)
            score = iou - 0.0008 * yw - (0.06 if p else 0)
            if best is None or score > best[0]:
                best = (score, iou, y, p)
    ext = np.ptp(v, 0)
    return dict(iou=round(best[1], 3), yaw=best[2], pitch=best[3], base_cut=cut,
                relative_thickness=round(float(ext.min() / ext.max()), 3), elev=round(float(elev), 1))


if __name__ == "__main__":
    import sys
    print(fit_pose(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.0))
