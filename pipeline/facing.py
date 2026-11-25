# Which way an object faces, from one crop, by asking Qwen3-VL.
# Angles are "relative": 0 = its front points at the camera, +90 = it faces the viewer's right,
# -90 = the viewer's left, 180 = away from the camera. Seen from above, counterclockwise is positive.
import math, re
import numpy as np

QUESTIONS = {
    # which of its own sides we see
    "side": ("This is a {name}. Which part of the {name} is facing the camera? Use the object's OWN left and right. "
             "Answer with exactly one of: front, back, left side, right side, front-left corner, front-right corner, "
             "back-left corner, back-right corner."),
    # where its front points in the picture
    "direction": ("This is a {name}. In this picture, which way is the front of the {name} pointing? "
                  "Answer with exactly one of: toward the camera, away from the camera, left, right, "
                  "toward the camera and left, toward the camera and right, away and left, away and right."),
}

_SIDE = [("front-left", -45), ("front-right", 45), ("back-left", -135), ("back-right", 135),
         ("front left", -45), ("front right", 45), ("back left", -135), ("back right", 135),
         ("left", -90), ("right", 90), ("front", 0), ("back", 180), ("rear", 180)]
_DIRECTION = [("toward the camera and left", -45), ("toward the camera and right", 45),
              ("away and left", -135), ("away and right", 135),
              ("toward the camera", 0), ("towards the camera", 0), ("away", 180), ("left", -90), ("right", 90)]


def parse_answer(key, text):
    t = text.lower().replace("\n", " ")
    for word, ang in (_SIDE if key == "side" else _DIRECTION):
        if re.search(r"\b" + re.escape(word) + r"\b", t):
            return float(ang)
    return None


def relative_angle(to_cam, front):
    """Signed angle (deg, counterclockwise from above) that turns the object->camera direction into its front."""
    a = math.atan2(to_cam[0] * front[1] - to_cam[1] * front[0], to_cam[0] * front[0] + to_cam[1] * front[1])
    return math.degrees(a)


def angle_diff(a, b):
    d = abs((a - b) % 360)
    return min(d, 360 - d)


def front_from_relative(to_cam, rel_deg):
    r = math.radians(rel_deg)
    v = np.asarray(to_cam, float) / max(np.linalg.norm(to_cam), 1e-9)
    return np.array([v[0] * math.cos(r) - v[1] * math.sin(r), v[0] * math.sin(r) + v[1] * math.cos(r)])
