#!/usr/bin/env python3
"""
Single source of truth for the four-tag pool frame (04.08.2026 layout).

Four tag36h11 markers sit at the four corners of the pool, in two sizes: the
0.20 m pair on the camera-side (near) edge and the 0.40 m pair on the far
edge.  The pool frame has its ORIGIN AT THE NEAR-LEFT TAG, +x along the near
edge to the near-right tag, +y across to the far edge, +z up out of the
water.  All four centres are taken to be coplanar at z = 0.

    far-left  501 (0, y_edge) ------------ 500 (x_edge, y_edge) far-right
                    |                              |
                    | +y                           |
                    |                              |
    near-left 100 (0, 0) ------ +x ------- 102 (x_edge, 0)   near-right
              ORIGIN                                      camera side

Edit the dictionaries below and every consumer follows:

  * usb_cam/pool_frame_4tag.py        -- pool-frame PnP + visualization
  * usb_cam/track_rov_topview.py      -- overhead tracker default layout
  * usb_cam/usb_cam_apriltag.py       -- bench detection/calibration view
  * usb_cam/usb_cam_waypoints.py      -- waypoint pattern frame
  * combined_view/keyboard_stabilize_topview.py -- controller runs

Deliberately dependency-free (no numpy, no cv2) so the control-path modules
can import it without cost.
"""

from itertools import combinations

# ===========================================================================
# EDIT THESE
# ===========================================================================

# Tag IDs at the four rectangle corners.  "origin" is pool (0, 0).
POOL_RECT = {
    "corners": {
        "origin": 100,      # near-left  (camera side, left)  -> (0, 0)
        "x_axis": 102,      # near-right (camera side, right) -> (x_edge, 0)
        "y_axis": 501,      # far-left                        -> (0, y_edge)
        "diagonal": 500,    # far-right                  -> (x_edge, y_edge)
    },
    # CENTRE-TO-CENTRE distances.  Accept "426cm", "4.26m", or metres.
    #
    # Best-fit rectangle through the tag centres measured on 2026-08-04 (see
    # TAG_LAYOUT below, which is what is actually in force).  Kept so a
    # rectangle is still available if TAG_LAYOUT is ever set back to None.
    "x_edge_m": "441.3cm",  # origin -> x_axis  (near edge, 100 -> 102)
    "y_edge_m": "207.7cm",  # origin -> y_axis  (side edge, 100 -> 501)
}

# Black-square edge length of each reference tag, metres.  Mixed sizes are
# the whole point: solving a 0.20 m tag as a 0.40 m tag puts it at twice its
# true range.
TAG_SIZE_M = {
    100: 0.20,
    102: 0.20,
    501: 0.40,
    500: 0.40,
}

# Tag centres in metres, {id: (x, y)}.  Takes precedence over POOL_RECT.
#
# MEASURED FROM THE TAGS THEMSELVES on 2026-08-04, not from a tape: 120
# frames in which all four tags held a valid pose, each frame giving four
# independent camera-frame 3-D centres from per-tag IPPE PnP, fitted to a
# plane, projected onto it, expressed with the origin at 100 and +x toward
# 102, then taken frame-wise median.  Repeatability was 17-27 mm frame to
# frame and ~15 mm between separate capture runs.
#
# The four tags are NOT quite a rectangle, so the explicit centres are used
# rather than two edge lengths.  Forcing the best-fit 4.413 x 2.077 rectangle
# leaves a 40 mm worst-pair residual, and this removes it:
#
#     near edge 100-102  4.427 m     far edge  501-500  4.400 m   (28 mm apart)
#     left edge 100-501  2.038 m     right edge 102-500 2.117 m   (79 mm apart)
#     corner angles      89.32  89.88  89.08  91.71 deg
#
# The 79 mm left/right difference and 501's 1.71 deg corner both exceed the
# measurement noise, so they are real placement, not error.
#
# These span 153 mm more in x and 57 mm more in y than the old tape-measured
# 4.26 x 2.02.  That is expected rather than alarming: the tags are mounted
# on boards that stand proud of the pool frame, so the tag centres enclose
# more than the pool does, and the near (two 0.20 m tags) and far (two 0.40 m
# tags) edges agree to 28 mm -- which they could not if a tag size were wrong.
#
# Re-measure by running: python3 usb_cam/pool_frame_4tag.py
TAG_LAYOUT = {
    100: (0.000, 0.000),    # near-left, origin by construction
    102: (4.427, 0.000),    # near-right, on +x by construction
    501: (0.024, 2.038),    # far-left
    500: (4.423, 2.117),    # far-right
}

# Edge length assumed for a tag that is NOT part of the layout (the ROV
# marker, a hand-held tag), metres.
OTHER_TAG_SIZE_M = 0.20

# ===========================================================================


def parse_length_m(value):
    """'426cm' / '202 cm' / '4.26m' / 4.26 (metres) -> metres."""

    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower().replace(" ", "")
    for suffix, factor in (("cm", 0.01), ("mm", 0.001), ("m", 1.0)):
        if text.endswith(suffix):
            return float(text[: -len(suffix)]) * factor
    return float(text)


def rect_wh():
    """(x_edge, y_edge) in metres: the extent the pool frame spans.

    With TAG_LAYOUT set the tags are not exactly a rectangle, so this is
    their bounding extent.  It is what the overlay rectangle, the rectified
    top-view canvas and the waypoint wall-clearance gate size themselves
    from, none of which can work with None -- the top view in particular
    silently disappears without it.
    """

    if isinstance(TAG_LAYOUT, dict) and TAG_LAYOUT:
        xs = [x for x, _ in TAG_LAYOUT.values()]
        ys = [y for _, y in TAG_LAYOUT.values()]
        return max(xs) - min(xs), max(ys) - min(ys)
    width = parse_length_m(POOL_RECT["x_edge_m"])
    height = parse_length_m(POOL_RECT["y_edge_m"])
    if width <= 0.0 or height <= 0.0:
        raise ValueError("POOL_RECT edge lengths must be positive")
    return width, height


def tag_centres():
    """{tag id: (x, y)} in metres, validated."""

    if isinstance(TAG_LAYOUT, dict) and TAG_LAYOUT:
        centres = {int(tag): (float(x), float(y))
                   for tag, (x, y) in TAG_LAYOUT.items()}
    else:
        corners = POOL_RECT["corners"]
        missing = {"origin", "x_axis", "y_axis", "diagonal"} - set(corners)
        if missing:
            raise ValueError(
                f"POOL_RECT['corners'] is missing {sorted(missing)}")
        width, height = rect_wh()
        centres = {
            int(corners["origin"]): (0.0, 0.0),
            int(corners["x_axis"]): (width, 0.0),
            int(corners["y_axis"]): (0.0, height),
            int(corners["diagonal"]): (width, height),
        }
        if len(centres) != 4:
            raise ValueError("POOL_RECT corner tag IDs must be distinct")

    if len(centres) < 3:
        raise ValueError("the pool frame needs at least three tags")
    for tag in centres:
        if tag not in TAG_SIZE_M:
            raise ValueError(f"TAG_SIZE_M has no entry for tag {tag}")
        if float(TAG_SIZE_M[tag]) <= 0.0:
            raise ValueError(f"TAG_SIZE_M[{tag}] must be positive")
    for (tag_a, a), (tag_b, b) in combinations(centres.items(), 2):
        if math_dist(a, b) <= 1e-6:
            raise ValueError(f"tags {tag_a} and {tag_b} share a position")
    points = list(centres.values())
    base = points[0]
    if all(abs(cross(base, points[i], points[j])) <= 1e-9
           for i, j in combinations(range(1, len(points)), 2)):
        raise ValueError("tag centres must not be collinear")
    return centres


def tag_sizes():
    """{tag id: black-square edge in metres} for the configured tags."""

    return {tag: float(TAG_SIZE_M[tag]) for tag in tag_centres()}


def cross(origin, a, b):
    return ((a[0] - origin[0]) * (b[1] - origin[1])
            - (a[1] - origin[1]) * (b[0] - origin[0]))


def math_dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def layout_spec():
    """Layout as track_rov_topview's ``--ref-layout`` string."""

    return ";".join(
        f"{tag}:{x:.4f},{y:.4f}"
        for tag, (x, y) in sorted(tag_centres().items()))


def tag_size_spec():
    """Sizes as track_rov_topview's ``--ref-tag-size`` string."""

    return ";".join(
        f"{tag}:{size:.4f}" for tag, size in sorted(tag_sizes().items()))


def describe(prefix="[layout]"):
    """Human-readable report lines, including every configured pair."""

    centres = tag_centres()
    sizes = tag_sizes()
    lines = [f"{prefix} pool frame: origin at tag "
             f"{min(centres, key=lambda t: centres[t])}, +x along the near "
             "edge, +y into the pool, +z up"]
    for tag in sorted(centres):
        x, y = centres[tag]
        lines.append(f"{prefix}   ID {tag}: ({x:+.3f}, {y:+.3f}) m, "
                     f"edge {sizes[tag]:.3f} m")
    dimensions = rect_wh()
    if dimensions is not None:
        lines.append(
            f"{prefix}   rectangle {dimensions[0]:.3f} x "
            f"{dimensions[1]:.3f} m, diagonal "
            f"{math_dist((0.0, 0.0), dimensions):.3f} m")
    for tag_a, tag_b in combinations(sorted(centres), 2):
        lines.append(
            f"{prefix}   configured |{tag_a}-{tag_b}| = "
            f"{math_dist(centres[tag_a], centres[tag_b]):.3f} m")
    return lines


if __name__ == "__main__":
    print("\n".join(describe()))
    print(f"[layout] --ref-layout   {layout_spec()}")
    print(f"[layout] --ref-tag-size {tag_size_spec()}")
