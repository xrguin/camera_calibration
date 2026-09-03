#!/usr/bin/env python3
"""
AprilTag (tag36h11) detection + pose on a local USB camera, with an optional
reference-geometry calibration step like track_rov_topview.py.

The reference board is described by the EDITABLE VARIABLES in the
"Reference-board geometry" block below (REF_RECT / REF_LAYOUT / gates), not
by command-line arguments.  Plain mode (REF_RECT and REF_LAYOUT both None)
behaves as before: per-tag solvePnP IPPE_SQUARE poses with distance labels.

With a reference spec the script first sanity-checks the geometry map itself
(distinct positions, non-collinear for 3+, all expected pairwise distances
printed), then per frame:

  1. each reference tag gets an independent IPPE_SQUARE PnP, gated on
     reprojection RMS (<= %.1f px) and positive depth;
  2. every pairwise PnP distance is compared against the expected map
     distance — the live "does the calibration make sense" check;
  3. a plane homography candidate is estimated exactly as in
     track_rov_topview.py (2-tag joint corner fit / 3-tag SQPnP+range /
     4+-tag RANSAC homography), gated on world fit RMS and PnP baseline
     error, and frozen after REF_INIT_FRAMES accepted samples.

After the freeze every detected tag (reference or not) is also mapped
through the homography and labelled with its map x,y in metres, and the
frozen map is monitored for camera drift.

Usage:
    python3 usb_cam/usb_cam_apriltag.py

The default --calib is usb_cam/usb_cam_calibration.npz (this camera's own
calibration); the BlueROV calibration files in the repo root are never read.
In DISTORTION_MODE "auto" a two-tag layout uses raw pixels for geometry (the
present calibration views do not cover the extreme image edges — see
track_rov_topview.py).

Controls: q or Esc quit, r resets a frozen/accumulating calibration.
"""

import argparse
import math
import os
import time
from itertools import combinations

import cv2
import numpy as np

try:
    from usb_cam import track_rov_topview as topview
    from usb_cam import pool_layout_4tag as pool_layout
except ImportError:  # run as a plain script: usb_cam/ itself is on sys.path
    import track_rov_topview as topview
    import pool_layout_4tag as pool_layout

__doc__ = __doc__ % topview.MAX_REF_POSE_REPROJ_PX

APRILTAG_DICT = cv2.aruco.DICT_APRILTAG_36h11

# Stable device path for the bench camera — survives replugs, unlike the
# /dev/videoN index (which already shifted 4 -> 5 once).
DEFAULT_DEVICE = ("/dev/v4l/by-id/usb-Global_Shutter_Camera_"
                  "Global_Shutter_Camera_01.00.00-video-index0")

# ===========================================================================
# Reference-board geometry — EDIT THESE VARIABLES DIRECTLY
#
# The DEFAULT is the four-tag pool rectangle in pool_layout_4tag.py, which
# track_rov_topview.py and combined_view read from the same place — re-measure
# it there and every consumer follows.  World frame: origin at the near-left
# tag 100, +x along the camera-side edge toward 102, +y into the pool, metres.
#
# REF_RECT is the local override: rectangle corners on the tag plane, keys
# tl/tr/br/bl giving the tag ID at that corner (at least two, e.g. only tl+br
# for a diagonal); "lr" is the measured left->right centre-to-centre distance
# and "tb" the top->bottom one.  Distances accept "426cm", "2.02m", or a bare
# number in metres.  Its origin is the BOTTOM-LEFT corner of that rectangle.
# To use it, set REF_LAYOUT = None and fill REF_RECT in.
#
# Set REF_RECT and REF_LAYOUT both to None to disable calibration.
REF_RECT = None

# Alternative to REF_RECT: explicit tag centres in metres, {id: (x, y)}.
# Takes precedence over REF_RECT when set.
REF_LAYOUT = pool_layout.tag_centres()

# Reference-tag black-square edge in metres: one number for every tag, or a
# per-tag {id: size} map.  The pool frame mixes 0.20 m and 0.40 m tags.
REF_TAG_SIZE_M = pool_layout.tag_sizes()
REF_INIT_FRAMES = 30         # accepted samples before the map freezes
MAX_REF_WORLD_ERROR_M = 0.05  # max world/corner fit RMS, metres
MAX_REF_RANGE_ERROR_M = 0.60  # max PnP pairwise/baseline distance error, m
REF_DRIFT_M = 0.12           # frozen-map RMS that indicates camera movement
REF_DRIFT_FRAMES = 5         # consecutive drift frames before reacquiring
DISTORTION_MODE = "auto"     # "auto" | "raw" | "calibrated"
# ===========================================================================

_CORNER_ALIASES = {
    "tl": "tl", "topleft": "tl",
    "tr": "tr", "topright": "tr",
    "br": "br", "bottomright": "br",
    "bl": "bl", "bottomleft": "bl",
}
_WIDTH_ALIASES = {"w", "width", "lr", "leftright"}
_HEIGHT_ALIASES = {"h", "height", "tb", "topbottom"}


def parse_length_m(value):
    """'426cm' / '202 cm' / '4.26m' / 4.26 (metres) -> metres."""

    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip().lower().replace(" ", "")
    for suffix, factor in (("cm", 0.01), ("mm", 0.001), ("m", 1.0)):
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * factor
    return float(value)


def parse_ref_rect(spec):
    """Turn a REF_RECT dict (or 'tl=500; br=501; lr=426cm; tb=202cm' string)
    into {id: (x, y)} metres.

    World frame (matches track_rov_topview.py): origin at the BOTTOM-LEFT
    rectangle corner, +x along the bottom edge left->right (length lr/w),
    +y along the left edge bottom->top (length tb/h), so tl=(0, h) and
    br=(w, 0).  Corner IDs may be written as 500 or 'id500'.
    """

    if isinstance(spec, dict):
        items = list(spec.items())
    else:
        items = []
        for raw_item in str(spec).replace(",", ";").split(";"):
            item = raw_item.strip()
            if not item:
                continue
            for separator in ("=", ":"):
                if separator in item:
                    items.append(item.split(separator, 1))
                    break
            else:
                raise ValueError(
                    f"cannot parse {item!r}: expected key=value pairs like "
                    "'tl=500; br=501; lr=426cm; tb=202cm'")

    corners = {}
    width = height = None
    for key, value in items:
        key = str(key).strip().lower().replace("-", "").replace("_", "")
        if key in _CORNER_ALIASES:
            corner = _CORNER_ALIASES[key]
            if corner in corners:
                raise ValueError(f"rectangle corner {corner!r} given twice")
            tag = str(value).strip().lower()
            if tag.startswith("id"):
                tag = tag[2:]
            corners[corner] = int(tag)
        elif key in _WIDTH_ALIASES:
            width = parse_length_m(value)
        elif key in _HEIGHT_ALIASES:
            height = parse_length_m(value)
        else:
            raise ValueError(
                f"unknown key {key!r}: corners tl/tr/br/bl, width lr/w, "
                "height tb/h")

    if len(corners) < 2:
        raise ValueError("need at least two rectangle corner tag IDs")
    if len(set(corners.values())) != len(corners):
        raise ValueError("rectangle corner tag IDs must be distinct")
    if any(corner in corners for corner in ("tr", "br")) and width is None:
        raise ValueError(
            "corners tr/br need the left->right distance (lr=... or w=...)")
    if any(corner in corners for corner in ("tl", "tr")) and height is None:
        raise ValueError(
            "corners tl/tr need the top->bottom distance (tb=... or h=...)")
    if width is not None and width <= 0:
        raise ValueError("left->right distance must be positive")
    if height is not None and height <= 0:
        raise ValueError("top->bottom distance must be positive")

    place = {
        "tl": (0.0, height or 0.0),
        "tr": (width or 0.0, height or 0.0),
        "br": (width or 0.0, 0.0),
        "bl": (0.0, 0.0),
    }
    return {corners[corner]: place[corner] for corner in corners}


def build_ref_world():
    """Resolve the module-level geometry variables into {id: (x, y)}."""

    if REF_LAYOUT:
        return {int(tag): (float(x), float(y))
                for tag, (x, y) in REF_LAYOUT.items()}
    if REF_RECT:
        return parse_ref_rect(REF_RECT)
    return {}


def resolve_ref_tag_sizes(ref_world):
    """REF_TAG_SIZE_M (one number or {id: size}) -> {id: size} in metres.

    The pool frame mixes 0.20 m and 0.40 m references, and a 0.20 m tag
    solved as a 0.40 m one lands at twice its true range, so each tag has to
    carry its own edge length into solvePnP.
    """

    if isinstance(REF_TAG_SIZE_M, dict):
        missing = sorted(set(ref_world) - set(REF_TAG_SIZE_M))
        if missing:
            raise ValueError(
                f"REF_TAG_SIZE_M has no entry for tag(s) {missing}")
        sizes = {tag: float(REF_TAG_SIZE_M[tag]) for tag in ref_world}
    else:
        sizes = {tag: float(REF_TAG_SIZE_M) for tag in ref_world}
    for tag, size in sizes.items():
        if size <= 0:
            raise ValueError(f"REF_TAG_SIZE_M for tag {tag} must be positive")
    return sizes


def check_layout_geometry(ref_world):
    """Static sanity check of the user-supplied geometry map."""

    if len(ref_world) < 2:
        raise ValueError("reference map needs at least two tag IDs")
    xy = np.asarray(list(ref_world.values()), dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError("reference map contains non-finite coordinates")
    for (tag_a, pos_a), (tag_b, pos_b) in combinations(ref_world.items(), 2):
        if math.dist(pos_a, pos_b) <= 1e-6:
            raise ValueError(
                f"reference tags {tag_a} and {tag_b} share the same position")
    if len(ref_world) >= 3 and np.linalg.matrix_rank(xy[1:] - xy[0]) < 2:
        raise ValueError("3+ reference tag centres must not be collinear")


def describe_layout(ref_world, ref_tag_sizes):
    print(f"[geometry] reference map, {len(ref_world)} tags:")
    for tag, (x, y) in ref_world.items():
        print(f"[geometry]   ID {tag}: ({x:+.3f}, {y:+.3f}) m, "
              f"black-square edge {ref_tag_sizes[tag]:.3f} m")
    for (tag_a, pos_a), (tag_b, pos_b) in combinations(ref_world.items(), 2):
        print(f"[geometry]   expected |{tag_a}-{tag_b}| = "
              f"{math.dist(pos_a, pos_b):.3f} m")


def pnp_pair_errors(ref_seen, ref_world):
    """(tag_a, tag_b, PnP distance, expected distance) for valid pose pairs."""

    valid = [tag for tag in ref_world
             if tag in ref_seen and ref_seen[tag].get("pose_valid", False)]
    rows = []
    for tag_a, tag_b in combinations(valid, 2):
        measured = float(np.linalg.norm(
            np.asarray(ref_seen[tag_b]["tvec"], dtype=float)
            - np.asarray(ref_seen[tag_a]["tvec"], dtype=float)))
        expected = float(np.linalg.norm(
            np.asarray(ref_world[tag_b], dtype=float)
            - np.asarray(ref_world[tag_a], dtype=float)))
        rows.append((tag_a, tag_b, measured, expected))
    return rows


def gate_candidate(candidate_H, quality, max_pair_error, ref_world):
    """Return the reject-reason list for one homography candidate."""

    if candidate_H is None:
        return [quality["method"]]
    reject = []
    if len(ref_world) == 2 and not quality.get("pnp_valid", False):
        reject.append("reference_pnp_invalid")
    world_rms = quality["world_rms_m"]
    if not np.isfinite(world_rms):
        reject.append("world_rms_nonfinite")
    elif world_rms > MAX_REF_WORLD_ERROR_M:
        reject.append(f"world_rms>{MAX_REF_WORLD_ERROR_M:.3f}")
    range_rms = quality["range_rms_m"]
    if np.isfinite(range_rms) and range_rms > MAX_REF_RANGE_ERROR_M:
        reject.append(f"pnp_baseline_error>{MAX_REF_RANGE_ERROR_M:.3f}")
    if (np.isfinite(max_pair_error)
            and max_pair_error > MAX_REF_RANGE_ERROR_M):
        reject.append(f"pair_distance_error>{MAX_REF_RANGE_ERROR_M:.3f}")
    return reject


def print_freeze_report(H_mat, ref_seen, ref_world, quality, pair_rows,
                        K, geometry_dist, n_samples):
    print(f"[calib] map FROZEN after {n_samples} accepted samples "
          f"({quality['method']}); world fit RMS="
          f"{quality['world_rms_m']:.3f} m "
          f"(gate {MAX_REF_WORLD_ERROR_M:.3f})")
    for tag_a, tag_b, measured, expected in pair_rows:
        print(f"[calib]   pair {tag_a}-{tag_b}: PnP={measured:.3f} m "
              f"expected={expected:.3f} m error={measured - expected:+.3f} m")
    for tag in ref_world:
        if tag not in ref_seen:
            continue
        centre_und = topview.undistort_pts(
            [ref_seen[tag]["center_px"]], K, geometry_dist)
        mapped = topview.project(H_mat, centre_und)[0]
        expected = np.asarray(ref_world[tag], dtype=float)
        print(f"[calib]   ID {tag}: map=({mapped[0]:+.3f},{mapped[1]:+.3f}) m "
              f"expected=({expected[0]:+.3f},{expected[1]:+.3f}) m "
              f"error={np.linalg.norm(mapped - expected):.3f} m")


def load_calibration(path, frame_shape, fov_deg):
    """Return (K, dist, is_calibrated) for the live frame size.

    Loads the .npz (camera_matrix + dist_coeffs, rescaled from its native
    resolution) if given; otherwise builds an approximate pinhole from the
    assumed horizontal FOV, with zero distortion.
    """
    fh, fw = frame_shape[:2]
    if path:
        try:
            data = np.load(path)
            K = data["camera_matrix"].astype(np.float64)
            dist = data["dist_coeffs"].astype(np.float64)
            cw = int(data["image_w"]) if "image_w" in data else 0
            ch = int(data["image_h"]) if "image_h" in data else 0
            if cw and ch and (cw, ch) != (fw, fh):
                K = K.copy()
                K[0] *= fw / cw
                K[1] *= fh / ch
            print(f"[calib] loaded {path}")
            return K, dist, True
        except (FileNotFoundError, KeyError) as e:
            print(f"[calib] {path} unavailable ({e}); falling back to FOV estimate")

    f = fw / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    K = np.array([[f, 0.0, fw / 2.0],
                  [0.0, f, fh / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    print(f"[calib] APPROXIMATE intrinsics from {fov_deg:.0f} deg HFOV "
          f"(f={f:.0f} px) — distances are indicative, not metric. "
          "Calibrate with calibrate_camera.py for real pose.")
    return K, np.zeros((1, 5)), False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=DEFAULT_DEVICE,
                    help="V4L2 device index or path (default: stable by-id path "
                         "of the Global Shutter bench camera)")
    # 1920x1200 is the calibration's native size.  Any other capture size
    # makes load_calibration() rescale K, which is only correct if the driver
    # SCALES the sensor; the 1080-row mode is a CROP, so the rescale shifts
    # the principal point and biases every pose it feeds.
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--tag-size", type=float, default=0.40,
                    help="Physical tag edge length in metres for NON-reference "
                         "tags (default 0.40)")
    ap.add_argument("--calib",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "usb_cam_calibration.npz"),
                    help=".npz with camera_matrix + dist_coeffs for THIS camera "
                         "(default: usb_cam_calibration.npz next to this script)")
    ap.add_argument("--fov", type=float, default=60.0,
                    help="Assumed horizontal FOV in degrees when uncalibrated (default 60)")
    ap.add_argument("--print-rate", type=float, default=2.0,
                    help="Max detection prints per second (default 2)")
    ap.add_argument("--fps", type=float, default=90.0,
                    help="Requested camera capture rate (default 90; the "
                         "driver default of 30 measures ~15 fps in practice)")
    ap.add_argument("--record", nargs="?", const="", default=None,
                    metavar="PATH",
                    help="Record the annotated view (exactly what the preview "
                         "window shows, overlays included) to an MP4. Bare "
                         "--record writes usb_cam_apriltag_<timestamp>.mp4 "
                         "into the usb_cam/ folder next to this script")
    ap.add_argument("--record-fps", type=float, default=30.0,
                    help="MP4 cadence, independent of --fps (default 30)")
    ap.add_argument("--display-fps", type=float, default=15.0,
                    help="Max preview redraws per second (default 15)")
    args = ap.parse_args()

    ref_tag_sizes = {}
    try:
        ref_world = build_ref_world()
        if ref_world:
            check_layout_geometry(ref_world)
            ref_tag_sizes = resolve_ref_tag_sizes(ref_world)
        if ref_world and REF_INIT_FRAMES < 1:
            raise ValueError("REF_INIT_FRAMES must be at least 1")
    except ValueError as exc:
        raise SystemExit(f"bad REF_RECT/REF_LAYOUT configuration: {exc}")
    calibrate = bool(ref_world)
    if calibrate:
        describe_layout(ref_world, ref_tag_sizes)

    cap = topview.open_camera(args.device, args.width, args.height, args.fps)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.device!r} — check "
                         "ls /dev/v4l/by-id/ and that nothing else has it open")
    grabber = topview.FrameGrabber(cap, name="apriltag-capture")
    grabber.start()
    if not grabber.wait_first():
        grabber.close()
        cap.release()
        raise SystemExit(f"No frames from {args.device!r} within 5 s"
                         + (f" ({grabber.error})" if grabber.error else ""))

    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    obj_pts = topview.make_tag_object_points(args.tag_size)
    ref_obj = topview.make_tag_object_map(ref_tag_sizes)
    axis_len = args.tag_size * 0.5

    use_raw_geometry = (
        DISTORTION_MODE == "raw"
        or (DISTORTION_MODE == "auto" and len(ref_world) == 2))
    if calibrate:
        print("[geometry] "
              + ("raw-pixel mapping/PnP (two-tag layout or DISTORTION_MODE "
                 "'raw'): saved distortion bypassed for geometry"
                 if use_raw_geometry
                 else "applying saved camera distortion coefficients"))

    K = dist = geometry_dist = None
    H_mat = None
    H_samples = []
    H_quality = {"method": "none", "world_rms_m": np.nan,
                 "range_rms_m": np.nan}
    drift_count = 0
    win = "AprilTag tag36h11 (USB bench camera)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    t_prev, fps, last_print = time.time(), 0.0, 0.0
    display_period = 1.0 / max(float(args.display_fps), 0.1)
    next_display = 0.0
    writer = None
    record_path = None
    if args.record is not None:
        # Auto-named clips land next to this script (usb_cam/), matching the
        # --calib default, rather than wherever the shell happens to be.  An
        # explicit --record PATH is honoured as given.
        record_path = args.record or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"usb_cam_apriltag_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    last_sequence = 0

    try:
        while True:
            # Latest-frame slot fed by the capture thread; skipped frames show up
            # as sequence jumps rather than as a backlog of stale images.
            frame, _frame_host_ns, sequence = grabber.snapshot()
            if sequence == last_sequence:
                if grabber.stopped:
                    print("Frame grab failed, stopping."
                          + (f" ({grabber.error})" if grabber.error else ""))
                    break
                time.sleep(0.001)
                continue
            last_sequence = sequence
            if K is None:
                K, dist, is_calibrated = load_calibration(
                    args.calib, frame.shape, args.fov)
                geometry_dist = np.zeros_like(dist) if use_raw_geometry else dist
                if calibrate and not is_calibrated:
                    print("[calib] WARNING: reference calibration on approximate "
                          "FOV intrinsics — PnP validation is indicative only")
                print(f"[run] {frame.shape[1]}x{frame.shape[0]}, "
                      f"tag={args.tag_size} m"
                      + (", ref tags "
                         + "/".join(f"{tag}:{size:g}m" for tag, size
                                    in sorted(ref_tag_sizes.items()))
                         if calibrate else "")
                      + ". Detecting tag36h11 ...")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            # --- reference calibration --------------------------------------
            ref_seen = {}
            pair_rows = []
            reject = []
            if calibrate:
                ref_seen = topview.detect_references(
                    corners, ids, ref_world, ref_obj, K, geometry_dist)
                pair_rows = pnp_pair_errors(ref_seen, ref_world)
                max_pair_error = (
                    max(abs(measured - expected)
                        for _, _, measured, expected in pair_rows)
                    if pair_rows else np.nan)
                candidate_H, candidate_quality = (
                    topview.estimate_reference_homography(
                        ref_seen, ref_world, K, geometry_dist, ref_obj))
                reject = gate_candidate(
                    candidate_H, candidate_quality, max_pair_error, ref_world)
                if not reject and H_mat is None:
                    H_samples.append(candidate_H)
                    if len(H_samples) >= REF_INIT_FRAMES:
                        H_mat = topview.average_homographies(H_samples)
                        H_quality = candidate_quality
                        print_freeze_report(
                            H_mat, ref_seen, ref_world, candidate_quality,
                            pair_rows, K, geometry_dist, len(H_samples))

                # Frozen-map health: reference centres must keep mapping to
                # their expected positions, otherwise the camera moved.
                ref_world_rms = np.nan
                if H_mat is not None and ref_seen:
                    seen_ids = list(ref_seen)
                    seen_world = topview.project(H_mat, topview.undistort_pts(
                        [ref_seen[tag]["center_px"] for tag in seen_ids],
                        K, geometry_dist))
                    expected_world = np.asarray(
                        [ref_world[tag] for tag in seen_ids], dtype=float)
                    ref_world_rms = float(np.sqrt(np.mean(np.sum(
                        (seen_world - expected_world) ** 2, axis=1))))
                if (H_mat is not None and len(ref_seen) >= 2
                        and ref_world_rms > REF_DRIFT_M):
                    drift_count += 1
                else:
                    drift_count = 0
                if drift_count >= REF_DRIFT_FRAMES:
                    print(f"[calib] reference drift {ref_world_rms:.3f} m for "
                          f"{drift_count} frames; clearing map and reacquiring")
                    H_mat = None
                    H_samples.clear()
                    drift_count = 0

            # --- generic tag poses -------------------------------------------
            detections = []
            if ids is not None:
                for mc, mid in zip(corners, ids.flatten()):
                    mid = int(mid)
                    if mid in ref_world:
                        continue        # reference tags get their own overlay
                    img_pts = mc.reshape(-1, 2).astype(np.float64)
                    ok_pnp, rvec, tvec = cv2.solvePnP(
                        obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if not ok_pnp:
                        continue
                    detections.append((mid, rvec, tvec, img_pts))

            for mid, rvec, tvec, img_pts in detections:
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, axis_len)
                c = img_pts.astype(int)
                cv2.polylines(frame, [c.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                center = c.mean(axis=0).astype(int)
                d = float(np.linalg.norm(tvec))
                label = f"id={mid} {d:.2f}m"
                if H_mat is not None:
                    centre_px = topview.quadrilateral_center(img_pts)
                    mapped = topview.project(H_mat, topview.undistort_pts(
                        [centre_px], K, geometry_dist))[0]
                    label += f" map=({mapped[0]:+.2f},{mapped[1]:+.2f})m"
                cv2.putText(frame, label, (center[0] - 40, center[1] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # --- reference overlays ------------------------------------------
            if calibrate:
                good = H_mat is not None and drift_count == 0
                ref_color = (0, 255, 0) if good else (0, 165, 255)
                for tag in ref_world:
                    if tag not in ref_seen:
                        continue
                    detection = ref_seen[tag]
                    p = detection["center_px"].astype(int)
                    cv2.polylines(
                        frame,
                        [np.rint(detection["corners_px"]).astype(np.int32)
                         .reshape(-1, 1, 2)],
                        True, ref_color, 2, cv2.LINE_AA)
                    cv2.circle(frame, tuple(p), 7, ref_color, 2)
                    camera_xyz = np.asarray(detection["tvec"], dtype=float)
                    camera_text = (
                        f"cam xyz=({camera_xyz[0]:+.2f},{camera_xyz[1]:+.2f},"
                        f"{camera_xyz[2]:+.2f}) m"
                        if detection["pose_valid"] else "cam xyz=PnP invalid")
                    pnp_text = (
                        f"PnP reproj={detection['pnp_reproj_rms_px']:.2f} px"
                        if np.isfinite(detection["pnp_reproj_rms_px"])
                        else "PnP reproj=unavailable")
                    lines = [f"ID {tag} (ref)", camera_text, pnp_text]
                    if H_mat is not None:
                        mapped = topview.project(H_mat, topview.undistort_pts(
                            [detection["center_px"]], K, geometry_dist))[0]
                        expected = np.asarray(ref_world[tag], dtype=float)
                        lines.append(
                            f"map=({mapped[0]:+.3f},{mapped[1]:+.3f}) m "
                            f"err={np.linalg.norm(mapped - expected):.3f} m")
                    topview.draw_text_block(frame, lines, p, ref_color)

                for index, (tag_a, tag_b, measured, expected) in enumerate(
                        pair_rows):
                    cv2.putText(
                        frame,
                        f"{tag_a}-{tag_b}: PnP={measured:.3f} m "
                        f"target={expected:.3f} m "
                        f"err={measured - expected:+.3f} m",
                        (10, 60 + 26 * index), cv2.FONT_HERSHEY_SIMPLEX,
                        0.64, ref_color, 2, cv2.LINE_AA)
                if reject and H_mat is None:
                    cv2.putText(frame, "reject: " + ";".join(reject),
                                (10, 60 + 26 * max(len(pair_rows), 1)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.64,
                                (0, 165, 255), 2, cv2.LINE_AA)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            if detections and (now - last_print) >= 1.0 / max(args.print_rate, 0.1):
                last_print = now
                ids_str = ", ".join(f"#{mid}@{float(np.linalg.norm(t)):.2f}m"
                                    for mid, _, t, _ in detections)
                print(f"[detect] {len(detections)} tag(s): {ids_str}")

            if calibrate:
                stage = (
                    f"CALIBRATED ({H_quality['method']})" if H_mat is not None
                    else f"CALIBRATING {len(H_samples)}/{REF_INIT_FRAMES}"
                         " - hold camera/tags fixed")
                status = (f"{stage} | refs:{len(ref_seen)}/{len(ref_world)} "
                          f"tags:{len(detections)} {fps:.0f} fps")
            else:
                status = f"tags: {len(detections)}  {fps:.0f} fps"
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if (not calibrate or H_mat is not None)
                        else (0, 165, 255), 2)
            # Record the same annotated frame the preview shows -- every overlay
            # above is drawn onto `frame` in place, so the MP4 is exactly what you
            # see.  The writer thread holds this frame by reference and encodes it
            # on its own cadence; the capture thread hands out a fresh buffer each
            # time, so it is never redrawn underneath the encoder.
            if record_path is not None:
                if writer is None:
                    size = (frame.shape[1], frame.shape[0])
                    writer = topview.WallClockVideoWriter(
                        record_path, args.record_fps, size)
                    writer.submit(frame)
                    writer.start()
                    print(f"[record] {record_path} at {args.record_fps:.0f} fps "
                          f"({size[0]}x{size[1]}), overlays included")
                if writer.error is not None:
                    raise RuntimeError(f"MP4 encoder failed: {writer.error}")
                writer.submit(frame)

            if time.monotonic() >= next_display:
                next_display = time.monotonic() + display_period
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r") and calibrate:
                    print("[calib] manual reset; reacquiring reference map")
                    H_mat = None
                    H_samples.clear()
                    drift_count = 0

    except KeyboardInterrupt:
        print("\n[run] interrupted")
    finally:
        # Always finalize: cv2.VideoWriter only writes the MP4 moov
        # atom on release(), so skipping this on Ctrl-C leaves the
        # whole recording unplayable.
        grabber.close()
        cap.release()
        if writer is not None:
            writer.close()
            print(f"[record] wrote {writer.frames_written} frames to "
                  f"{record_path}; encoder dropped "
                  f"{writer.dropped_seconds:.3f} s")
        print(f"[run] camera delivered {grabber.frames_read} frames")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
