#!/usr/bin/env python3
"""
Four-tag pool-frame PnP on the USB bench camera, with a live visualization of
the pool frame the tags define.

Layout (04.08.2026): four tag36h11 markers at the four corners of the pool,
mixed sizes -- two 0.20 m tags on the camera-side (near) edge and two 0.40 m
tags on the far edge.  The pool frame has its ORIGIN AT THE NEAR-LEFT TAG,
+x running along the near edge to the near-right tag, +y from the near edge
across to the far edge, +z up out of the water.  All four tag centres are
assumed coplanar at z = 0.

    far-left  501 (0, y_edge) ------------ 500 (x_edge, y_edge) far-right
                    |                              |
                    | +y                           |
                    |                              |
    near-left 100 (0, 0) ------ +x ------- 102 (x_edge, 0)   near-right
              ORIGIN                                      camera side

Everything is configured by the dictionaries in the CONFIGURATION block
below -- there are no command-line arguments.  Edit POOL_RECT / TAG_SIZE_M
when the tags move or are re-measured.

Per frame:

  1. every configured tag is detected and gets its OWN IPPE_SQUARE PnP using
     ITS OWN edge length, so the 0.20 m and 0.40 m tags are no longer forced
     through a single size (which is what usb_cam_apriltag.py does, and what
     would put the small tags at twice their true range);
  2. the pool pose is initialised by rigidly aligning the configured tag
     centres to the camera-frame centres those per-tag poses recover (a
     Kabsch fit -- unambiguous for 3+ non-collinear tags, unlike a 4-point
     planar PnP which returns two mirrored solutions);
  3. optionally refined over all 16 tag corners: with the current pose each
     tag's detected corners are back-projected onto z = 0, a per-tag in-plane
     yaw is fitted with the tag centre held at its measured position, and the
     resulting 16 world corners are fed to solvePnP.  Alternating these two
     steps is the same trick estimate_two_tag_homography() uses, lifted from
     a plane homography to a full 6-DoF pose;
  4. gated on centre reprojection RMS, per-tag pose quality, and every
     pairwise PnP-vs-layout distance, then accumulated and frozen after
     PNP["init_frames"] accepted samples.  A frozen pose is monitored for
     drift and re-acquired if the camera moves.

The visualization draws, through the solved pose:

  * the pool rectangle and a metre grid, densely sampled so the drawn lines
    follow the real lens distortion instead of cutting straight across it;
  * the pool-frame axes at the origin tag;
  * each reference tag with its measured position and reprojection error;
  * any OTHER detected tag (the ROV marker, say) labelled with its pool
    coordinates, obtained both by intersecting its centre ray with z = 0 and
    from its own PnP;
  * an optional rectified top view (metric bird's-eye warp of the pool).

Usage:
    python3 usb_cam/pool_frame_4tag.py

Controls: q or Esc quit, r re-acquire the pose, g toggle the grid,
v toggle the top view.
"""

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

_HERE = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# CONFIGURATION
#
# The GEOMETRY -- which tag sits at which corner, the two edge lengths, and
# each tag's black-square size -- lives in pool_layout_4tag.py, because
# track_rov_topview.py, usb_cam_apriltag.py and combined_view read the same
# definition.  Edit it there once and every consumer follows.
#
# The dictionaries below are this script's own runtime settings.
# ===========================================================================

OTHER_TAG_SIZE_M = pool_layout.OTHER_TAG_SIZE_M

CAMERA = {
    # Stable device path -- survives replugs, unlike /dev/videoN.
    "device": ("/dev/v4l/by-id/usb-Global_Shutter_Camera_"
               "Global_Shutter_Camera_01.00.00-video-index0"),
    "video": None,          # path to an MP4 to replay instead of the camera
    # 1920x1200 is the calibration's native size: any other frame size makes
    # load_calibration() rescale K, which is only correct if the driver
    # SCALES the sensor.  The 1080-row mode is a CROP, so the rescale shifts
    # the principal point and biases every pose.  Keep 1200 here.
    "width": 1920,
    "height": 1200,
    "fps": 90.0,
    "calib": os.path.join(_HERE, "usb_cam_calibration.npz"),
    "display_fps": 15.0,
    "print_rate": 1.0,      # status lines per second on stdout
}

# "calibrated" applies the saved distortion coefficients (correct for a real
# metric PnP on this 116-deg-FOV lens); "raw" bypasses them, matching what
# usb_cam_apriltag.py falls back to for two-tag layouts because the present
# calibration views do not cover the extreme image edges.  A PnP wants
# "calibrated"; if the corner tags reproject badly, recalibrate with views
# that reach the edges rather than switching to "raw".
DISTORTION_MODE = "calibrated"

PNP = {
    "min_tags": 3,              # tags needed before a pose is attempted
    "refine_corners": True,     # alternate yaw/pose fit over all 16 corners
    "refine_iters": 12,
    "max_tag_reproj_px": 5.0,   # per-tag IPPE gate
    # A tag whose centre fails this round-trip sits outside the radius the
    # distortion model was fitted on; its pose is meaningless, so it is
    # dropped rather than averaged in.  Recalibrate with edge coverage --
    # see usb_cam/calibrate.sh.
    "max_distortion_round_trip_px": 1.0,
    "max_centre_rms_px": 4.0,   # pool-pose gate, tag centres
    "max_pair_error_m": 0.15,   # |PnP pair distance - configured distance|
    "init_frames": 30,          # accepted poses before the pose freezes
    "drift_px": 12.0,           # frozen-pose centre RMS meaning "camera moved"
    "drift_frames": 5,
}

DRAW = {
    "rectangle": True,
    "grid": True,
    "grid_step_m": 0.5,
    "axes": True,
    "axis_len_m": 0.6,
    "tag_labels": True,
    "pair_table": True,
    "other_tags": True,
    "topview": True,
    "topview_px_per_m": 160.0,
    "topview_margin_m": 0.30,
    "topview_window_xy": (40, 700),
}

RECORD = {
    "enabled": False,
    "path": None,           # None -> pool_frame_4tag_<timestamp>.mp4 in usb_cam/
    "fps": 30.0,
}

# ===========================================================================

APRILTAG_DICT = cv2.aruco.DICT_APRILTAG_36h11

COLOR_OK = (0, 255, 0)
COLOR_WAIT = (0, 165, 255)
COLOR_RECT = (34, 59, 194)
COLOR_GRID = (120, 120, 120)
COLOR_OTHER = (0, 255, 255)


def build_layout():
    """Resolve pool_layout_4tag into {id: {"xy": array, "size_m": float}}."""

    centres = pool_layout.tag_centres()
    sizes = pool_layout.tag_sizes()
    layout = {tag: {"xy": np.asarray(xy, dtype=float),
                    "size_m": sizes[tag]}
              for tag, xy in centres.items()}
    return layout, pool_layout.rect_wh()


def describe_layout(layout, rect_wh):
    for line in pool_layout.describe():
        print(line)


def load_calibration(path, frame_shape):
    """Return (K, dist) for the live frame size, from the saved .npz."""

    frame_h, frame_w = frame_shape[:2]
    data = np.load(path)
    K = data["camera_matrix"].astype(np.float64)
    dist = data["dist_coeffs"].astype(np.float64)
    calib_w = int(data["image_w"]) if "image_w" in data else 0
    calib_h = int(data["image_h"]) if "image_h" in data else 0
    print(f"[calib] loaded {path} ({calib_w}x{calib_h})")
    if calib_w and calib_h and (calib_w, calib_h) != (frame_w, frame_h):
        K = K.copy()
        K[0] *= frame_w / calib_w
        K[1] *= frame_h / calib_h
        print(f"[calib] WARNING: capturing {frame_w}x{frame_h} but calibrated "
              f"at {calib_w}x{calib_h}; K was RESCALED, which is only valid "
              "if the driver scales the sensor. A cropped mode needs a "
              "principal-point shift instead, and every pose below inherits "
              "the error. Capture at the calibrated size.")
    return K, dist


def distortion_round_trip_px(pixels, K, dist):
    """Undistort then re-distort each pixel; return the error in pixels.

    ``cv2.undistortPoints`` iterates, and when the radial polynomial folds --
    which a low-order fit on a wide lens does outside the radius it was
    actually fitted on -- it exhausts its iterations and returns the INPUT
    unchanged, with no error flag.  Every pose built on such a point is then
    silently wrong by hundreds of pixels while the image centre still looks
    perfect.  Round-tripping is the only cheap way to see it.
    """

    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if not len(pixels):
        return np.zeros(0)
    normalized = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    rays = np.column_stack([normalized, np.ones(len(normalized))])
    back, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K, dist)
    return np.linalg.norm(back.reshape(-1, 2) - pixels, axis=1)


def make_tag_corners_local(size_m):
    """Tag black-square corners in the tag's own plane, ArUco corner order."""

    half = float(size_m) / 2.0
    return np.array([[-half, half], [half, half],
                     [half, -half], [-half, -half]], dtype=np.float64)


def detect_layout_tags(corners, ids, layout, K, dist):
    """Detect the configured tags, each with its OWN edge length.

    usb_cam_apriltag.py hands one object-point set to every reference tag,
    so a 0.20 m tag solved as a 0.40 m tag lands at twice its true range and
    poisons every pairwise distance check.  Here each tag carries its size.
    """

    seen = {}
    if ids is None:
        return seen
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        marker_id = int(marker_id)
        if marker_id not in layout:
            continue
        image_corners = marker_corners.reshape(-1, 2).astype(np.float64)
        center_px = topview.quadrilateral_center(image_corners)
        size_m = layout[marker_id]["size_m"]
        object_points = np.column_stack([
            make_tag_corners_local(size_m), np.zeros(4)])
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_corners, K, dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        reproj_rms = np.nan
        tvec_xyz = np.full(3, np.nan)
        if ok:
            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, K, dist)
            reproj_rms = float(np.sqrt(np.mean(np.sum(
                (projected.reshape(-1, 2) - image_corners) ** 2, axis=1))))
            tvec_xyz = np.asarray(tvec, dtype=float).reshape(3)
        # Reject a tag sitting where the distortion model is not invertible
        # before its pose can poison the pool fit.
        model_error = float(distortion_round_trip_px(
            [center_px], K, dist)[0])
        model_ok = model_error <= PNP["max_distortion_round_trip_px"]
        pose_valid = bool(
            ok and model_ok and np.isfinite(tvec_xyz).all()
            and tvec_xyz[2] > 0.0
            and np.isfinite(reproj_rms)
            and reproj_rms <= PNP["max_tag_reproj_px"])
        seen[marker_id] = {
            "model_round_trip_px": model_error,
            "model_ok": model_ok,
            "corners_px": image_corners,
            "center_px": center_px,
            "size_m": size_m,
            "rvec": np.asarray(rvec, dtype=float).reshape(3) if ok
            else np.full(3, np.nan),
            "tvec": tvec_xyz,
            "range_m": (float(np.linalg.norm(tvec_xyz)) if pose_valid
                        else np.nan),
            "reproj_rms_px": reproj_rms,
            "pose_valid": pose_valid,
        }
    return seen


def pair_distance_rows(seen, layout):
    """(a, b, PnP distance, configured distance) for valid per-tag poses."""

    valid = [tag for tag in sorted(layout)
             if tag in seen and seen[tag]["pose_valid"]]
    rows = []
    for tag_a, tag_b in combinations(valid, 2):
        measured = float(np.linalg.norm(
            seen[tag_b]["tvec"] - seen[tag_a]["tvec"]))
        expected = float(np.linalg.norm(
            layout[tag_b]["xy"] - layout[tag_a]["xy"]))
        rows.append((tag_a, tag_b, measured, expected))
    return rows


def rigid_transform(source, target):
    """Proper rotation + translation with target ~= R @ source + t."""

    source = np.asarray(source, dtype=float).reshape(-1, 3)
    target = np.asarray(target, dtype=float).reshape(-1, 3)
    source_c = source.mean(axis=0)
    target_c = target.mean(axis=0)
    covariance = (source - source_c).T @ (target - target_c)
    left, _, right_t = np.linalg.svd(covariance)
    reflection = np.diag([1.0, 1.0,
                          float(np.sign(np.linalg.det(right_t.T @ left.T)))])
    rotation = right_t.T @ reflection @ left.T
    return rotation, target_c - rotation @ source_c


def camera_position_pool(rvec, tvec):
    """Camera centre expressed in the pool frame."""

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    return (-rotation.T @ np.asarray(tvec, dtype=float).reshape(3))


def camera_to_pool(points_cam, rvec, tvec):
    """Camera-frame points -> pool-frame points."""

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    points = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    return (rotation.T @ (points - np.asarray(
        tvec, dtype=float).reshape(3)).T).T


def pixels_to_pool_plane(pixels, rvec, tvec, K, dist):
    """Intersect the back-projected rays of raw pixels with the z = 0 plane.

    This is the pool-frame position of anything that actually lies on the tag
    plane, and the honest reading for anything that does not: a submerged ROV
    seen from above appears where its ray pierces that plane.
    """

    pixels = np.asarray(pixels, dtype=float).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(pixels, K, dist).reshape(-1, 2)
    rays_cam = np.column_stack([normalized, np.ones(len(normalized))])
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    origin = camera_position_pool(rvec, tvec)
    rays_pool = (rotation.T @ rays_cam.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = -origin[2] / rays_pool[:, 2]
    points = origin + scale[:, None] * rays_pool
    points[~np.isfinite(scale) | (scale <= 0.0)] = np.nan
    return points[:, :2]


def tag_yaw_world_corners(seen, layout, rvec, tvec, K, dist):
    """World corners of every seen tag under the current pose.

    Each tag's detected corners are back-projected onto z = 0 and a pure
    in-plane rotation is fitted with the centre pinned to its configured
    position -- the measured layout stays the metric truth, only the printed
    orientation of each square is recovered.
    """

    world_corners = {}
    for tag, detection in seen.items():
        plane_pts = pixels_to_pool_plane(
            detection["corners_px"], rvec, tvec, K, dist)
        if not np.isfinite(plane_pts).all():
            continue
        local = make_tag_corners_local(detection["size_m"])
        centred = plane_pts - layout[tag]["xy"]
        left, _, right_t = np.linalg.svd(local.T @ centred)
        reflection = np.diag([1.0,
                              float(np.sign(np.linalg.det(right_t.T @ left.T)))])
        rotation = right_t.T @ reflection @ left.T
        world_corners[tag] = np.column_stack([
            layout[tag]["xy"] + local @ rotation.T, np.zeros(4)])
    return world_corners


def centre_reprojection_rms(seen, layout, rvec, tvec, K, dist):
    """RMS pixel error of the configured tag centres under a pose."""

    tags = sorted(seen)
    if not tags:
        return np.nan
    world = np.asarray(
        [[*layout[tag]["xy"], 0.0] for tag in tags], dtype=float)
    projected, _ = cv2.projectPoints(world, rvec, tvec, K, dist)
    observed = np.asarray([seen[tag]["center_px"] for tag in tags])
    return float(np.sqrt(np.mean(np.sum(
        (projected.reshape(-1, 2) - observed) ** 2, axis=1))))


def solve_pool_pose(seen, layout, K, dist):
    """Solve the pool-frame pose from the detected layout tags.

    Returns ``(rvec, tvec, quality)``; ``rvec`` is None when no pose could be
    produced and ``quality["method"]`` says why.
    """

    quality = {"method": "insufficient", "n_tags": len(seen),
               "centre_rms_px": np.nan, "corner_rms_px": np.nan,
               "camera_height_m": np.nan, "refined": False}
    tags = sorted(seen)
    if len(tags) < int(PNP["min_tags"]):
        return None, None, quality

    valid = [tag for tag in tags if seen[tag]["pose_valid"]]
    world_centres = np.asarray(
        [[*layout[tag]["xy"], 0.0] for tag in tags], dtype=float)
    image_centres = np.asarray(
        [seen[tag]["center_px"] for tag in tags], dtype=float)

    rvec = tvec = None
    if len(valid) >= 3:
        # Metric alignment of the configured centres onto the camera-frame
        # centres the per-tag PnPs recovered.  Both sides are already in
        # metres, so this fixes rotation, translation AND the mirror branch a
        # four-point planar PnP would leave open.
        source = np.asarray(
            [[*layout[tag]["xy"], 0.0] for tag in valid], dtype=float)
        target = np.asarray([seen[tag]["tvec"] for tag in valid], dtype=float)
        rotation, translation = rigid_transform(source, target)
        rvec = cv2.Rodrigues(rotation)[0]
        tvec = translation.reshape(3, 1)
        quality["method"] = f"{len(valid)}-tag centre alignment"
    elif len(tags) >= 4:
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            world_centres, image_centres, K, dist, flags=cv2.SOLVEPNP_IPPE)
        best = None
        if ok:
            for candidate_r, candidate_t in zip(rvecs, tvecs):
                # The camera must sit above the water, not mirrored below it.
                if camera_position_pool(candidate_r, candidate_t)[2] <= 0.0:
                    continue
                rms = centre_reprojection_rms(
                    seen, layout, candidate_r, candidate_t, K, dist)
                if np.isfinite(rms) and (best is None or rms < best[0]):
                    best = (rms, candidate_r, candidate_t)
        if best is None:
            quality["method"] = "planar-PnP-no-physical-branch"
            return None, None, quality
        _, rvec, tvec = best
        quality["method"] = f"{len(tags)}-tag planar PnP (centres)"
    else:
        quality["method"] = (
            f"only {len(valid)} valid tag pose(s) of {len(tags)} seen")
        return None, None, quality

    # Least-squares polish on the centres before anything else looks at it.
    ok, rvec, tvec = cv2.solvePnP(
        world_centres, image_centres, K, dist, rvec=rvec, tvec=tvec,
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        quality["method"] += " (centre polish failed)"
        return None, None, quality

    corner_rms = np.nan
    if PNP["refine_corners"]:
        for _ in range(int(PNP["refine_iters"])):
            world_corners = tag_yaw_world_corners(
                seen, layout, rvec, tvec, K, dist)
            if len(world_corners) < int(PNP["min_tags"]):
                break
            corner_tags = sorted(world_corners)
            object_points = np.vstack(
                [world_corners[tag] for tag in corner_tags])
            image_points = np.vstack(
                [seen[tag]["corners_px"] for tag in corner_tags])
            ok, new_rvec, new_tvec = cv2.solvePnP(
                object_points, image_points, K, dist, rvec=rvec.copy(),
                tvec=tvec.copy(), useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                break
            change = (float(np.linalg.norm(new_rvec - rvec))
                      + float(np.linalg.norm(new_tvec - tvec)))
            rvec, tvec = new_rvec, new_tvec
            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, K, dist)
            corner_rms = float(np.sqrt(np.mean(np.sum(
                (projected.reshape(-1, 2) - image_points) ** 2, axis=1))))
            quality["refined"] = True
            if change < 1e-9:
                break

    quality["centre_rms_px"] = centre_reprojection_rms(
        seen, layout, rvec, tvec, K, dist)
    quality["corner_rms_px"] = corner_rms
    quality["camera_height_m"] = float(camera_position_pool(rvec, tvec)[2])
    return rvec, tvec, quality


def gate_pose(rvec, quality, pair_rows):
    """Reject reasons for one pose candidate; empty list means accept."""

    if rvec is None:
        return [quality["method"]]
    reject = []
    centre_rms = quality["centre_rms_px"]
    if not np.isfinite(centre_rms):
        reject.append("centre_rms_nonfinite")
    elif centre_rms > PNP["max_centre_rms_px"]:
        reject.append(f"centre_rms>{PNP['max_centre_rms_px']:.1f}px")
    if not np.isfinite(quality["camera_height_m"]):
        reject.append("camera_height_nonfinite")
    elif quality["camera_height_m"] <= 0.0:
        reject.append("camera_below_tag_plane")
    worst = max((abs(measured - expected)
                 for _, _, measured, expected in pair_rows), default=0.0)
    if worst > PNP["max_pair_error_m"]:
        reject.append(f"pair_distance_error>{PNP['max_pair_error_m']:.2f}m")
    return reject


def average_poses(samples):
    """Combine accepted pose samples: rotation by SVD, translation by median."""

    rotations = np.stack(
        [cv2.Rodrigues(np.asarray(rvec).reshape(3, 1))[0]
         for rvec, _ in samples])
    left, _, right_t = np.linalg.svd(rotations.sum(axis=0))
    reflection = np.diag([1.0, 1.0,
                          float(np.sign(np.linalg.det(left @ right_t)))])
    rotation = left @ reflection @ right_t
    tvec = np.median(
        np.stack([np.asarray(t).reshape(3) for _, t in samples]), axis=0)
    return cv2.Rodrigues(rotation)[0], tvec.reshape(3, 1)


def print_freeze_report(rvec, tvec, quality, seen, layout, pair_rows,
                        K, dist, n_samples):
    camera = camera_position_pool(rvec, tvec)
    print(f"[pose] FROZEN after {n_samples} accepted samples "
          f"({quality['method']}"
          + (" + 16-corner refit" if quality["refined"] else "")
          + f"); centre RMS={quality['centre_rms_px']:.2f} px"
          + (f", corner RMS={quality['corner_rms_px']:.2f} px"
             if np.isfinite(quality["corner_rms_px"]) else ""))
    print(f"[pose]   camera in pool frame: "
          f"({camera[0]:+.3f}, {camera[1]:+.3f}, {camera[2]:+.3f}) m")
    for tag_a, tag_b, measured, expected in pair_rows:
        print(f"[pose]   pair {tag_a}-{tag_b}: PnP={measured:.3f} m "
              f"configured={expected:.3f} m "
              f"error={measured - expected:+.3f} m")
    for tag in sorted(seen):
        world = np.array([[*layout[tag]["xy"], 0.0]])
        projected, _ = cv2.projectPoints(world, rvec, tvec, K, dist)
        error = float(np.linalg.norm(
            projected.reshape(2) - seen[tag]["center_px"]))
        x, y = layout[tag]["xy"]
        print(f"[pose]   ID {tag} at ({x:+.3f},{y:+.3f}) m: "
              f"reprojects {error:.2f} px from its detected centre")


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def project_world(points, rvec, tvec, K, dist):
    """Pool-frame points -> raw image pixels, with a visibility mask.

    Points behind the camera project to nonsense, so they are masked rather
    than drawn.
    """

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    in_camera = (rotation @ points.T).T + np.asarray(
        tvec, dtype=float).reshape(3)
    visible = in_camera[:, 2] > 1e-3
    pixels = np.full((len(points), 2), np.nan)
    if visible.any():
        projected, _ = cv2.projectPoints(
            points[visible], rvec, tvec, K, dist)
        pixels[visible] = projected.reshape(-1, 2)
    return pixels, visible


def draw_world_polyline(frame, path_m, rvec, tvec, K, dist, color,
                        thickness=2, closed=False, samples=24):
    """Draw a pool-frame polyline, densely sampled so it follows the lens.

    A straight line in the pool is a curve in a 116-degree fisheye-ish image;
    projecting only the endpoints would draw a chord across the pool and make
    a good calibration look wrong.
    """

    path = np.asarray(path_m, dtype=float).reshape(-1, 2)
    if len(path) < 2:
        return
    if closed:
        path = np.vstack([path, path[:1]])
    dense = []
    for start, end in zip(path[:-1], path[1:]):
        steps = np.linspace(0.0, 1.0, max(int(samples), 2), endpoint=False)
        dense.append(start + steps[:, None] * (end - start))
    dense.append(path[-1:])
    dense = np.vstack(dense)
    pixels, visible = project_world(
        np.column_stack([dense, np.zeros(len(dense))]),
        rvec, tvec, K, dist)
    for index in range(len(dense) - 1):
        if not (visible[index] and visible[index + 1]):
            continue
        cv2.line(frame,
                 tuple(np.rint(pixels[index]).astype(int)),
                 tuple(np.rint(pixels[index + 1]).astype(int)),
                 color, thickness, cv2.LINE_AA)


def draw_pool_frame(frame, rvec, tvec, K, dist, layout, rect_wh, show_grid):
    """Rectangle, metre grid and pool axes, all through the solved pose."""

    if rect_wh is None:
        corners = np.asarray([entry["xy"] for entry in layout.values()])
        rect_wh = (float(corners[:, 0].max()), float(corners[:, 1].max()))
    width, height = float(rect_wh[0]), float(rect_wh[1])

    if show_grid and DRAW["grid"] and DRAW["grid_step_m"] > 0:
        step = float(DRAW["grid_step_m"])
        x = step
        while x < width - 1e-9:
            draw_world_polyline(frame, [(x, 0.0), (x, height)],
                                rvec, tvec, K, dist, COLOR_GRID, 1)
            x += step
        y = step
        while y < height - 1e-9:
            draw_world_polyline(frame, [(0.0, y), (width, y)],
                                rvec, tvec, K, dist, COLOR_GRID, 1)
            y += step

    if DRAW["rectangle"]:
        draw_world_polyline(
            frame,
            [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)],
            rvec, tvec, K, dist, COLOR_RECT, 3, closed=True)

    if DRAW["axes"]:
        length = float(DRAW["axis_len_m"])
        origin_px, origin_ok = project_world(
            [(0.0, 0.0, 0.0)], rvec, tvec, K, dist)
        tips, tips_ok = project_world(
            [(length, 0.0, 0.0), (0.0, length, 0.0), (0.0, 0.0, length)],
            rvec, tvec, K, dist)
        axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        axis_names = ["+x", "+y", "+z"]
        if origin_ok[0]:
            base = tuple(np.rint(origin_px[0]).astype(int))
            for tip, ok, color, name in zip(tips, tips_ok, axis_colors,
                                            axis_names):
                if not ok:
                    continue
                point = tuple(np.rint(tip).astype(int))
                cv2.arrowedLine(frame, base, point, color, 3, cv2.LINE_AA,
                                tipLength=0.2)
                cv2.putText(frame, name, (point[0] + 6, point[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                            cv2.LINE_AA)


def make_topview_maps(rvec, tvec, K, dist, rect_wh):
    """Remap tables that rectify the pool into a metric bird's-eye image.

    Built once per frozen pose: every destination pixel is a known pool
    coordinate, projected through the same distortion model the PnP used, so
    the warp is exact rather than a homography approximation.
    """

    if rect_wh is None:
        return None
    px_per_m = float(DRAW["topview_px_per_m"])
    margin = float(DRAW["topview_margin_m"])
    width_m, height_m = float(rect_wh[0]), float(rect_wh[1])
    out_w = int(round((width_m + 2 * margin) * px_per_m))
    out_h = int(round((height_m + 2 * margin) * px_per_m))
    if out_w < 8 or out_h < 8 or out_w * out_h > 12_000_000:
        return None
    cols = (np.arange(out_w) + 0.5) / px_per_m - margin
    rows = (height_m + margin) - (np.arange(out_h) + 0.5) / px_per_m
    grid_x, grid_y = np.meshgrid(cols, rows)
    world = np.column_stack([
        grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)])
    pixels, visible = project_world(world, rvec, tvec, K, dist)
    pixels[~visible] = -1.0
    return {
        "map_x": pixels[:, 0].reshape(out_h, out_w).astype(np.float32),
        "map_y": pixels[:, 1].reshape(out_h, out_w).astype(np.float32),
        "size": (out_w, out_h),
        "px_per_m": px_per_m,
        "margin_m": margin,
        "rect_wh": (width_m, height_m),
    }


def topview_pixel(maps, points_m):
    """Pool metres -> top-view pixels."""

    points = np.asarray(points_m, dtype=float).reshape(-1, 2)
    px_per_m = maps["px_per_m"]
    margin = maps["margin_m"]
    height_m = maps["rect_wh"][1]
    return np.column_stack([
        (points[:, 0] + margin) * px_per_m,
        (height_m + margin - points[:, 1]) * px_per_m,
    ])


def render_topview(frame, maps, layout, marks):
    """Warp the live frame to the pool plane and annotate it."""

    view = cv2.remap(frame, maps["map_x"], maps["map_y"], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    width_m, height_m = maps["rect_wh"]
    rect = topview_pixel(maps, [(0, 0), (width_m, 0), (width_m, height_m),
                                (0, height_m)])
    cv2.polylines(view, [np.rint(rect).astype(np.int32).reshape(-1, 1, 2)],
                  True, COLOR_RECT, 2, cv2.LINE_AA)
    if DRAW["grid"] and DRAW["grid_step_m"] > 0:
        step = float(DRAW["grid_step_m"])
        x = step
        while x < width_m - 1e-9:
            a, b = topview_pixel(maps, [(x, 0.0), (x, height_m)])
            cv2.line(view, tuple(np.rint(a).astype(int)),
                     tuple(np.rint(b).astype(int)), COLOR_GRID, 1,
                     cv2.LINE_AA)
            x += step
        y = step
        while y < height_m - 1e-9:
            a, b = topview_pixel(maps, [(0.0, y), (width_m, y)])
            cv2.line(view, tuple(np.rint(a).astype(int)),
                     tuple(np.rint(b).astype(int)), COLOR_GRID, 1,
                     cv2.LINE_AA)
            y += step
    for tag, entry in sorted(layout.items()):
        point = topview_pixel(maps, [entry["xy"]])[0]
        cv2.circle(view, tuple(np.rint(point).astype(int)), 8, COLOR_OK, 2,
                   cv2.LINE_AA)
        cv2.putText(view, str(tag),
                    (int(point[0]) + 10, int(point[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_OK, 2, cv2.LINE_AA)
    for label, position in marks:
        if not np.isfinite(position).all():
            continue
        point = topview_pixel(maps, [position])[0]
        cv2.drawMarker(view, tuple(np.rint(point).astype(int)), COLOR_OTHER,
                       cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        cv2.putText(view, label,
                    (int(point[0]) + 10, int(point[1]) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_OTHER, 2,
                    cv2.LINE_AA)
    origin = topview_pixel(maps, [(0.0, 0.0)])[0]
    cv2.putText(view, "pool (0,0)  +x right  +y up",
                (int(origin[0]) + 12, int(origin[1]) - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                cv2.LINE_AA)
    return view


def open_source():
    """Open the configured camera, or the replay video when one is set."""

    if CAMERA["video"]:
        cap = cv2.VideoCapture(CAMERA["video"])
        if not cap.isOpened():
            raise SystemExit(f"cannot open video {CAMERA['video']!r}")
        return cap, True
    cap = topview.open_camera(
        CAMERA["device"], CAMERA["width"], CAMERA["height"], CAMERA["fps"])
    if not cap.isOpened():
        raise SystemExit(
            f"Cannot open camera {CAMERA['device']!r} — check "
            "ls /dev/v4l/by-id/ and that nothing else has it open")
    return cap, False


def main():
    try:
        layout, rect_wh = build_layout()
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"bad POOL_RECT/TAG_SIZE_M configuration: {exc}")
    describe_layout(layout, rect_wh)

    cap, is_video = open_source()
    grabber = None
    if not is_video:
        grabber = topview.FrameGrabber(cap, name="pool-frame-capture")
        grabber.start()
        if not grabber.wait_first():
            grabber.close()
            cap.release()
            raise SystemExit(
                f"No frames from {CAMERA['device']!r} within 5 s"
                + (f" ({grabber.error})" if grabber.error else ""))

    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    other_obj = np.column_stack([
        make_tag_corners_local(OTHER_TAG_SIZE_M), np.zeros(4)])

    K = dist = geometry_dist = None
    rvec = tvec = None
    quality = {"method": "none", "centre_rms_px": np.nan,
               "corner_rms_px": np.nan, "camera_height_m": np.nan,
               "refined": False}
    samples = []
    drift_count = 0
    warned_model_tags = None
    topview_maps = None
    show_grid = True
    show_topview = bool(DRAW["topview"])

    window = "Pool frame PnP (4 tags)"
    topview_window = "Pool frame — rectified top view"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    writer = None
    record_path = None
    if RECORD["enabled"]:
        record_path = RECORD["path"] or os.path.join(
            _HERE, f"pool_frame_4tag_{time.strftime('%Y%m%d_%H%M%S')}.mp4")

    display_period = 1.0 / max(float(CAMERA["display_fps"]), 0.1)
    next_display = 0.0
    t_prev, fps, last_print = time.time(), 0.0, 0.0
    last_sequence = 0

    try:
        while True:
            if is_video:
                ok, frame = cap.read()
                if not ok:
                    print("[run] end of video")
                    break
            else:
                frame, _host_ns, sequence = grabber.snapshot()
                if sequence == last_sequence:
                    if grabber.stopped:
                        print("Frame grab failed, stopping."
                              + (f" ({grabber.error})" if grabber.error
                                 else ""))
                        break
                    time.sleep(0.001)
                    continue
                last_sequence = sequence

            if K is None:
                K, dist = load_calibration(CAMERA["calib"], frame.shape)
                geometry_dist = (np.zeros_like(dist)
                                 if DISTORTION_MODE == "raw" else dist)
                print(f"[run] {frame.shape[1]}x{frame.shape[0]}, "
                      f"distortion mode {DISTORTION_MODE!r}, "
                      f"{len(layout)} layout tags. Detecting tag36h11 ...")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            seen = detect_layout_tags(corners, ids, layout, K, geometry_dist)
            pair_rows = pair_distance_rows(seen, layout)

            bad_model = sorted(tag for tag, d in seen.items()
                               if not d["model_ok"])
            if bad_model and bad_model != warned_model_tags:
                warned_model_tags = bad_model
                print("[calib] distortion model is NOT invertible at tag(s) "
                      + ", ".join(
                          f"{tag} ({seen[tag]['model_round_trip_px']:.0f} px "
                          "round-trip)" for tag in bad_model)
                      + " — those tags sit outside the radius the calibration "
                      "was fitted on and are excluded. Recalibrate with edge "
                      "coverage (./usb_cam/calibrate.sh) or set "
                      "DISTORTION_MODE='raw'.")

            candidate_r, candidate_t, candidate_q = solve_pool_pose(
                seen, layout, K, geometry_dist)
            reject = gate_pose(candidate_r, candidate_q, pair_rows)

            if not reject and rvec is None:
                samples.append((candidate_r, candidate_t))
                if len(samples) >= int(PNP["init_frames"]):
                    rvec, tvec = average_poses(samples)
                    quality = candidate_q
                    print_freeze_report(
                        rvec, tvec, quality, seen, layout, pair_rows,
                        K, geometry_dist, len(samples))
                    topview_maps = make_topview_maps(
                        rvec, tvec, K, geometry_dist, rect_wh)

            # Frozen-pose health: the tag centres must keep reprojecting onto
            # their detections, otherwise the camera moved.
            live_rms = np.nan
            if rvec is not None and seen:
                live_rms = centre_reprojection_rms(
                    seen, layout, rvec, tvec, K, geometry_dist)
            if (rvec is not None and len(seen) >= int(PNP["min_tags"])
                    and np.isfinite(live_rms) and live_rms > PNP["drift_px"]):
                drift_count += 1
            else:
                drift_count = 0
            if drift_count >= int(PNP["drift_frames"]):
                print(f"[pose] drift {live_rms:.1f} px for {drift_count} "
                      "frames; clearing the pose and re-acquiring")
                rvec = tvec = None
                samples.clear()
                topview_maps = None
                drift_count = 0

            # --- overlays ------------------------------------------------
            good = rvec is not None and drift_count == 0
            color = COLOR_OK if good else COLOR_WAIT
            marks = []

            # The top view warps a copy taken BEFORE the overlays, so the
            # rectified pool shows the water rather than smeared label
            # boxes.  Copying only on a frame that will actually be shown
            # keeps the cost off the capture path.
            display_due = time.monotonic() >= next_display
            clean_frame = (
                frame.copy()
                if display_due and show_topview and topview_maps is not None
                else None)

            if rvec is not None:
                draw_pool_frame(frame, rvec, tvec, K, geometry_dist,
                                layout, rect_wh, show_grid)

            for tag in sorted(seen):
                detection = seen[tag]
                anchor = np.rint(detection["center_px"]).astype(int)
                cv2.polylines(
                    frame,
                    [np.rint(detection["corners_px"]).astype(np.int32)
                     .reshape(-1, 1, 2)],
                    True, color, 2, cv2.LINE_AA)
                cv2.circle(frame, tuple(anchor), 7, color, 2, cv2.LINE_AA)
                if not DRAW["tag_labels"]:
                    continue
                x, y = layout[tag]["xy"]
                lines = [
                    f"ID {tag}  edge {detection['size_m']:.2f} m",
                    f"pool=({x:+.3f},{y:+.3f}) m",
                    (f"cam xyz=({detection['tvec'][0]:+.2f},"
                     f"{detection['tvec'][1]:+.2f},"
                     f"{detection['tvec'][2]:+.2f}) m"
                     if detection["pose_valid"] else "cam xyz=PnP invalid"),
                    (f"tag reproj={detection['reproj_rms_px']:.2f} px"
                     if np.isfinite(detection["reproj_rms_px"])
                     else "tag reproj=unavailable"),
                ]
                if not detection["model_ok"]:
                    lines.append(
                        f"DISTORTION INVALID HERE "
                        f"({detection['model_round_trip_px']:.0f} px "
                        "round-trip) - recalibrate")
                if rvec is not None:
                    world = np.array([[x, y, 0.0]])
                    projected, _ = cv2.projectPoints(
                        world, rvec, tvec, K, geometry_dist)
                    lines.append(
                        "pose reproj="
                        f"{float(np.linalg.norm(projected.reshape(2) - detection['center_px'])):.2f} px")
                topview.draw_text_block(frame, lines, anchor, color)

            if DRAW["other_tags"] and ids is not None:
                for marker_corners, marker_id in zip(corners, ids.flatten()):
                    marker_id = int(marker_id)
                    if marker_id in layout:
                        continue
                    image_corners = marker_corners.reshape(-1, 2).astype(
                        np.float64)
                    centre_px = topview.quadrilateral_center(image_corners)
                    cv2.polylines(
                        frame,
                        [np.rint(image_corners).astype(np.int32)
                         .reshape(-1, 1, 2)],
                        True, COLOR_OTHER, 2, cv2.LINE_AA)
                    lines = [f"ID {marker_id}"]
                    # Same distortion treatment as the pool pose, so the
                    # camera-frame result can be converted into it.
                    ok_pnp, other_r, other_t = cv2.solvePnP(
                        other_obj, image_corners, K, geometry_dist,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if rvec is not None:
                        plane_xy = pixels_to_pool_plane(
                            [centre_px], rvec, tvec, K, geometry_dist)[0]
                        if np.isfinite(plane_xy).all():
                            lines.append(f"pool=({plane_xy[0]:+.2f},"
                                         f"{plane_xy[1]:+.2f}) m")
                            marks.append((f"id {marker_id}", plane_xy))
                        if ok_pnp:
                            pnp_pool = camera_to_pool(
                                np.asarray(other_t).reshape(1, 3),
                                rvec, tvec)[0]
                            lines.append(
                                f"pnp z={pnp_pool[2]:+.2f} m "
                                f"(edge {OTHER_TAG_SIZE_M:.2f} m)")
                    elif ok_pnp:
                        lines.append(
                            f"range={float(np.linalg.norm(other_t)):.2f} m")
                    topview.draw_text_block(
                        frame, lines, np.rint(centre_px).astype(int),
                        COLOR_OTHER)

            if DRAW["pair_table"]:
                for index, (tag_a, tag_b, measured, expected) in enumerate(
                        pair_rows):
                    cv2.putText(
                        frame,
                        f"{tag_a}-{tag_b}: PnP={measured:.3f} m "
                        f"configured={expected:.3f} m "
                        f"err={measured - expected:+.3f} m",
                        (10, 62 + 26 * index), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, color, 2, cv2.LINE_AA)
                if reject and rvec is None:
                    cv2.putText(
                        frame, "reject: " + "; ".join(reject),
                        (10, 62 + 26 * max(len(pair_rows), 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLOR_WAIT, 2,
                        cv2.LINE_AA)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            if rvec is not None:
                camera = camera_position_pool(rvec, tvec)
                status = (f"POSE OK ({quality['method']}"
                          + (" +corners" if quality["refined"] else "")
                          + f") | cam=({camera[0]:+.2f},{camera[1]:+.2f},"
                          f"{camera[2]:+.2f}) m | live RMS="
                          + (f"{live_rms:.1f} px" if np.isfinite(live_rms)
                             else "n/a")
                          + f" | tags:{len(seen)}/{len(layout)} {fps:.0f} fps")
            else:
                status = (f"SOLVING {len(samples)}/{int(PNP['init_frames'])} "
                          f"({candidate_q['method']}) | "
                          f"tags:{len(seen)}/{len(layout)} {fps:.0f} fps")
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, 2, cv2.LINE_AA)

            if now - last_print >= 1.0 / max(float(CAMERA["print_rate"]), 0.1):
                last_print = now
                print(f"[run] {status}")

            if record_path is not None:
                if writer is None:
                    writer = topview.WallClockVideoWriter(
                        record_path, RECORD["fps"],
                        (frame.shape[1], frame.shape[0]))
                    writer.submit(frame)
                    writer.start()
                    print(f"[record] {record_path} at {RECORD['fps']:.0f} fps,"
                          " overlays included")
                if writer.error is not None:
                    raise RuntimeError(f"MP4 encoder failed: {writer.error}")
                writer.submit(frame)

            if display_due:
                next_display = time.monotonic() + display_period
                cv2.imshow(window, frame)
                if show_topview and topview_maps is not None:
                    view = render_topview(
                        clean_frame if clean_frame is not None else frame,
                        topview_maps, layout, marks)
                    cv2.namedWindow(topview_window, cv2.WINDOW_NORMAL)
                    cv2.moveWindow(topview_window,
                                   *DRAW["topview_window_xy"])
                    cv2.imshow(topview_window, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    print("[pose] manual reset; re-acquiring")
                    rvec = tvec = None
                    samples.clear()
                    topview_maps = None
                    drift_count = 0
                if key == ord("g"):
                    show_grid = not show_grid
                if key == ord("v"):
                    show_topview = not show_topview
                    if not show_topview:
                        cv2.destroyWindow(topview_window)

    except KeyboardInterrupt:
        print("\n[run] interrupted")
    finally:
        if grabber is not None:
            grabber.close()
        cap.release()
        if writer is not None:
            writer.close()
            print(f"[record] wrote {writer.frames_written} frames to "
                  f"{record_path}")
        if grabber is not None:
            print(f"[run] camera delivered {grabber.frames_read} frames")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
