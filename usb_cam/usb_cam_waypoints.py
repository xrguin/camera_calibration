#!/usr/bin/env python3
"""Waypoint-profile sanity preview in the overhead-camera world frame.

Same reference-board calibration flow as ``usb_cam_apriltag.py`` (tags from
REF_RECT, frozen homography, drift monitor), but instead of labelling
arbitrary tags the frozen map is used to DRAW a planned waypoint pattern on
the live camera image: numbered waypoints, the smooth ghost path a 3rd-order
Fossen reference model would fly through them, and the wall-margin band.

The waypoints and ghost parameters are the EDITABLE VARIABLES below — type
in candidate patterns and look at the live view to judge whether they make
sense in the real tank.  World frame matches track_rov_topview.py and the
combined_view overhead recorder: origin at the near-left reference tag,
+x along the camera-side edge, +y into the pool, metres — so a pattern
accepted here transfers unchanged to the runtime.

On startup (and in --offline mode, which needs no camera) the script also
simulates the ghost through one full waypoint cycle and prints a per-leg
report: leg length, peak ghost speed, peak body-frame u / v_starboard at
the constant crab heading, wall clearance — each checked against the caps
below.  --offline writes ``waypoint_preview.png`` (footprint + velocity
traces) next to this script instead of opening the camera.

Usage:
    python3 usb_cam/usb_cam_waypoints.py             # live camera overlay
    python3 usb_cam/usb_cam_waypoints.py --offline   # PNG + report only

Controls (live mode): q or Esc quit, r resets the frozen calibration.
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import track_rov_topview as topview          # noqa: E402
import usb_cam_apriltag as ref_cam           # noqa: E402
import pool_layout_4tag as pool_layout       # noqa: E402

# ===========================================================================
# Waypoint pattern — EDIT THESE VARIABLES DIRECTLY
#
# Waypoints are (x, y) metres in the pool frame (origin at the near-left
# tag 100, +x along the camera-side edge to 102, +y into the pool).  The
# frame is measured from the tags in pool_layout_4tag.py and spans
# 4.427 x 2.117 m as of 2026-08-04.
#
# Of everything in this block, the RUNTIME consumes WAYPOINTS_M,
# WAYPOINT_DEPTHS_M, POOL_DEPTH_MAX_M, HEADING_DEG and ARRIVAL_TOL_M --
# those are what combined_view's _waypoint_flags() turns into --mpc-*
# arguments.  The cruise speeds are NOT here: each controller arm declares
# its own in control_v2/waypoint_policy.py and hands it to the functions
# below, so the feasibility gate always checks the speed that will fly.
# The rest
# (the ghost profile, corner fillet, hold, loop flag) drives the preview and
# the feasibility report only: line-of-sight guidance replaced the
# pre-computed ghost on 2026-08-02, so nothing downstream flies it.
# Each entry is (x, y) or (x, y, heading_deg) — all entries must agree.
#
# The optional third element (added 2026-08-05) is the heading the MPC
# heading hold TRACKS while flying toward that waypoint, degrees CCW from
# +x (0 = bow toward +x, matching HEADING_DEG's convention).  Values are
# wrapped, so 720 and 0 are the same setpoint and the hold always rotates
# the SHORT way (a deliberate >180 deg rotation direction cannot be
# expressed).  The setpoint switches the moment the target waypoint
# switches, so the vehicle rotates en route.  Translation is unaffected:
# line-of-sight u/v guidance re-rotates through the measured heading every
# cycle, whatever the bow is doing.
#
# Drop the third element from every entry to restore the 2026-08-04
# behavior: constant crab flight with the hold locking the engage heading.
# Rectangle pattern (2026-08-04 .. 2026-08-17), kept whole for restoration:
# comment the figure-8 assignments below back out in favour of these.
#
# WAYPOINTS_M = [
#     (1.00, 0.40, 0),
#     (3.30, 0.40, 90),
#     (3.30, 1.20, 180),
#     (1.00, 1.20, 270),
# ]
# WAYPOINT_DEPTHS_M = [0.25, 0.10, 0.2, 0.0]    # rectangle, no pump
# WAYPOINT_DEPTHS_M = [0.0, 0.10, 0.25, 0.0]    # rectangle, pump tests


def figure8_pattern(count=16, center_m=(2.15, 0.80), half_width_m=1.15,
                    half_height_m=0.40, depth_mid_m=0.19, depth_amp_m=0.09):
    """Tangent-heading figure-8 with per-lobe depth cycles (2026-08-17).

    An identification pattern, not a mission: the lemniscate of Gerono
    (x = sin t, y = sin 2t, scaled) sweeps heading continuously through
    both turn directions -- the east lobe clockwise, the west lobe
    counterclockwise -- so yaw data is sign-symmetric within one lap and
    the RLS excitation spans command space in a single run instead of
    only across many.  The crossing revisits the same position with two
    different velocity histories each lap, which is what separates
    position-dependent effects from history-dependent ones; the west
    lobe tip (1.00, 0.80) sits on the pump-jet axis (apex (1.0, 0),
    +y) at one of its hold ranges, so the identical pattern serves the
    pump-on discrimination runs later with the calm east lobe as the
    in-run control.

    Geometry.  The footprint equals the rectangle pattern's envelope
    (x 1.00..3.30, y 0.40..1.20), so wall clearance is boundary-equal
    to the 0.40 m margin exactly as before.  Sampling is uniform in the
    curve parameter, NOT arc length, on purpose: parameter-uniform
    points concentrate where curvature peaks, which puts the shortest
    legs where the tangent turns fastest.  At count=16 the legs run
    0.28..0.52 m and consecutive hold headings step at most ~52 deg --
    about 0.27 rad/s at the 0.12 m/s baseline cruise, inside the turn
    budget the preflight checks.  The shortest leg is 1.84x the 0.15 m
    arrival radius, tighter than the rectangle's but different in kind
    from the 20260802_194321 stall: that run had to ENTER a terminal
    radius, while a curve chain crosses each radius transversally on
    the way to the next point.

    Heading is the analytic path tangent at the DESTINATION waypoint
    (degrees CCW from +x): the hold rotates the short way toward it en
    route, so the bow sweeps the tangent continuously instead of
    holding a crab angle.  Translation is unaffected -- line-of-sight
    u/v guidance re-rotates through the measured heading every cycle.

    Depth rides cos(2*theta): exactly one cycle per lobe, so both
    lobes carry an IDENTICAL mean depth (0.19 m) and the refraction
    scale error that depth injects into the overhead planar velocities
    (uncorrected tag-plane metres; s = (H + d/1.33)/H) is decorrelated
    from turn direction at first order -- pass --top-rov-depth 0.19 so
    the residual is the +/-3% around mid-depth rather than the full
    +/-9%.  The deepest point (0.28 m) lands at the crossing under the
    camera centre where the planar displacement error is smallest, the
    shallowest (0.10 m) at the lobe tips near the frame edges where
    marker contrast is worst.  The whole plan spans 0.10..0.28 m: the
    2026-08-17 envelope decision is (0, 0.3], and the ~0.10 m depth
    overshoot the loop shows on target changes stays clear of the
    0.45 m floor abort.  Consecutive targets differ by at most 0.064 m,
    a ~0.02 m/s heave reference en route -- small, but persistent and
    sign-alternating, which is the excitation the unmasked heave rows
    never had on the flat patterns (the adaptive_all arms learned heave
    from noise on 20260811_180314 precisely because w_ref was ~0).
    """

    points = []
    depths = []
    for index in range(count):
        theta = 2.0 * math.pi * index / count
        x = center_m[0] + half_width_m * math.sin(theta)
        y = center_m[1] + half_height_m * math.sin(2.0 * theta)
        dx = half_width_m * math.cos(theta)
        dy = 2.0 * half_height_m * math.cos(2.0 * theta)
        heading = math.degrees(math.atan2(dy, dx)) % 360.0
        points.append((round(x, 4), round(y, 4), round(heading, 2)))
        depths.append(
            round(depth_mid_m + depth_amp_m * math.cos(2.0 * theta), 4))
    return points, depths


WAYPOINTS_M, WAYPOINT_DEPTHS_M = figure8_pattern()
WAYPOINT_LOOP = True         # return to the first waypoint at the end

# Constant crab heading of the vehicle while flying the pattern, degrees
# CCW from +x.  0 deg = bow toward +x.  At 180 deg (bow toward -x) the long
# legs of the default box are still pure surge — alternating ahead/astern —
# and the short legs pure sway, with signs flipped relative to 0 deg.
# With per-waypoint headings above, this remains the ghost-preview and
# speed-ellipse BASE only; the runtime ellipse follows the measured
# heading instead.
HEADING_DEG = 180.0


def waypoints_xy():
    """The pattern's (x, y) pairs, whatever width WAYPOINTS_M uses."""

    return [(float(p[0]), float(p[1])) for p in WAYPOINTS_M]


def waypoint_headings_deg():
    """Per-waypoint hold headings in degrees, or None when not declared."""

    widths = {len(p) for p in WAYPOINTS_M}
    if widths == {2}:
        return None
    if widths != {3}:
        raise SystemExit(
            "WAYPOINTS_M entries must be all (x, y) or all "
            "(x, y, heading_deg)")
    return [float(p[2]) % 360.0 for p in WAYPOINTS_M]

# Depth targets stay parallel to WAYPOINTS_M so the established (x, y) and
# (x, y, heading_deg) formats remain backward compatible.  Each value is an
# absolute positive-down depth below the surface.  The target switches with
# the corresponding XY waypoint; waypoint completion is handled by the
# runtime's XY/depth/heading settle policy.

# WAYPOINT_DEPTHS_M is produced by figure8_pattern() above, parallel to
# WAYPOINTS_M entry for entry.  The rectangle depth plans are preserved in
# the commented block with the rectangle pattern.

def waypoint_depths_m():
    """Absolute positive-down target depth for each waypoint."""

    depths = [float(value) for value in WAYPOINT_DEPTHS_M]
    if len(depths) != len(WAYPOINTS_M):
        raise SystemExit(
            "WAYPOINT_DEPTHS_M must contain exactly one depth for each "
            f"WAYPOINTS_M entry ({len(depths)} != {len(WAYPOINTS_M)})")
    return depths


# Surface-engage protocol (2026-08-12): the operator brings the vehicle to
# the WATER SURFACE before pressing m, and the pressure captured at
# engagement IS the surface reference — every waypoint depth below is a
# true positive-down depth from that surface.  The old DEPTH_HOLD_M
# declared-engage-depth constant is gone: it silently shifted the whole
# depth scale by 0.30 m whenever engagement did not actually happen at the
# declared depth (all of 2026-08-11's surface engagements did exactly
# that).  The launcher warns at engagement when the session display gauge
# says the vehicle was not at the surface.

# Absolute waypoint-depth envelope -- FLOOR ONLY since 2026-08-13.
#
# The shallow bound is gone.  It was there to catch a vehicle drifting up
# to the surface mid-mission, but in practice it fired on the up-swing of
# the depth loop's own oscillation rather than on any fault: run
# 20260813_154340 overshot a 0.10 m target to 0.20 m, rebounded past the
# target, and tripped the 0.05 m floor on the way back -- ending a run
# that was not in trouble.  A shallow bound only works if it is smaller
# than the depth loop's overshoot, and at these targets it cannot be.
#
# The floor stays: it is the bound that protects the vehicle and the
# pressure sensor, and the vehicle is never legitimately below it.
# Depth is still recorded every cycle (mpc_depth_m), so surface
# excursions remain fully visible in the log -- what is gone is the
# abort, not the evidence.  Pass --mpc-depth-min-m to put a shallow
# bound back for one run.
POOL_DEPTH_MAX_M = 0.45

# Ghost speed profile — TRAPEZOIDAL, per axis.
#
# This replaced a 3rd-order critically damped reference model on
# 2026-08-02.  That model produced a bell-shaped speed curve, zero at both
# ends of every leg with long low-speed tails, and the tails are exactly
# where the thruster dead zone lives.  Measured on the reference that run
# 20260730_183158 actually flew:
#
#     axis   peak      MEDIAN     mean
#     u      0.1377    0.0211     0.0440   m/s
#     v      0.0655    0.0006     0.0148   m/s
#
# so 89.4% of its sway commands sat below the ~0.084 effective knee and
# produced no thrust at all.  Only ~57% of the cycle had |u| > 0.02 m/s.
# A trapezoid holds a flat cruise instead, keeping the command clear of
# the knee for nearly the whole leg.
#
# Cruise values come from the measured envelope (see
# control_v2/SWAY_MEASURED_FACTS.md): gain above the knee is ~1.25 m/s per
# unit effective on surge and ~1.15 on sway, so
#
#     surge 0.12 m/s -> 0.12/1.25 + 0.084 = 0.180 effective
#     sway  0.10 m/s -> 0.10/1.15 + 0.084 = 0.171 effective
#
# both roughly twice the knee, both leaving room for the 0.05 m/s
# correction inside the 0.25 box, and both under the measured single-axis
# maxima of 0.207 / 0.191 m/s.
# The cruise speeds themselves moved to control_v2/waypoint_policy.py on
# 2026-08-14, so each controller arm declares the speed it flies and there
# is exactly one definition of it.  Nothing here holds a copy: every
# function below takes cruise_u/cruise_v as an argument, and the CLI names
# an arm (--arm) to obtain them.  The arithmetic above is the worked
# example for the arms' current 0.12/0.10 and is what cruise_feasibility()
# recomputes for whatever an arm asks for.
ACCEL_M_S2 = 0.08            # ~1.5 s to cruise, about 3 vehicle time constants

# Corner fillet radius.  A square corner forces surge and sway both through
# zero -- back into the dead zone twice per corner -- so the corners are
# rounded and the speed never returns to zero between the start and the end
# of a lap.  Through a fillet BOTH axes run near cruise at once, which needs
# planar budget: s_u + s_v = 0.180 + 0.171 = 0.351 plus yaw, infeasible
# against the old 0.35 and comfortable against the 0.60 a waypoint run now
# launches with.
CORNER_RADIUS_M = 0.15
PATH_SAMPLE_DS_M = 0.005     # arc-length resolution of the generated path

GHOST_DT = 0.05              # 20 Hz, matching the control loop
# Switch to the next waypoint inside this radius.
#
# Raised from 0.10 after run 20260802_194321, which closed to 0.119 m of
# waypoint 2 and never got inside 0.10, so it never advanced past it and
# spent 40 s circling.  Flying through at 0.10 m/s the vehicle carries
# ~0.05 m past on its ~0.5 s time constant, and the overhead fix has its
# own centimetres of noise, so a radius near the miss distance stalls the
# pattern.  0.15 m is still small against the 0.80 m shortest leg.
ARRIVAL_TOL_M = 0.1
HOLD_S = 0.2                # dwell at the END of a lap only
LEG_TIMEOUT_S = 60.0

# Checks: MPC reference caps and wall clearance.
U_MAX = 0.20                 # body surge cap, m/s (vel_mpc default u_ref)
V_MAX = 0.20                 # body sway cap, m/s
WALL_MARGIN_M = 0.40         # required clearance to the reference-rect edge
# Reserved authority for the backstepping position correction: the
# feasibility gate requires peak_feedforward + this <= U_MAX/V_MAX so the
# clamped Lambda*e_pos term never pushes the reference over the caps.
CORRECTION_HEADROOM_M_S = 0.05

# U_MAX/V_MAX above are KINEMATIC caps.  They say nothing about whether the
# thrusters can produce the demanded velocity, and on 2026-07-30 that gap
# was an order of magnitude on sway: with the then-fixed 0.10 effective sway
# box and an 0.085 thruster deadband the vehicle could deliver at most about
# 0.019 m/s, while this gate happily passed a 0.065 m/s sway leg.  The
# actuator check below closes that gap by asking the control model what
# effective input each leg needs and comparing it against the real box.
#
# Planning PilotGain, matching control_v2.keyboard_control_v2's
# EXPECTED_PILOT_GAIN.  safety * PilotGain is a hard ceiling on effective
# command -- a full keyboard press reaches exactly 0.25 at 0.5/0.5 -- and the
# MPC boxes are bounded so their deadband-compensated command fits it.
PLANNING_PILOT_GAIN = 0.5
PLANNING_SAFETY = 0.5
# ===========================================================================


def actuator_report(u, v_starboard, legs):
    """Per-leg effective-input demand against the real vel_mpc4 limits.

    Returns ``(rows, warnings)``.  Imported lazily so the camera preview and
    the pattern editor keep working without the control stack or its model
    archive present.
    """

    try:
        from control_v2 import thruster_deadband
        from control_v2.vel_mpc4_controller import (
            PLANAR_BUDGET,
            S_MAX_FLOOR,
            S_MAX_MARGIN,
            model_dc_input,
        )
        from control_v2.vel_mpc4_runtime import PLANAR_COMMANDED_L1_MAX
        from control_v2.vel_mpc_controller import GaussianEDMDcModel
    except Exception as error:                    # pragma: no cover
        print(f"[profile] actuator check skipped: {error}")
        return [], []

    band = thruster_deadband.as_deadband(None)
    dc_input = np.linalg.inv(model_dc_input(GaussianEDMDcModel()))
    # Same box derivation the launcher uses: bounded so the compensated
    # command fits safety * PilotGain, which is all the host can express.
    ceiling = PLANNING_SAFETY * PLANNING_PILOT_GAIN
    cap = np.minimum(
        np.array([0.25, 0.25, 0.20, 0.25]),
        thruster_deadband.invert(np.full(4, ceiling), band))
    floor = np.minimum(np.asarray(S_MAX_FLOOR, dtype=float), cap)

    rows, warnings = [], []
    for start, stop, source, target in legs:
        seg = slice(start, stop)
        peak = np.array([
            float(np.max(np.abs(u[seg]))) + CORRECTION_HEADROOM_M_S,
            float(np.max(np.abs(v_starboard[seg]))) + CORRECTION_HEADROOM_M_S,
            0.0, 0.0,
        ])
        required = np.abs(dc_input @ peak)
        s_max = np.clip(required * S_MAX_MARGIN + band, floor, cap)
        commanded = thruster_deadband.compensate(required, band)
        planar_l1 = float(commanded[[0, 1, 3]].sum())
        rc_l1 = planar_l1 / PLANNING_PILOT_GAIN

        flags = []
        for axis in (0, 1):
            if required[axis] > s_max[axis] + 1e-9:
                flags.append(
                    f"{'uv'[axis]} needs {required[axis]:.3f} eff > "
                    f"{s_max[axis]:.3f} box")
        if float(required[[0, 1, 3]].sum()) > PLANAR_BUDGET + 1e-9:
            flags.append(f"planar demand > {PLANAR_BUDGET:g} budget")
        if planar_l1 > PLANAR_COMMANDED_L1_MAX + 1e-9:
            flags.append(
                f"compensated planar {planar_l1:.3f} > "
                f"{PLANAR_COMMANDED_L1_MAX:g}")
        if rc_l1 > 1.0 + 1e-9:
            flags.append(
                f"rc L1 {rc_l1:.2f} > 1 at PilotGain "
                f"{PLANNING_PILOT_GAIN:g}")
        rows.append((source, target, required, s_max, commanded, rc_l1,
                     flags))
        warnings.extend(
            f"leg {source}->{target}: {flag}" for flag in flags)
    return rows, warnings


# Measured plant, from control_v2/SWAY_MEASURED_FACTS.md.  Used instead of
# the control model for the FLOOR check below, because the model is ~35%
# optimistic on sway and the floor is about what the thrusters physically
# do, not about what the MPC believes.
MEASURED_KNEE_EFF = 0.084        # fitted knee (bootstrap 0.065-0.097)
MEASURED_KNEE_MARGIN = 0.030     # stay this clear of it: within ~0.03 the
                                 # gain is ~10x uncertain
MEASURED_GAIN_U = 1.25           # m/s per unit effective, above the knee
MEASURED_GAIN_V = 1.15


# Which arm's speeds the standalone preview describes when none is named.
# Any arm would do; this one is the fixed Gaussian reference arm.
DEFAULT_PREVIEW_ARM = "vel_mpc4_waypoint"


def arm_cruise(arm):
    """The cruise speeds declared by a controller arm module.

    The speeds live with the arm (``control_v2/<arm>.py``'s ``TUNING``),
    so the preview has to name one rather than keep a copy.  Imported
    lazily, like the S_MAX_FLOOR import below, to keep usb_cam usable
    without control_v2 on the path.

    Run as a plain script, ``usb_cam/`` is on sys.path but the repository
    root is not, so the root is added here -- otherwise ``control_v2``
    resolves only under ``python -m``.
    """

    import importlib
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        module = importlib.import_module(f"control_v2.{arm}")
    except ImportError as error:
        raise SystemExit(
            f"--arm {arm}: no such controller module (control_v2/{arm}.py). "
            f"{error}")
    try:
        tuning = module.TUNING
    except AttributeError:
        raise SystemExit(
            f"--arm {arm}: control_v2/{arm}.py declares no TUNING, so it is "
            "not a waypoint arm")
    return float(tuning.cruise_u), float(tuning.cruise_v)


def cruise_feasibility(cruise_u, cruise_v):
    """Can the thrusters actually deliver the waypoint cruise speeds?

    Line-of-sight guidance commands one speed for the whole run, so the
    feasibility question collapses from "is every point of a trajectory
    flyable" to "are these two numbers flyable".  Both bounds matter:

    * the FLOOR -- a cruise below the thruster knee produces no thrust at
      all.  This is the check the old gate lacked: it tested only maxima,
      so it passed run ``20260730_183158``'s 0.065 m/s sway leg while the
      vehicle could deliver nothing;
    * the CEILING -- the command must fit the 0.25 axis box, with room for
      the turn, where both axes run at once.

    Returns ``(rows, warnings)``.
    """

    cruise_u = float(cruise_u)
    cruise_v = float(cruise_v)
    try:
        from control_v2.vel_mpc4_controller import S_MAX_FLOOR
    except Exception as error:                    # pragma: no cover
        print(f"[preflight] actuator check skipped: {error}")
        return [], []

    box = float(np.min(np.asarray(S_MAX_FLOOR, dtype=float)[[0, 1]]))
    floor = MEASURED_KNEE_EFF + MEASURED_KNEE_MARGIN
    rows, warnings = [], []
    demand = {}
    for name, cruise, gain in (("surge", cruise_u, MEASURED_GAIN_U),
                               ("sway", cruise_v, MEASURED_GAIN_V)):
        required = cruise / gain + MEASURED_KNEE_EFF
        demand[name] = required
        flags = []
        if required < floor:
            flags.append(
                f"{cruise:.3f} m/s needs only {required:.3f} effective, "
                f"inside the dead zone (knee {MEASURED_KNEE_EFF:.3f} "
                f"+ {MEASURED_KNEE_MARGIN:.3f} margin) -- no thrust")
        if required > box + 1e-9:
            flags.append(
                f"{cruise:.3f} m/s needs {required:.3f} effective > "
                f"{box:.3f} box")
        rows.append((name, cruise, required, flags))
        warnings.extend(f"{name} cruise: {flag}" for flag in flags)

    # A turn runs both axes at once; the dead zone is paid twice.
    turn = demand["surge"] + demand["sway"]
    rows.append(("turn (both axes)", float("nan"), turn, []))
    return rows, warnings, turn


def preflight_report(cruise_u, cruise_v, waypoints=None,
                     margin_m=WALL_MARGIN_M, depths_m=None,
                     min_depth_m=None,
                     max_depth_m=POOL_DEPTH_MAX_M,
                     heading_policy="measured"):
    """Print the waypoint preflight and abort the launch if infeasible.

    Called from ``combined_view.keyboard_stabilize_topview`` before a
    waypoint run so an unflyable pattern stops on the bench, not in the
    water.  Raises ``SystemExit`` on any warning.
    """

    heading_policy = str(heading_policy)
    if heading_policy not in ("measured", "reference_rollout"):
        raise ValueError(
            "heading_policy must be 'measured' or 'reference_rollout'")
    default_pattern = waypoints is None
    points = np.asarray(
        waypoints_xy() if default_pattern else waypoints,
        dtype=float)[:, :2]
    cruise_u = float(cruise_u)
    cruise_v = float(cruise_v)
    # None means "no shallow bound"; only the floor is enforced.
    min_depth_m = None if min_depth_m is None else float(min_depth_m)
    max_depth_m = float(max_depth_m)
    if depths_m is None:
        # Custom XY-only preflight callers retain a useful default instead of
        # being coupled to the shipped pattern's list length: the middle of
        # the allowed envelope.
        depths = (waypoint_depths_m() if default_pattern else
                  [0.5 * ((min_depth_m or 0.0) + max_depth_m)] * len(points))
    else:
        depths = [float(value) for value in depths_m]

    headings = waypoint_headings_deg() if default_pattern else None
    print(f"[preflight] {len(points)} waypoints, heading {HEADING_DEG:g} deg, "
          f"cruise u={cruise_u:g} v={cruise_v:g} m/s")
    if headings is not None:
        heading_label = (
            "per-waypoint hold headings"
            if heading_policy == "measured"
            else "per-waypoint pose headings"
        )
        turn_note = (
            "hold rotates the short way"
            if heading_policy == "measured"
            else "reference uses the shortest yaw branch"
        )
        ellipse_note = (
            "ellipse follows the measured heading"
            if heading_policy == "measured"
            else (
                "full-MPC rates rotate at each reference-yaw endpoint; "
                "the horizon uses the minimum directional ellipse")
        )
        print(f"[preflight]   {heading_label} "
              + " ".join(f"{h:g}" for h in headings)
              + f" deg ({turn_note}; {ellipse_note})")
    warnings = []
    if len(depths) != len(points):
        warnings.append(
            f"depth plan has {len(depths)} entries for {len(points)} "
            "waypoints")
    floor = 0.0 if min_depth_m is None else min_depth_m
    if (not np.isfinite(max_depth_m) or max_depth_m <= 0.0
            or (min_depth_m is not None
                and (not np.isfinite(min_depth_m)
                     or min_depth_m < 0.0
                     or min_depth_m >= max_depth_m))):
        warnings.append(
            "depth envelope must be finite, nonnegative, and ordered "
            f"(got {floor:g}..{max_depth_m:g} m)")
    for index, depth in enumerate(depths):
        if not np.isfinite(depth):
            warnings.append(f"waypoint {index} depth is not finite")
        elif not floor <= depth <= max_depth_m:
            warnings.append(
                f"waypoint {index} depth {depth:.3f} m is outside "
                f"[{floor:.3f}, {max_depth_m:.3f}] m")
    print("[preflight]   surface-engage protocol: bring the vehicle to "
          "the WATER SURFACE before pressing m (the engage pressure "
          "capture is the 0 m depth datum)")
    envelope_note = (
        f"floor abort at {max_depth_m:.3f} m; no shallow bound"
        if min_depth_m is None
        else f"allowed {min_depth_m:.3f}..{max_depth_m:.3f} m")
    print(f"[preflight]   waypoint depths  "
          + " ".join(f"{depth:.3f}" for depth in depths)
          + f" m ({envelope_note})")

    rows, cruise_warnings, turn = cruise_feasibility(cruise_u, cruise_v)
    warnings.extend(cruise_warnings)
    for name, cruise, required, flags in rows:
        speed = "  --  " if not np.isfinite(cruise) else f"{cruise:.3f}"
        print(f"[preflight]   {name:16s} speed {speed} m/s  "
              f"effective {required:.3f}"
              + ("  " + "; ".join(flags) if flags else ""))

    # Wall clearance in the reference rectangle.  Read from the configured
    # pool layout rather than hardcoded, so a re-measurement of the tag
    # rectangle actually reaches this preflight gate: a stale extent here
    # would clear a pattern that runs into the wall.
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    extent = np.asarray(
        pool_layout.rect_wh() or rect_extent(ref_cam.build_ref_world()),
        dtype=float)
    clearance = float(min(lo.min(), float(np.min(extent - hi))))
    print(f"[preflight]   wall clearance   {clearance:.2f} m "
          f"(margin {margin_m:.2f})")
    if clearance < margin_m:
        warnings.append(
            f"wall clearance {clearance:.2f} m < {margin_m:.2f} m margin")

    try:
        from control_v2.vel_mpc4_controller import PLANAR_BUDGET
        from combined_view.keyboard_stabilize_topview import (
            WAYPOINT_PLANAR_BUDGET,
        )
    except Exception:                             # pragma: no cover
        PLANAR_BUDGET, WAYPOINT_PLANAR_BUDGET = 0.35, 0.60
    print(f"[preflight]   turn demand {turn:.3f} planar, budget "
          f"{WAYPOINT_PLANAR_BUDGET:g} (step runs use {PLANAR_BUDGET:g})")
    if turn > WAYPOINT_PLANAR_BUDGET + 1e-9:
        warnings.append(
            f"turn needs {turn:.3f} planar > "
            f"{WAYPOINT_PLANAR_BUDGET:g} budget")

    if warnings:
        for warning in warnings:
            print(f"[preflight] REFUSED: {warning}")
        raise SystemExit(
            "waypoint preflight failed; edit the pattern or the cruise "
            "speeds in usb_cam/usb_cam_waypoints.py")
    print("[preflight] OK")
    return rows


def _corner_fillet(prev_pt, vertex, next_pt, radius):
    """Circular fillet at ``vertex``: (entry, exit, centre, sign, turn).

    ``sign`` is +1 for a left turn, -1 for a right one, and ``turn`` is the
    direction change in radians.  Returns ``None`` when the corner is
    straight, a reversal, or too tight for the radius to fit — the caller
    then leaves it square.
    """

    d_in = vertex - prev_pt
    d_out = next_pt - vertex
    len_in = float(np.linalg.norm(d_in))
    len_out = float(np.linalg.norm(d_out))
    if len_in < 1e-9 or len_out < 1e-9:
        return None
    d_in = d_in / len_in
    d_out = d_out / len_out
    turn = float(np.arccos(np.clip(float(np.dot(d_in, d_out)), -1.0, 1.0)))
    if turn < 1e-3 or turn > math.pi - 1e-3:
        return None
    # Tangent length from the vertex for an interior angle of (pi - turn).
    offset = radius / math.tan((math.pi - turn) / 2.0)
    if offset > 0.5 * min(len_in, len_out):
        return None                      # fillet would eat the whole leg
    cross = float(d_in[0] * d_out[1] - d_in[1] * d_out[0])
    sign = 1.0 if cross > 0 else -1.0
    normal = sign * np.array([-d_in[1], d_in[0]])
    entry = vertex - offset * d_in
    return entry, vertex + offset * d_out, entry + radius * normal, sign, turn


def _rounded_path(sequence, radius, ds):
    """Arc-length samples of a polyline with its interior corners rounded.

    Returns ``(xy, leg_of_sample)``.  A fillet is split at its midpoint
    between the incoming and outgoing leg, so the per-leg peak table
    attributes each half to the leg that is actually driving it.
    """

    points = [np.asarray(p, dtype=float) for p in sequence]
    fillets = [None] * len(points)
    for k in range(1, len(points) - 1):
        fillets[k] = _corner_fillet(
            points[k - 1], points[k], points[k + 1], radius)

    xy, leg_of = [], []

    def _line(start, end, leg):
        length = float(np.linalg.norm(end - start))
        count = max(int(round(length / ds)), 1)
        for i in range(count):
            xy.append(start + (end - start) * (i / count))
            leg_of.append(leg)

    cursor = points[0]
    for k in range(1, len(points)):
        fillet = fillets[k] if k < len(points) - 1 else None
        if fillet is None:
            _line(cursor, points[k], k - 1)
            cursor = points[k]
            continue
        entry, exit_pt, centre, sign, turn = fillet
        _line(cursor, entry, k - 1)
        start_angle = math.atan2(entry[1] - centre[1], entry[0] - centre[0])
        arc_length = radius * turn
        count = max(int(round(arc_length / ds)), 1)
        for i in range(count):
            fraction = i / count
            angle = start_angle + sign * turn * fraction
            xy.append(centre + radius * np.array(
                [math.cos(angle), math.sin(angle)]))
            leg_of.append(k - 1 if fraction < 0.5 else k)
        cursor = exit_pt
    xy.append(points[-1])
    leg_of.append(len(points) - 2)
    return np.asarray(xy), np.asarray(leg_of, dtype=int)


def _speed_limit(direction, cruise_u, cruise_v, heading_deg):
    """Fastest path speed that keeps BOTH body axes at or under cruise.

    The ellipse limit: a leg that is pure surge gets ``cruise_u``, pure
    sway gets ``cruise_v``, and anything diagonal gets whatever keeps the
    more demanding axis at its own cruise.
    """

    psi = math.radians(heading_deg)
    u = direction[0] * math.cos(psi) + direction[1] * math.sin(psi)
    v = direction[0] * math.sin(psi) - direction[1] * math.cos(psi)
    scale = math.hypot(u / cruise_u, v / cruise_v)
    return 1.0 / scale if scale > 1e-9 else cruise_u


def simulate_ghost(waypoints, *, cruise_u, cruise_v,
                   loop=WAYPOINT_LOOP, dt=GHOST_DT,
                   accel=ACCEL_M_S2, radius=CORNER_RADIUS_M,
                   heading_deg=HEADING_DEG, hold_s=HOLD_S,
                   ds=PATH_SAMPLE_DS_M, tol=ARRIVAL_TOL_M,
                   timeout_s=LEG_TIMEOUT_S):
    """Fly the ghost once round the pattern on a trapezoidal speed profile.

    The path is the waypoint polyline with its interior corners rounded, so
    the ghost never stops between the start and the end of a lap.  Speed is
    planned along arc length: a per-point cap from the body-axis cruise
    limits, then a forward/backward acceleration pass, which is what makes
    the profile trapezoidal rather than bell-shaped.

    Starts and ends at rest on waypoint 0 so successive laps tile
    seamlessly, with a single ``hold_s`` dwell at the end of the lap
    instead of one at every corner.

    Returns ``(t, eta, eta_dot, legs)``, ``legs[i] = (start_row, stop_row,
    from_index, to_index)`` — the same contract as before.
    """

    points = [np.asarray(p, dtype=float) for p in waypoints]
    if len(points) < 2:
        raise ValueError("need at least two waypoints")
    sequence = points + ([points[0]] if loop else [])
    xy, leg_of = _rounded_path(sequence, radius, ds)

    # Per-sample speed cap from the direction of travel.
    steps = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(steps, axis=1)
    keep = seg_len > 1e-12
    steps, seg_len = steps[keep], seg_len[keep]
    xy = np.vstack([xy[:-1][keep], xy[-1]])
    leg_of = np.append(leg_of[:-1][keep], leg_of[-1])
    caps = np.array([
        _speed_limit(step / length, cruise_u, cruise_v, heading_deg)
        for step, length in zip(steps, seg_len)])
    caps = np.append(caps, caps[-1])

    # Forward then backward acceleration pass: v^2 = v0^2 + 2*a*ds.
    speed = caps.copy()
    speed[0] = 0.0
    for i in range(len(seg_len)):
        speed[i + 1] = min(
            speed[i + 1], math.sqrt(speed[i] ** 2 + 2.0 * accel * seg_len[i]))
    speed[-1] = 0.0
    for i in range(len(seg_len) - 1, -1, -1):
        speed[i] = min(
            speed[i], math.sqrt(speed[i + 1] ** 2 + 2.0 * accel * seg_len[i]))

    # Arc length -> time, then resample onto the control grid.
    times = np.zeros(len(xy))
    for i in range(len(seg_len)):
        mean_speed = 0.5 * (speed[i] + speed[i + 1])
        times[i + 1] = times[i] + seg_len[i] / max(mean_speed, 1e-6)
    duration = float(times[-1])

    grid = np.arange(0.0, duration + hold_s + 0.5 * dt, dt)
    travel = np.minimum(grid, duration)
    eta = np.column_stack([np.interp(travel, times, xy[:, 0]),
                           np.interp(travel, times, xy[:, 1])])
    along = np.interp(travel, times, speed)
    heading_of = np.zeros((len(grid), 2))
    direction = np.vstack([steps / seg_len[:, None], steps[-1] / seg_len[-1]])
    heading_of[:, 0] = np.interp(travel, times, direction[:, 0])
    heading_of[:, 1] = np.interp(travel, times, direction[:, 1])
    norms = np.linalg.norm(heading_of, axis=1)
    norms[norms < 1e-9] = 1.0
    eta_dot = heading_of / norms[:, None] * along[:, None]
    eta_dot[grid > duration] = 0.0

    sample_leg = np.interp(travel, times, leg_of).round().astype(int)
    sample_leg[grid > duration] = int(leg_of[-1])
    legs = []
    for index in range(len(sequence) - 1):
        rows = np.where(sample_leg == index)[0]
        if rows.size == 0:
            continue
        to_index = (index + 1) % len(points)
        legs.append((int(rows[0]), int(rows[-1]) + 1, index, to_index))
    if duration > timeout_s * len(legs):
        print(f"[ghost] WARNING: lap takes {duration:.0f} s; raise the "
              "cruise speeds or shorten the pattern")

    return grid, eta, eta_dot, legs


def body_velocities(eta_dot, heading_deg):
    """World (vx, vy) -> body (u, v_starboard) at a fixed crab heading."""

    psi = math.radians(heading_deg)
    u = eta_dot[:, 0] * math.cos(psi) + eta_dot[:, 1] * math.sin(psi)
    v_starboard = (eta_dot[:, 0] * math.sin(psi)
                   - eta_dot[:, 1] * math.cos(psi))
    return u, v_starboard


def rect_extent(ref_world):
    xy = np.asarray(list(ref_world.values()), dtype=float)
    return float(xy[:, 0].max()), float(xy[:, 1].max())


def profile_report(ref_world, *, cruise_u, cruise_v):
    """Simulate the ghost and print the per-leg feasibility table.

    ``cruise_u``/``cruise_v`` come from the controller arm that will fly
    (``control_v2.waypoint_policy.WaypointTuning``); there is no default,
    so a preview can never quietly describe a speed no arm commands.

    Returns (t, eta, eta_dot, legs, warnings).
    """

    width, height = rect_extent(ref_world)
    pattern_xy = waypoints_xy()
    headings = waypoint_headings_deg()
    depths = waypoint_depths_m()
    t, eta, eta_dot, legs = simulate_ghost(
        pattern_xy, cruise_u=cruise_u, cruise_v=cruise_v)
    u, v_starboard = body_velocities(eta_dot, HEADING_DEG)
    clearance = np.minimum.reduce([
        eta[:, 0], width - eta[:, 0], eta[:, 1], height - eta[:, 1]])

    warnings = []
    if not np.isfinite(POOL_DEPTH_MAX_M) or POOL_DEPTH_MAX_M <= 0.0:
        warnings.append("POOL_DEPTH_MAX_M must be finite and positive")
    for x, y in pattern_xy:
        if not (0.0 <= x <= width and 0.0 <= y <= height):
            warnings.append(f"waypoint ({x:.2f},{y:.2f}) outside the "
                            f"{width:.2f} x {height:.2f} m reference rect")
    for index, depth in enumerate(depths):
        if not np.isfinite(depth) or depth < 0.0 \
                or depth > POOL_DEPTH_MAX_M:
            warnings.append(
                f"waypoint {index} depth {depth!r} outside the "
                f"0..{POOL_DEPTH_MAX_M:.2f} m positive-down envelope")

    print(f"[profile] {len(pattern_xy)} waypoints, loop={WAYPOINT_LOOP}, "
          f"heading {HEADING_DEG:.0f} deg, cruise "
          f"u={cruise_u:g} v={cruise_v:g} m/s, "
          f"cycle {t[-1]:.1f} s")
    print("[profile] surface-engage protocol: bring the vehicle to the "
          "WATER SURFACE before pressing m (engagement captures the "
          "surface reference); waypoint depths "
          + " ".join(f"{d:.2f}" for d in depths)
          + f" m below the surface (floor abort at {POOL_DEPTH_MAX_M:.2f} "
            "m; no shallow bound)")
    if headings is None:
        print(f"[profile] pressure depth TARGETS the active waypoint; "
              f"yaw HOLD {HEADING_DEG:.0f} deg (gyro_r_lpf feedback)")
    else:
        print(f"[profile] pressure depth TARGETS the active waypoint; "
              f"per-waypoint yaw setpoints "
              + " ".join(f"{h:g}" for h in headings) + " deg")
        print("[profile] NOTE: the body-velocity table below assumes the "
              f"{HEADING_DEG:.0f} deg base heading; while rotating, the "
              "runtime ellipse follows the measured heading instead, so "
              "per-leg u/v splits differ but stay under the same cruise "
              "caps")
    header = (f"{'leg':>7} {'len_m':>6} {'peak_spd':>8} {'peak_u':>7} "
              f"{'peak_v':>7} {'min_clear':>9}")
    print("[profile] " + header)
    for start, stop, source, target in legs:
        length = float(np.linalg.norm(
            np.asarray(pattern_xy[target], dtype=float)
            - np.asarray(pattern_xy[source], dtype=float)))
        seg = slice(start, stop)
        peak_speed = float(np.max(np.linalg.norm(eta_dot[seg], axis=1)))
        peak_u = float(np.max(np.abs(u[seg])))
        peak_v = float(np.max(np.abs(v_starboard[seg])))
        min_clear = float(np.min(clearance[seg]))
        flags = []
        if peak_u + CORRECTION_HEADROOM_M_S > U_MAX:
            flags.append(f"u+corr>{U_MAX}")
        if peak_v + CORRECTION_HEADROOM_M_S > V_MAX:
            flags.append(f"v+corr>{V_MAX}")
        if min_clear < WALL_MARGIN_M:
            flags.append(f"clear<{WALL_MARGIN_M}")
        line = (f"{source}->{target:<4} {length:6.2f} {peak_speed:8.3f} "
                f"{peak_u:7.3f} {peak_v:7.3f} {min_clear:9.2f}")
        print("[profile] " + line + ("   ! " + ", ".join(flags)
                                     if flags else ""))
        warnings.extend(
            f"leg {source}->{target}: {flag}" for flag in flags)

    rows, actuator_warnings = actuator_report(u, v_starboard, legs)
    if rows:
        print(f"[profile] actuator demand at the effective-rc stage "
              f"(PilotGain {PLANNING_PILOT_GAIN:g}):")
        print(f"[profile] {'leg':>7} {'req_u':>6} {'box_u':>6} "
              f"{'req_v':>6} {'box_v':>6} {'cmd_u':>6} {'cmd_v':>6} "
              f"{'rc_L1':>6}")
        for source, target, required, s_max, commanded, rc_l1, flags in rows:
            line = (f"{source}->{target:<4} {required[0]:6.3f} "
                    f"{s_max[0]:6.3f} {required[1]:6.3f} {s_max[1]:6.3f} "
                    f"{commanded[0]:6.3f} {commanded[1]:6.3f} {rc_l1:6.2f}")
            print("[profile] " + line
                  + ("   ! " + ", ".join(flags) if flags else ""))
        warnings.extend(actuator_warnings)
        if actuator_warnings:
            print(f"[profile] the box is bounded so its deadband-compensated "
                  f"command fits safety*PilotGain = "
                  f"{PLANNING_SAFETY * PLANNING_PILOT_GAIN:g}; to fit the "
                  f"pattern either lower GHOST_OMEGA, shorten the offending "
                  f"leg, or raise the vehicle PilotGain and pass a matching "
                  f"--mpc-planning-pilot-gain")

    for warning in warnings:
        print(f"[profile] WARNING: {warning}")
    if not warnings:
        print("[profile] all legs inside velocity caps (with "
              f"{CORRECTION_HEADROOM_M_S} m/s correction headroom), "
              "actuator limits, and wall margin")
    return t, eta, eta_dot, legs, warnings


def save_offline_png(ref_world, t, eta, eta_dot, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width, height = rect_extent(ref_world)
    u, v_starboard = body_velocities(eta_dot, HEADING_DEG)

    figure, (map_axis, vel_axis) = plt.subplots(
        2, 1, figsize=(10, 9),
        gridspec_kw={"height_ratios": [1.4, 1.0]})

    map_axis.add_patch(plt.Rectangle(
        (0, 0), width, height, fill=False, color="#c23b22", lw=1.5,
        label="reference rect"))
    map_axis.add_patch(plt.Rectangle(
        (WALL_MARGIN_M, WALL_MARGIN_M),
        width - 2 * WALL_MARGIN_M, height - 2 * WALL_MARGIN_M,
        fill=False, color="#c27b22", lw=1.0, ls="--",
        label=f"{WALL_MARGIN_M} m margin"))
    map_axis.plot(eta[:, 0], eta[:, 1], color="#1f6fb2", lw=1.4,
                  label="ghost path")
    pattern_xy = waypoints_xy()
    headings = waypoint_headings_deg()
    depths = waypoint_depths_m()
    for index, (x, y) in enumerate(pattern_xy):
        map_axis.plot(x, y, "o", color="#2a9d3a", ms=8)
        map_axis.annotate(f"{index}  d={depths[index]:.2f} m", (x, y),
                          textcoords="offset points", xytext=(6, 6))
        if headings is not None:
            angle = math.radians(headings[index])
            map_axis.annotate(
                "", xy=(x + 0.22 * math.cos(angle),
                        y + 0.22 * math.sin(angle)),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="#2a9d3a", lw=1.4))
    psi = math.radians(HEADING_DEG)
    center = np.mean(np.asarray(pattern_xy, dtype=float), axis=0)
    map_axis.annotate(
        "", xy=(center[0] + 0.3 * math.cos(psi),
                center[1] + 0.3 * math.sin(psi)),
        xytext=tuple(center),
        arrowprops=dict(arrowstyle="->", color="#5b4b8a", lw=2))
    map_axis.text(center[0], center[1] - 0.12, "bow", color="#5b4b8a",
                  ha="center")
    map_axis.set_aspect("equal")
    map_axis.set_xlabel("x [m]")
    map_axis.set_ylabel("y [m]")
    yaw_note = (f"yaw hold {HEADING_DEG:.0f} deg" if headings is None
                else "per-waypoint yaw setpoints (green arrows)")
    map_axis.set_title(
        "waypoint pattern in the overhead world frame — absolute pressure "
        f"depth targets below the engage-surface datum, {yaw_note}")
    map_axis.legend(fontsize=9, loc="upper right")

    vel_axis.plot(t, u, color="#1f6fb2", lw=1.2, label="u (surge)")
    vel_axis.plot(t, v_starboard, color="#2a9d3a", lw=1.2,
                  label="v (starboard)")
    for cap, label in ((U_MAX, "u cap"), (-U_MAX, None),
                       (V_MAX, None), (-V_MAX, None)):
        vel_axis.axhline(cap, color="#c23b22", lw=0.8, ls="--",
                         label=label)
    vel_axis.set_xlabel("time [s]")
    vel_axis.set_ylabel("body velocity [m/s]")
    vel_axis.set_title(
        f"ghost body-frame velocity references at heading "
        f"{HEADING_DEG:.0f} deg")
    vel_axis.legend(fontsize=9, loc="upper right")

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"[profile] preview written to {path}")


def world_to_pixels(H_mat, K, geometry_dist, dist, use_raw_geometry,
                    points_m):
    """World metres -> raw display pixels through the frozen map."""

    undistorted = topview.project(
        np.linalg.inv(H_mat), np.asarray(points_m, dtype=float))
    if use_raw_geometry:
        return undistorted
    return topview.distort_pixels(undistorted, K, dist)


def run_camera(args, ref_world, eta, warnings):
    import cv2

    ref_obj = topview.make_tag_object_map(
        ref_cam.resolve_ref_tag_sizes(ref_world))
    use_raw_geometry = (
        ref_cam.DISTORTION_MODE == "raw"
        or (ref_cam.DISTORTION_MODE == "auto" and len(ref_world) == 2))

    cap = topview.open_camera(args.device, args.width, args.height, args.fps)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.device!r} — check "
                         "ls /dev/v4l/by-id/ and that nothing else has it "
                         "open")
    grabber = topview.FrameGrabber(cap, name="waypoints-capture")
    grabber.start()
    if not grabber.wait_first():
        grabber.close()
        cap.release()
        raise SystemExit(f"No frames from {args.device!r} within 5 s"
                         + (f" ({grabber.error})" if grabber.error else ""))

    dictionary = cv2.aruco.getPredefinedDictionary(ref_cam.APRILTAG_DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    width, height = rect_extent(ref_world)
    rect_world = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    margin_world = [
        (WALL_MARGIN_M, WALL_MARGIN_M),
        (width - WALL_MARGIN_M, WALL_MARGIN_M),
        (width - WALL_MARGIN_M, height - WALL_MARGIN_M),
        (WALL_MARGIN_M, height - WALL_MARGIN_M),
    ]
    ghost_world = eta[::4]        # ~5 Hz polyline is plenty for display

    K = dist = geometry_dist = None
    H_mat = None
    H_samples = []
    drift_count = 0
    win = "waypoint preview (overhead camera)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    writer = None
    record_path = None
    if args.record is not None:
        # Auto-named clips land next to this script (usb_cam/), matching the
        # --calib and waypoint_preview.png defaults.  An explicit
        # --record PATH is honoured as given.
        record_path = args.record or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"usb_cam_waypoints_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    display_period = 1.0 / max(float(args.display_fps), 0.1)
    next_display = 0.0

    last_sequence = 0
    try:
        while True:
            frame, _host_ns, sequence = grabber.snapshot()
            if sequence == last_sequence:
                if grabber.stopped:
                    print("Frame grab failed, stopping."
                          + (f" ({grabber.error})" if grabber.error else ""))
                    break
                time.sleep(0.001)
                continue
            last_sequence = sequence
            if K is None:
                K, dist, is_calibrated = ref_cam.load_calibration(
                    args.calib, frame.shape, args.fov)
                geometry_dist = (np.zeros_like(dist) if use_raw_geometry
                                 else dist)
                if not is_calibrated:
                    print("[calib] WARNING: approximate FOV intrinsics — the "
                          "overlay is indicative only")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            ref_seen = topview.detect_references(
                corners, ids, ref_world, ref_obj, K, geometry_dist)
            pair_rows = ref_cam.pnp_pair_errors(ref_seen, ref_world)
            max_pair_error = (
                max(abs(measured - expected)
                    for _, _, measured, expected in pair_rows)
                if pair_rows else np.nan)
            candidate_H, candidate_quality = (
                topview.estimate_reference_homography(
                    ref_seen, ref_world, K, geometry_dist, ref_obj))
            reject = ref_cam.gate_candidate(
                candidate_H, candidate_quality, max_pair_error, ref_world)
            if not reject and H_mat is None:
                H_samples.append(candidate_H)
                if len(H_samples) >= ref_cam.REF_INIT_FRAMES:
                    H_mat = topview.average_homographies(H_samples)
                    ref_cam.print_freeze_report(
                        H_mat, ref_seen, ref_world, candidate_quality,
                        pair_rows, K, geometry_dist, len(H_samples))

            # Frozen-map drift monitor, as in usb_cam_apriltag.py.
            if H_mat is not None and len(ref_seen) >= 2:
                seen_ids = list(ref_seen)
                seen_world = topview.project(H_mat, topview.undistort_pts(
                    [ref_seen[tag]["center_px"] for tag in seen_ids],
                    K, geometry_dist))
                expected_world = np.asarray(
                    [ref_world[tag] for tag in seen_ids], dtype=float)
                world_rms = float(np.sqrt(np.mean(np.sum(
                    (seen_world - expected_world) ** 2, axis=1))))
                drift_count = (drift_count + 1
                               if world_rms > ref_cam.REF_DRIFT_M else 0)
                if drift_count >= ref_cam.REF_DRIFT_FRAMES:
                    print(f"[calib] reference drift {world_rms:.3f} m; "
                          "clearing map and reacquiring")
                    H_mat = None
                    H_samples.clear()
                    drift_count = 0

            # --- waypoint overlay -------------------------------------------
            if H_mat is not None:
                def to_px(points_m):
                    return np.rint(world_to_pixels(
                        H_mat, K, geometry_dist, dist, use_raw_geometry,
                        points_m)).astype(np.int32)

                cv2.polylines(frame, [to_px(rect_world).reshape(-1, 1, 2)],
                              True, (34, 59, 194), 2, cv2.LINE_AA)
                cv2.polylines(frame, [to_px(margin_world).reshape(-1, 1, 2)],
                              True, (34, 123, 194), 1, cv2.LINE_AA)
                cv2.polylines(frame, [to_px(ghost_world).reshape(-1, 1, 2)],
                              bool(WAYPOINT_LOOP), (178, 111, 31), 2,
                              cv2.LINE_AA)
                pattern_xy = waypoints_xy()
                headings = waypoint_headings_deg()
                depths = waypoint_depths_m()
                for index, waypoint in enumerate(pattern_xy):
                    pixel = to_px([waypoint])[0]
                    cv2.circle(frame, tuple(pixel), 9, (58, 157, 42), 2,
                               cv2.LINE_AA)
                    cv2.putText(frame,
                                f"{index} d={depths[index]:.2f}m",
                                (pixel[0] + 10, pixel[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (58, 157, 42), 2, cv2.LINE_AA)
                    if headings is not None:
                        angle = math.radians(headings[index])
                        tip_wp = (waypoint[0] + 0.25 * math.cos(angle),
                                  waypoint[1] + 0.25 * math.sin(angle))
                        base_wp_px, tip_wp_px = to_px([waypoint, tip_wp])
                        cv2.arrowedLine(
                            frame, tuple(base_wp_px), tuple(tip_wp_px),
                            (58, 157, 42), 2, cv2.LINE_AA, tipLength=0.3)

                # Bow direction at the pattern centre (held heading).
                psi = math.radians(HEADING_DEG)
                center = np.mean(np.asarray(pattern_xy, dtype=float), axis=0)
                tip = center + 0.35 * np.array([math.cos(psi), math.sin(psi)])
                base_px, tip_px = to_px([center, tip])
                cv2.arrowedLine(frame, tuple(base_px), tuple(tip_px),
                                (138, 75, 91), 3, cv2.LINE_AA, tipLength=0.3)
                cv2.putText(frame,
                            f"bow {HEADING_DEG:.0f} deg | engage at "
                            "surface (depth datum 0 m)",
                            (tip_px[0] + 8, tip_px[1] + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (138, 75, 91), 2, cv2.LINE_AA)

            stage = ("CALIBRATED" if H_mat is not None
                     else f"CALIBRATING {len(H_samples)}"
                          f"/{ref_cam.REF_INIT_FRAMES}")
            verdict = ("profile OK" if not warnings
                       else f"{len(warnings)} profile warning(s) — see "
                            "terminal")
            cv2.putText(frame, f"{stage} | {verdict}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if H_mat is not None and not warnings
                        else (0, 165, 255), 2)
            if reject and H_mat is None:
                cv2.putText(frame, "reject: " + ";".join(reject), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.64,
                            (0, 165, 255), 2, cv2.LINE_AA)

            # Records the annotated frame -- every overlay above is drawn onto
            # `frame` in place, so the MP4 matches the preview exactly.
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
                if key == ord("r"):
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
                  f"{record_path}; encoder dropped {writer.dropped_seconds:.3f} s")
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=ref_cam.DEFAULT_DEVICE)
    # Match the calibration's native 1920x1200; any other size makes K be
    # rescaled, which a cropped sensor mode makes wrong.
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--calib",
                    default=os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "usb_cam_calibration.npz"))
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=90.0,
                    help="requested camera capture rate (default 90)")
    ap.add_argument("--record", nargs="?", const="", default=None,
                    metavar="PATH",
                    help="record the annotated preview (overlays included) to "
                         "an MP4. Bare --record writes "
                         "usb_cam_waypoints_<timestamp>.mp4 into the usb_cam/ "
                         "folder next to this script")
    ap.add_argument("--record-fps", type=float, default=30.0,
                    help="MP4 cadence, independent of --fps (default 30)")
    ap.add_argument("--display-fps", type=float, default=15.0,
                    help="max preview redraws per second (default 15)")
    ap.add_argument("--offline", action="store_true",
                    help="no camera: print the report and write "
                         "waypoint_preview.png next to this script")
    ap.add_argument("--arm", default=DEFAULT_PREVIEW_ARM,
                    help="controller arm whose cruise speeds the preview "
                         "and feasibility report describe, e.g. "
                         "vel_mpc_mz_gle_waypoint. The speeds live with the "
                         f"arm, not here (default: {DEFAULT_PREVIEW_ARM})")
    args = ap.parse_args()

    cruise_u, cruise_v = arm_cruise(args.arm)
    print(f"[profile] cruise from {args.arm}: "
          f"u={cruise_u:g} v={cruise_v:g} m/s")

    ref_world = ref_cam.build_ref_world()
    if not ref_world:
        raise SystemExit("REF_RECT/REF_LAYOUT in usb_cam_apriltag.py must "
                         "be set: the waypoint overlay needs the frozen "
                         "reference map")
    ref_cam.check_layout_geometry(ref_world)
    ref_cam.describe_layout(ref_world, ref_cam.resolve_ref_tag_sizes(
        ref_world))

    t, eta, eta_dot, legs, warnings = profile_report(
        ref_world, cruise_u=cruise_u, cruise_v=cruise_v)

    if args.offline:
        save_offline_png(
            ref_world, t, eta, eta_dot,
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "waypoint_preview.png"))
        return

    run_camera(args, ref_world, eta, warnings)


if __name__ == "__main__":
    main()
