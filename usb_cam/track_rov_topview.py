#!/usr/bin/env python3
"""
Live overhead tracking of the ROV's red+right tags (right = green or white),
metrically referenced to two or more AprilTags (tag36h11) mounted above the
water at known positions.

Pipeline per frame:
  1. detect the reference tags -> homography (selected pixel space -> metres
     on the reference plane); by default detections are accumulated and a
     robust transform is frozen once the camera/static-tag geometry is well
     observed, then LATCHED for the rest of the run (since 2026-08-05):
     losing reference tags or sustained drift past --ref-drift-m is
     reported on the console, the overlay status, and the per-row
     ref_world_rms_m column, but no longer clears the map mid-run —
     --ref-drift-reacquire restores the old clear-and-remap behavior;
     show each tag's live camera xyz, pool xy, baseline, and
     calibration status
  2. detect the red/selected-right tags by HSV segmentation and show live
     boxes, midpoint, red->right arrow, heading, and trail, plus a white
     0-degree reference ray; metric mapping waits for reference calibration
  3. midpoint -> position, red->right axis -> heading, short-window regression
     plus a time-aware low-pass -> world velocity (vx, vy), yaw rate r, and
     body-frame surge/sway (u, v). Heading is unwrapped only inside that
     derivative window and is published/logged canonically in [0, 360 deg),
     so e.g. 362 deg and 2 deg cannot become different waypoint states.
     Rigid marker-spacing and last-good
     position/heading gates run before the regression; rejected candidates
     remain in the CSV with invalid velocity and an explicit reason.

Default reference frame -- the four-tag pool rectangle defined in
usb_cam/pool_layout_4tag.py, which combined_view and usb_cam_apriltag.py read
from the same place:

    ID 501 (0.024, 2.038) --------- ID 500 (4.423, 2.117)   far edge, 0.40 m
         |                               |
         | +y                            |
         |                               |
    ID 100 (0.000, 0.000) --- +x --- ID 102 (4.427, 0.000)  near edge, 0.20 m
    ORIGIN                                                     camera side

Those centres were measured FROM THE TAGS on 2026-08-04, not from a tape,
and they are deliberately not a perfect rectangle -- see pool_layout_4tag.py.

The origin sits at the near-left tag 100, +x runs along the camera-side edge
toward ID 102 and +y points from that edge into the pool.  The two edge
lengths and the mixed 0.20/0.40 m tag sizes are configured in
pool_layout_4tag.py; ``--ref-layout`` and ``--ref-tag-size`` override them,
and ``--ref-tag-size`` accepts either one number for all references or a
per-tag ``100:0.20;500:0.40`` map.  A tag solved at the wrong edge length
lands at the wrong range and corrupts every distance gate, so mixed-size
layouts must use the per-tag form.

Three or more tag centres are mapped by a direct homography (four or more) or
calibrated SQPnP with range-based branch selection (exactly three).  A two-tag
layout may be either an edge pair (equal y; the pool then extends +y by
--pool-width) or a diagonal pair, which spans the tracking rectangle by
itself.  The two-tag solution uses all four corners of each square, aligns
their baseline to its measured length, robustly accumulates the resulting
transforms, and then freezes the map. ``--baseline-scale-source configured``
uses the layout's configured baseline for metric scale; ``camera`` instead
uses the valid PnP baseline recovered from the two tags; two-tag calibration
already requires both PnP poses to be valid. In the default ``auto``
distortion mode a TWO-tag layout uses raw pixels: the present calibration
views do not cover the extreme image edges occupied by the reference tags,
and applying those extrapolated coefficients there produces invalid PnP
poses.  Three or more tags use the calibrated coefficients, so watch the
corner tags' reprojection error on the four-tag layout and recalibrate with
edge-reaching views if it is poor; ``--distortion-mode raw`` forces the
bypass.  --pool-width sets the positive-y tracking bound for edge-pair
layouts only.
Live ``cam xyz`` labels use OpenCV camera coordinates: +x image-right, +y
image-down, and +z forward from the camera. In raw mode these PnP values are
approximate. The configured reference baseline sets the standalone map scale;
combined view explicitly selects camera-PnP scale. ``pool xy`` uses the frame
above.

Marker heading is the red->right axis, CCW from +x.  The right marker is
bright lime under ``--marker-pair red-lime`` (the default since 2026-08-17,
when the tape was swapped), green under ``--marker-pair red-green``, or white
under ``--marker-pair red-white``.
Lime is a separate profile rather than a widened green band because the two
tapes segment on opposite cues: green stands out from the water by
saturation, lime by brightness.  Crossing them fails loudly: the lime tape
read under red-green pairs on ~13% of frames.  Runs recorded before the swap
must be replayed with ``--marker-pair red-green``; their CSVs record the pair
they ran with.  The current ROV's marker bar is transverse to
its bow, so ``--heading-offset`` defaults to +90 degrees to obtain vehicle
heading.  Pass ``--heading-offset 0`` when red->right itself points forward.
Body frame: u = forward along vehicle heading, v = toward port (left); negate
v downstream if you use starboard-positive.

Refraction: coordinates are metres ON THE TAG PLANE. The submerged ROV's
true displacement is larger by s = (H + d/1.33) / (H - h_tag), with H the
camera height above water, h_tag the tag-plane height above water, d the ROV
depth. Pass --cam-height/--tag-height/--rov-depth to apply a constant s
live; with varying depth, post-multiply using logged depth telemetry
instead (the CSV keeps uncorrected plane coordinates; the 'scale' column
records the factor already applied, 1.0 = none).

Pool span: near edge (100 -> 102) = 442.7cm, left edge (100 -> 501) =
203.8cm, near-pair tag edge 20cm, far-pair tag edge 40cm.  --pool-width
applies only to a two-tag edge-pair layout and is unused by the four-tag
frame.  Re-measure with usb_cam/pool_frame_4tag.py, which prints configured
against PnP distance for all six tag pairs, and update pool_layout_4tag.py.

Usage:
    python3 usb_cam/track_rov_topview.py
    python3 usb_cam/track_rov_topview.py --marker-pair red-white
    python3 usb_cam/track_rov_topview.py --marker-pair red-lime
    python3 usb_cam/track_rov_topview.py --video overhead.mp4
    # add --show-mask to tune HSV thresholds; Esc quits

Writes usb_cam/rov_track_<timestamp>.csv (override with --out).  The CSV
includes every candidate gate decision, raw reference corners, camera-frame
tag poses, measured/expected/used map baselines, scale source, and
candidate/active homographies so a failed calibration can be reproduced
without relying on the live overlay.
Live annotated MP4 output is paced by the wall clock at the camera's nominal
frame rate: when tracking is slower, the newest fully annotated frame is
repeated so playback duration remains equal to capture duration.  Offline
``--video`` replay instead preserves one output frame per source frame at the
source frame rate, independent of how long analysis takes.
"""

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from usb_cam import pool_layout_4tag as pool_layout
except ImportError:  # run as a plain script: usb_cam/ itself is on sys.path
    import pool_layout_4tag as pool_layout

DEFAULT_DEVICE = ("/dev/v4l/by-id/usb-Global_Shutter_Camera_"
                  "Global_Shutter_Camera_01.00.00-video-index0")
DEFAULT_CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "usb_cam_calibration.npz")
APRILTAG_DICT = cv2.aruco.DICT_APRILTAG_36h11

# August 2026 four-tag layout, metres, on the common tag plane: the origin is
# the near-left tag (100), +x along the camera-side edge to 102, +y into the
# pool.  Both the centres and the mixed 0.20/0.40 m tag sizes come from
# pool_layout_4tag.py, which every other consumer reads as well -- re-measure
# there, not here.  It replaces the July two-tag 500/501 diagonal.
DEFAULT_REF_LAYOUT = pool_layout.layout_spec()
DEFAULT_REF_TAG_SIZE_M = pool_layout.tag_size_spec()
DEFAULT_POOL_WIDTH_M = 1.975

# HSV thresholds (OpenCV H in [0,180]) — green/red were tuned on the overhead
# MP4. White assumes a matte bright marker: low saturation and high value.
# Lighting over the tank WILL differ. Verify with --show-mask before control.
#
# Green retightened 2026-08-14, measured on two raw overhead frames of the
# vehicle in the pool (NOT on the annotated MP4, whose own overlay is drawn
# in pure green at H=60 and would bias any fit):
#
#     marker   H 67-68 (median), S 185, V 233
#     water    H 98    (median), S 153, V 118
#
# The old H<=95 ceiling reached into the water and selected 286k px of pool
# surface as one blob — the pool, not the marker, was the largest green
# object in frame. Hue is what separates them: saturation cannot, because
# the water's median S of 153 is higher than plenty of marker edge pixels.
# The ceiling at 82 sits 14 below the water median and 14 above the marker
# median. Measured over both frames this keeps the marker as the largest
# blob by ~100x (842 and 1278 px) while no spurious blob exceeds 9 px.
#
# The floor at 63 costs nothing (the marker's darkest pixel is H=62 and the
# 5th percentile is 67 — moving the floor from 55 to 63 changed neither
# frame's blobs at all) and keeps the band clear of H=60, the pure green
# this tracker draws its own overlays in. That way re-running the detector
# over a recorded, annotated MP4 cannot lock onto our own annotations.
GREEN_LO = np.array([63, 140, 80])
GREEN_HI = np.array([82, 255, 255])
RED_LO = np.array([160, 90, 50])
RED_HI = np.array([180, 255, 255])
WHITE_LO = np.array([0, 0, 170])
WHITE_HI = np.array([180, 70, 255])

# Bright lime marker fitted 2026-08-17 on the 0817 overhead run. This tape is
# NOT the 'green' above and must not reuse its band: run under red-green it
# held tracking on 112/953 frames (11.8%), every reject 'no_color_pair'.
#
# Sampled over 296 frames of that run, isolating the tag by green-minus-blue
# dominance (G-B > 25 near the red marker) so the numbers below are not
# censored by the band they justify:
#
#     tag    H 35/59/79 (p1/med/p99)  S 58/91/160  V 133/250/255
#     water  H 93-96                  S 123-162    V 161-202
#
# Two things differ from the green tag. Saturation can no longer reject the
# water: the tag's median S of 91 sits BELOW the water's 123-162, the reverse
# of the green case, so an S floor near 140 removes the marker and keeps the
# pool. And the hue wanders far wider (p1 35, glinting yellow in sun) than
# green's 63-75, so the floor has to drop to 35.
#
# That leaves hue-ceiling plus value as the separator: 85 clears the tag's
# p99 of 79 while holding 8-11 counts below the water hue.
#
# Floors retuned 2026-08-17 on run 20260817_141522, flown ~1.5 h later and
# in visibly flatter light. The tape held 96.6% in the water and the band
# missed it on 0 of 1495 frames, so this is a MARGIN change, not a fix:
#
#   - V 170 -> 150. The dimmer run's per-frame median V fell to 159, under
#     the old floor, and the worst frame's blob shrank to 343 px against the
#     300 px MIN_AREA -- 14% headroom. At 150 that worst frame is 438 px
#     (46%). Do not chase this lower: the water's median V is 148, so a
#     floor much under 150 hands the pool the one cue that separates it.
#   - H 35 -> 45. Below 45 the tape contributes only glint fringe, not core:
#     its per-frame median hue never dropped below 56 across either run.
#     Dropping that fringe halves the spurious blobs (0.61 -> 0.29 per
#     frame) and costs nothing -- the worst-frame blob is 438 px either way.
#     The saved false-positive budget is what pays for the lower V floor.
#
# S floor 60 -> 20 on 2026-08-17 afternoon, when the sun came around onto
# the pool and tracking fell run over run: 89% (1527) -> 67% (1540) ->
# 52% (1545) -> 41% (1554), every reject 'no_color_pair'.
#
# Under direct sun the tape does not shift hue -- it BLOWS OUT. Measured
# over the 1554 failure window against a hue-free probe (bright, unsaturated,
# non-blue, near the red marker), the tape reads S 7-46 per-frame median
# (27 overall) at V 217-253, against S 83-194 in flat light. It has
# effectively become a white marker. The old S >= 60 floor passed 0.1% of
# its pixels, leaving no blob at all: the band found the TRUE tag on 0.0%
# of sunlit frames. At S >= 20 it finds it on 96.9%, and flat-light runs
# are unchanged at 100%.
#
# The cliff is sharp because the tape's sunlit median saturation is 27:
# S >= 30 already drops to 18.5%. Do not raise this floor back toward the
# old value without checking a sunlit run. Going lower buys nothing --
# S >= 10 also scores 96.9%, and 20 keeps more of the false-positive
# margin, which now matters because the hue ceiling is doing nearly all of
# the work of rejecting the water and the vehicle's own sun-blown hull.
#
# Honest caveat: this is a net win but not a uniform one. Pixel-pair rates
# over the annotated MP4s went 45.7 -> 92.4% (1554), 62.1 -> 78.3% (1545),
# 31.5 -> 36.1% (1527) and 66.4 -> 66.4% (1415), but 1540 REGRESSED,
# 48.4 -> 38.6%. No variant tried (S 25, hue ceiling 80, V 170) recovered
# 1540 without giving back most of the sunlit gain. That proxy lacks the
# metric spacing gate the live tracker applies, so it likely overstates the
# damage, but treat 1540-like light as unverified.
#
# None of this makes the tape a good sunlit marker. A matte, deeply
# saturated tape would not blow out in the first place; thresholds cannot
# recover colour the sensor has already clipped.
LIME_LO = np.array([45, 20, 150])
LIME_HI = np.array([85, 255, 255])

# red-lime since 2026-08-17: the lime tape is what is physically on the
# vehicle. red-green remains selectable for replaying runs recorded before
# the swap — their CSVs name their own pair, so nothing older is reinterpreted.
DEFAULT_MARKER_PAIR = "red-lime"


@dataclass(frozen=True)
class MarkerPairConfig:
    name: str
    right_name: str
    right_lo: np.ndarray
    right_hi: np.ndarray
    right_overlay_bgr: tuple[int, int, int]


MARKER_PAIR_CONFIGS = {
    "red-green": MarkerPairConfig(
        "red-green", "green", GREEN_LO, GREEN_HI, (0, 255, 0)),
    "red-white": MarkerPairConfig(
        "red-white", "white", WHITE_LO, WHITE_HI, (255, 255, 255)),
    # Overlay is violet, not the marker's own colour. The lime band spans
    # H 35-85 and the tape's median hue is 59, so the band cannot be shaped
    # around the pure green (H=60) this tracker draws with, the way
    # GREEN_LO/HI is. Violet lands at H=135, clear of the lime ceiling by 50
    # and of the red floor of 160 by 25, so re-running the detector over an
    # annotated MP4 cannot lock onto either overlay. Not magenta: that is
    # the withdrawn tag-baseline colour and stays withdrawn.
    "red-lime": MarkerPairConfig(
        "red-lime", "lime", LIME_LO, LIME_HI, (255, 0, 128)),
}


def marker_pair_config(name):
    """Return the validated red/right marker segmentation contract."""

    try:
        return MARKER_PAIR_CONFIGS[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"marker pair must be one of {sorted(MARKER_PAIR_CONFIGS)}"
        ) from exc

MIN_AREA = 300        # px^2, color blob accept range
MAX_AREA = 40000
MAX_PAIR_DIST_PX = 350
MAX_JUMP_PX = 250
ZERO_REFERENCE_LENGTH_PX = 150
# Same length as the datum on purpose: two segments of equal length from a
# common origin make the angle between them readable without a protractor.
HEADING_ARROW_LENGTH_PX = 150
MAX_REF_POSE_REPROJ_PX = 5.0

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Midpoint-trail history drawn on the overlay, in accepted samples.
TRAIL_POINTS = 400

N_REF_FRAMES = 30

# The physical red/right bar is a compact, rigid object. Candidate selection
# deliberately retains the older loose 0.80 m search radius so rejected
# geometry can still be inspected in the raw CSV.  These stricter defaults are
# applied to the selected candidate before it can enter MotionEstimator.
QUALITY_MIN_COLOR_SPACING_M = 0.10
QUALITY_MAX_COLOR_SPACING_M = 0.35
QUALITY_MAX_SPACING_ERROR_M = 0.08
QUALITY_MAX_HEADING_STEP_DEG = 45.0
QUALITY_MAX_SPEED_M_S = 1.0
QUALITY_JUMP_SLACK_M = 0.05
QUALITY_IDENTITY_MEMORY_S = 1.0


def capture_backend():
    """The cv2 capture backend for local cameras on this platform.

    V4L2 is Linux-only and does not exist on macOS, where OBS's virtual
    camera and every UVC device arrive through AVFoundation instead.  Both
    preflights and every camera open in this repository route through here
    so a mac never asks OpenCV for a backend it was not built with.
    """
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


# macOS camera identity.  An AVFoundation index is not a name for a camera:
# the device array reorders on nearly every open as Continuity cameras
# attach, sleep and wake, so an index read even a moment earlier can point
# somewhere else by the time OpenCV uses it.  Devices are therefore named by
# ``avf:<uniqueID>`` end to end and opened through usb_cam/avf_capture.py,
# which asks AVFoundation for that identity directly.
AVF_DEVICE_PREFIX = "avf:"
_AVF_ORDER_WARNED = False


def _macos_devices_via_pyobjc():
    import objc

    namespace = {}
    objc.loadBundle(
        "AVFoundation", namespace,
        bundle_path="/System/Library/Frameworks/AVFoundation.framework")
    return [{"name": str(d.localizedName()),
             "model_id": str(d.modelID()),
             "unique_id": str(d.uniqueID())}
            for d in namespace["AVCaptureDevice"].devicesWithMediaType_("vide")]


def _macos_devices_via_system_profiler():
    out = subprocess.run(
        ["system_profiler", "-json", "SPCameraDataType"],
        capture_output=True, text=True, timeout=20, check=True).stdout
    return [{"name": entry.get("_name", ""),
             "model_id": entry.get("spcamera_model-id", ""),
             "unique_id": entry.get("spcamera_unique-id", "")}
            for entry in json.loads(out).get("SPCameraDataType", [])]


def macos_video_devices():
    """Video capture devices in OpenCV's AVFoundation index order.

    OpenCV's AVFoundation backend opens a camera by *index* into
    ``[AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo]``, which is
    exactly the array pyobjc reads here, so an entry's position is the index
    to pass.  ``system_profiler`` is the fallback when pyobjc is missing: it
    reports the same devices but is a separate API whose order has been seen
    to differ, so that path warns once -- a wrong order there means opening
    the wrong camera, not an error.

    Each entry is ``{"name", "model_id", "unique_id"}``.  Returns ``[]`` off
    macOS, or when neither source answers.
    """
    global _AVF_ORDER_WARNED

    if sys.platform != "darwin":
        return []
    try:
        return _macos_devices_via_pyobjc()
    except Exception:
        pass
    try:
        devices = _macos_devices_via_system_profiler()
    except Exception:
        return []
    if devices and not _AVF_ORDER_WARNED:
        _AVF_ORDER_WARNED = True
        print("[camera] pyobjc is unavailable, so camera indices come from "
              "system_profiler order. Confirm the view is the pool before "
              "trusting a run, or install pyobjc-core.")
    return devices


def is_avf_device(device):
    """True for a macOS ``avf:<uniqueID>`` camera identity."""

    return str(device).startswith(AVF_DEVICE_PREFIX)


def list_local_cameras():
    """``(device spec, description)`` for every locally attached camera.

    The spec is what ``--device`` takes: a ``/dev/videoN`` path on Linux, an
    ``avf:<uniqueID>`` identity on macOS.  Lets the calibration and tracking
    tools show what is attached instead of leaving an operator to guess an
    index on a platform where the names are not in the filesystem.
    """
    if sys.platform == "darwin":
        return [(AVF_DEVICE_PREFIX + device["unique_id"],
                 f"{device['name']} [{device['model_id']}]")
                for device in macos_video_devices()]
    cameras = []
    for name_file in sorted(glob.glob("/sys/class/video4linux/video*/name")):
        try:
            with open(name_file) as handle:
                name = handle.read().strip()
        except OSError:
            continue
        cameras.append(("/dev/" + name_file.split("/")[-2], name))
    return cameras


def resolve_capture_device(device):
    """Turn a device spec into the argument ``cv2.VideoCapture`` expects.

    Paths, URLs and plain indices only.  ``avf:<uniqueID>`` deliberately does
    NOT resolve to an index here: macOS reorders its camera array on nearly
    every open, so an index handed to OpenCV can already name a different
    camera.  Those devices go to :mod:`usb_cam.avf_capture` instead, which
    opens by identity -- see :func:`open_capture`.
    """
    text = str(device)
    if text.startswith(AVF_DEVICE_PREFIX):
        raise RuntimeError(
            f"{text} names a macOS camera by identity; open it with "
            "open_capture(), not by index")
    return int(text) if text.isdigit() else device


def open_capture(device, width=None, height=None, fps=None):
    """Open any camera spec: the one place a capture object is created.

    ``avf:<uniqueID>`` opens through AVFoundation by identity; everything
    else is a cv2.VideoCapture on the platform's backend.  Both return the
    same read()/isOpened()/release() surface, so callers do not branch.
    """
    if is_avf_device(device):
        try:
            from usb_cam.avf_capture import open_avf_camera
        except ImportError:  # plain script: usb_cam/ itself is on sys.path
            from avf_capture import open_avf_camera

        return open_avf_camera(device, width=width, height=height, fps=fps)

    text = str(device)
    if text.startswith(("http://", "rtsp://")):
        # Network stream (e.g. DroidCam's MJPEG server on the phone): the
        # sender fixes format/size/rate, none of the local knobs apply.
        return cv2.VideoCapture(text)
    cap = cv2.VideoCapture(resolve_capture_device(device), capture_backend())
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap


def open_camera(device, width, height, fps=None, buffersize=4):
    """Open a UVC camera for MJPG capture at the requested size and rate.

    Three settings matter for throughput on the AR0234 bench camera:

    * ``FOURCC=MJPG`` -- the driver caps YUYV at 5 fps for 1920x1080 and
      above, versus 90 fps for MJPG.
    * ``CAP_PROP_FPS`` -- without it the driver stays at its 30 fps default,
      which measured 15 fps in practice.
    * ``CAP_PROP_BUFFERSIZE`` -- a single buffer forces the camera to idle
      while userspace copies the current frame, roughly halving the rate
      (41 fps versus 87 fps measured at 1920x1200).  Pair a deeper queue with
      :class:`FrameGrabber`, which drains it continuously so the extra
      buffers add headroom rather than latency.
    """
    cap = open_capture(device, width=width, height=height, fps=fps)
    if is_avf_device(device) or str(device).startswith(("http://", "rtsp://")):
        # AVFoundation and network senders choose the format themselves;
        # the V4L2 knobs below do not exist for either.
        return cap
    if sys.platform != "darwin":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if buffersize:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffersize))
    return cap


class FrameGrabber(threading.Thread):
    """Pull frames continuously so the camera never waits on processing.

    Detection, overlay drawing, and display together cost more than one frame
    period, so a loop that calls ``cap.read()`` inline leaves the sensor idle
    for most of every cycle.  This thread keeps a single latest-frame slot:
    when processing falls behind, intermediate frames are dropped rather than
    queued, and the growing sequence number reveals the skips.

    ``cv2.VideoCapture.read()`` allocates a fresh array per call, so each
    published frame is a buffer nothing else will touch.  That matters because
    :class:`WallClockVideoWriter` holds a submitted frame by reference and
    encodes it asynchronously -- recycling capture buffers here would let the
    encoder write a frame while the tracking loop is still drawing on it.
    """

    def __init__(self, cap, name="topview-capture"):
        super().__init__(name=name, daemon=True)
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._host_ns = 0
        self._sequence = 0
        self._stop_event = threading.Event()
        self.error = None
        self.frames_read = 0

    def run(self):
        try:
            while not self._stop_event.is_set():
                ok, frame = self._cap.read()
                if not ok:
                    break
                host_ns = time.monotonic_ns()
                with self._lock:
                    self._frame = frame
                    self._host_ns = host_ns
                    self._sequence += 1
                    self.frames_read += 1
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._stop_event.set()

    def snapshot(self):
        """Latest ``(frame, host_monotonic_ns, sequence)``; frame may be None."""

        with self._lock:
            return self._frame, self._host_ns, self._sequence

    @property
    def stopped(self):
        return self._stop_event.is_set()

    def wait_first(self, timeout=5.0):
        """Block until the first frame arrives.  True if one did."""

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.snapshot()[0] is not None:
                return True
            if self.stopped:
                return False
            time.sleep(0.002)
        return False

    def close(self):
        self._stop_event.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=2.0)


class WallClockVideoWriter(threading.Thread):
    """Write the newest frame at a fixed wall-clock cadence.

    The tracking loop may produce annotated frames much more slowly than the
    camera's nominal frame rate.  Repeating the latest complete frame keeps
    the MP4 timeline in real time without blocking tracking or control.
    """

    def __init__(self, path, fps, size):
        super().__init__(name="topview-video-writer", daemon=True)
        fps = float(fps)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("video output FPS must be positive and finite")
        self._writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open MP4 writer: {path}")
        self._period = 1.0 / fps
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame = None
        self._closed = False
        self.error = None
        self.frames_written = 0
        self.dropped_seconds = 0.0

    def submit(self, frame):
        """Replace the frame that will be emitted at the next cadence."""

        with self._lock:
            self._frame = frame

    def _write_latest(self):
        """Write one copy of the latest frame, if one has been submitted."""

        with self._lock:
            frame = self._frame
        if frame is None:
            return False
        self._writer.write(frame)
        self.frames_written += 1
        return True

    def run(self):
        deadline = time.monotonic()
        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                wait = deadline - now
                if wait > 0.0:
                    self._stop_event.wait(min(wait, 0.02))
                    continue
                self._write_latest()
                deadline += self._period
                if now - deadline > 1.0:
                    self.dropped_seconds += now - deadline
                    deadline = now
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def close(self):
        """Stop, join, and finalize the MP4 exactly once."""

        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=10.0)
        if self.is_alive():
            raise RuntimeError(
                "top-view MP4 encoder did not stop within 10 seconds")
        self._writer.release()
        if self.error is not None:
            raise RuntimeError(f"top-view MP4 encoder failed: {self.error}")


def find_blobs(hsv, lo, hi):
    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    blobs = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if MIN_AREA <= area <= MAX_AREA:
            x, y, w, h = stats[i, :4]
            if max(w, h) <= 3.0 * min(w, h):
                blobs.append((centroids[i][0], centroids[i][1], area, (x, y, w, h)))
    return blobs, mask


def pick_pair_pixel(right_blobs, reds, previous_mid):
    """Pixel-space preview pairing for the selected red/right marker pair."""

    best, best_cost = None, None
    for right in right_blobs:
        for red in reds:
            distance = math.hypot(right[0] - red[0], right[1] - red[1])
            if distance > MAX_PAIR_DIST_PX:
                continue
            midpoint = ((right[0] + red[0]) / 2.0,
                        (right[1] + red[1]) / 2.0)
            cost = distance
            if previous_mid is not None:
                jump = math.hypot(
                    midpoint[0] - previous_mid[0],
                    midpoint[1] - previous_mid[1])
                if jump > MAX_JUMP_PX:
                    continue
                cost += jump
            if best_cost is None or cost < best_cost:
                best, best_cost = (right, red), cost
    return best


def parse_ref_layout(spec):
    """Parse ``id:x,y;...`` into a tag-centre world-coordinate dictionary."""

    layout = {}
    try:
        for item in spec.split(";"):
            tag, xy = item.strip().split(":", 1)
            x, y = (float(value) for value in xy.split(","))
            layout[int(tag)] = (x, y)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "reference layout must look like "
            "'100:0,0;102:4.427,0'"
        ) from exc
    if len(layout) < 2:
        raise ValueError("reference layout needs at least two tag IDs")
    xy = np.asarray(list(layout.values()), dtype=float)
    if len(layout) == 2 and np.linalg.norm(xy[1] - xy[0]) <= 1e-9:
        raise ValueError("two reference tag centres must be distinct")
    if len(layout) >= 3 and np.linalg.matrix_rank(xy[1:] - xy[0]) < 2:
        raise ValueError("reference tag centres must not be collinear")
    return layout


def make_tag_object_points(tag_size_m):
    """AprilTag black-square corners in OpenCV ArUco corner order."""

    half = tag_size_m / 2.0
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0],
         [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def parse_ref_tag_sizes(spec, ref_world):
    """Parse ``--ref-tag-size`` into ``{id: black-square edge in metres}``.

    Accepts a single number applied to every reference (``0.40``) or a
    per-tag map (``100:0.20;500:0.40``).  The pool frame mixes 0.20 m and
    0.40 m tags, and solving a 0.20 m tag as a 0.40 m one places it at twice
    its true range, which silently corrupts every pairwise distance gate --
    hence the per-tag form.
    """

    text = str(spec).strip()
    if not text:
        raise ValueError("reference tag size must not be empty")
    if ":" not in text:
        size = float(text)
        if size <= 0.0:
            raise ValueError("reference tag size must be positive")
        return {int(tag): size for tag in ref_world}

    sizes = {}
    for item in text.replace(",", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            tag, value = item.split(":", 1)
            sizes[int(tag)] = float(value)
        except ValueError as exc:
            raise ValueError(
                "reference tag size must look like '0.40' or "
                "'100:0.20;500:0.40'"
            ) from exc
    missing = sorted(set(ref_world) - set(sizes))
    if missing:
        raise ValueError(
            f"reference tag size has no entry for tag(s) {missing}")
    for tag, size in sizes.items():
        if size <= 0.0:
            raise ValueError(f"reference tag size for {tag} must be positive")
    return {tag: sizes[tag] for tag in ref_world}


def make_tag_object_map(tag_sizes):
    """``{id: size}`` -> ``{id: object points}`` for per-tag PnP."""

    return {int(tag): make_tag_object_points(float(size))
            for tag, size in tag_sizes.items()}


def tag_object_points(tag_obj, marker_id):
    """Object points for one tag, from either a per-tag map or one array.

    A plain array keeps every pre-four-tag caller working unchanged.
    """

    if isinstance(tag_obj, dict):
        return tag_obj.get(int(marker_id))
    return tag_obj


def detect_references(corners, ids, ref_world, tag_obj, K, geometry_dist):
    """Return raw detected centres plus independently checked tag poses."""

    seen = {}
    if ids is None:
        return seen
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        marker_id = int(marker_id)
        if marker_id not in ref_world:
            continue
        image_corners = marker_corners.reshape(-1, 2).astype(np.float64)
        center_px = quadrilateral_center(image_corners)
        marker_obj = tag_object_points(tag_obj, marker_id)
        if marker_obj is None:
            continue
        ok, rvec, tvec = cv2.solvePnP(
            marker_obj, image_corners, K, geometry_dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        pnp_center_px = np.full(2, np.nan)
        pnp_reproj_rms_px = np.nan
        if ok:
            projected_corners, _ = cv2.projectPoints(
                marker_obj, rvec, tvec, K, geometry_dist)
            projected_corners = projected_corners.reshape(-1, 2)
            pnp_reproj_rms_px = float(np.sqrt(np.mean(np.sum(
                (projected_corners - image_corners) ** 2, axis=1))))
            projected_center, _ = cv2.projectPoints(
                np.zeros((1, 3), dtype=np.float64), rvec, tvec, K,
                geometry_dist)
            pnp_center_px = projected_center.reshape(2)
        tvec_array = (
            np.asarray(tvec, dtype=float).reshape(3)
            if ok else np.full(3, np.nan))
        pose_valid = bool(
            ok
            and np.isfinite(tvec_array).all()
            and tvec_array[2] > 0.0
            and np.isfinite(pnp_reproj_rms_px)
            and pnp_reproj_rms_px <= MAX_REF_POSE_REPROJ_PX
        )
        seen[marker_id] = {
            # Never replace this observed location with a PnP projection.
            # A poor camera/distortion model can move that projection hundreds
            # of pixels away from the AprilTag that was actually detected.
            "center_px": center_px,
            "pnp_center_px": pnp_center_px,
            "pnp_reproj_rms_px": pnp_reproj_rms_px,
            "pose_valid": pose_valid,
            "range_m": (
                float(np.linalg.norm(tvec_array)) if pose_valid else np.nan),
            "corners_px": image_corners,
            # Keep a rejected raw solution in the debug CSV, but do not use it
            # for a baseline/range gate.
            "tvec": tvec_array,
        }
    return seen


def trail_segments(trail, limit=TRAIL_POINTS):
    """Consecutive ``(start, end)`` pairs over the most recent trail points.

    Written as an explicit pairing because the previous
    ``zip(trail[-400:], trail[-399:])`` silently degenerated: while the trail
    held 399 points or fewer, both slices returned the *same* list, so every
    "segment" ran from a point to itself and the overlay drew a string of
    dots.  The trail only became a line once it passed 400 points -- about
    40 s at 10 Hz -- which is exactly the dotted-then-solid behaviour seen in
    the tank.
    """

    recent = trail[-int(limit):]
    return list(zip(recent, recent[1:]))


class ReferenceRoiDetector:
    """Detect the reference tags inside padded boxes around their last sighting.

    Full-frame ``detectMarkers`` costs about 30 ms of the roughly 53 ms the
    tracking loop spends per 1920x1200 frame, and nearly all of it is the
    adaptive-threshold and contour sweep over 2.3 Mpx.  The overhead camera is
    fixed and the reference tags do not move, so after the first sighting each
    tag can be searched inside a small box instead.

    Correctness rests on one rule: a ROI pass is used only when it finds
    *every* tag the previous full-frame pass found.  Anything less -- a tag
    lost, an empty cache, a box that walked off the sensor -- falls back to a
    full-frame detection on that same frame, so the caller sees exactly what
    an always-full-frame tracker would have seen.  A ROI hit is not
    numerically identical to a full-frame hit, because aruco thresholds
    adaptively over whatever pixels it is given; the difference is a small
    fraction of a pixel on the refined corners, which is why the box carries
    generous padding rather than hugging the tag.
    """

    def __init__(self, detector, *, pad_ratio=0.4, refresh_frames=150):
        self.detector = detector
        self.pad_ratio = float(pad_ratio)
        self.refresh_frames = int(refresh_frames)
        self.boxes = {}
        self.roi_frames = 0
        self.full_frames = 0
        self._since_full = 0

    def _box_for(self, image_corners, shape):
        corners = np.asarray(image_corners, dtype=float).reshape(-1, 2)
        low = corners.min(axis=0)
        high = corners.max(axis=0)
        pad = self.pad_ratio * float(np.max(high - low))
        height, width = shape[:2]
        x0 = int(max(0, np.floor(low[0] - pad)))
        y0 = int(max(0, np.floor(low[1] - pad)))
        x1 = int(min(width, np.ceil(high[0] + pad)))
        y1 = int(min(height, np.ceil(high[1] + pad)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return (x0, y0, x1, y1)

    def _remember(self, corners, ids, shape):
        boxes = {}
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                box = self._box_for(marker_corners, shape)
                if box is not None:
                    boxes[int(marker_id)] = box
        self.boxes = boxes

    def _detect_full(self, gray):
        corners, ids, _ = self.detector.detectMarkers(gray)
        self.full_frames += 1
        self._since_full = 0
        self._remember(corners, ids, gray.shape)
        return corners, ids

    def detect(self, gray, wanted_ids):
        """Return ``(corners, ids)`` for ``wanted_ids`` in full-frame pixels."""

        wanted = {int(value) for value in wanted_ids}
        cached = {
            tag: box for tag, box in self.boxes.items() if tag in wanted}
        # Periodically sweep the whole sensor so a reference that was missing
        # when the cache was built can rejoin, and so a slowly creeping box
        # cannot drift with the tag it is tracking.
        if not cached or self._since_full >= self.refresh_frames:
            return self._detect_full(gray)

        found_corners = []
        found_ids = []
        for tag, (x0, y0, x1, y1) in cached.items():
            roi_corners, roi_ids, _ = self.detector.detectMarkers(
                gray[y0:y1, x0:x1])
            if roi_ids is None:
                continue
            for marker_corners, marker_id in zip(
                    roi_corners, roi_ids.flatten()):
                if int(marker_id) != tag:
                    continue
                shifted = marker_corners.copy()
                shifted[..., 0] += x0
                shifted[..., 1] += y0
                found_corners.append(shifted)
                found_ids.append(tag)

        if set(found_ids) != set(cached):
            # A reference went missing inside its box: the tag may have been
            # occluded, or the camera may have moved.  Only a full sweep can
            # tell those apart, and the drift monitor must not be fed a
            # partial view that a full frame would not have produced.
            return self._detect_full(gray)

        self.roi_frames += 1
        self._since_full += 1
        self._remember(
            found_corners,
            np.asarray(found_ids, dtype=np.int32).reshape(-1, 1),
            gray.shape,
        )
        return found_corners, np.asarray(
            found_ids, dtype=np.int32).reshape(-1, 1)


def undistort_pts(pts, K, dist):
    """(N,2) raw pixel coords -> undistorted pixel coords."""
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(p, K, dist, P=K).reshape(-1, 2)


def project(H, pts):
    """(N,2) undistorted px -> world metres via homography."""
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def world_to_pixels(H_mat, K, dist, use_raw_geometry, points_m):
    """World metres -> raw display pixels through the frozen map.

    The inverse of the mapping the tracker uses to turn marker pixels into
    pool coordinates, so anything expressed in pool metres -- the reference
    rectangle, a safe-operating inset, a waypoint pattern -- can be drawn
    on the live frame.
    """

    undistorted = project(
        np.linalg.inv(H_mat), np.asarray(points_m, dtype=float))
    if use_raw_geometry:
        return undistorted
    return distort_pixels(undistorted, K, dist)


def fill_translucent(frame, polygon_px, color, alpha):
    """Alpha-fill a small polygon without copying the whole frame.

    The overhead frame is 1920x1080 and this runs every captured cycle, so
    the blend is confined to the polygon's bounding box rather than done
    over the full image.
    """

    poly = np.asarray(polygon_px, dtype=np.int32).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(poly)
    x0, y0 = max(int(x), 0), max(int(y), 0)
    x1 = min(int(x + w), frame.shape[1])
    y1 = min(int(y + h), frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    patch = roi.copy()
    cv2.fillPoly(patch, [poly - np.array([x0, y0])], color, cv2.LINE_AA)
    cv2.addWeighted(patch, alpha, roi, 1.0 - alpha, 0.0, roi)


def draw_pool_overlay(frame, H_mat, K, dist, use_raw_geometry, *,
                      rect_wh, margin_m=0.0, waypoints=(),
                      arrival_radius_m=0.0, heading_deg=None,
                      headings_deg=None, visited=(), align_states=()):
    """Draw the pool frame, the safe region and the waypoint pattern.

    Everything here is defined in pool metres and projected through the
    frozen homography, so the overlay is the *same* geometry the guidance
    uses rather than a redrawing of it -- if the calibration drifts, the
    overlay drifts with it and the discrepancy is visible.

    ``rect_wh`` is the reference rectangle the tags define; ``margin_m``
    insets it to the region the pattern is allowed to use.

    Two heading conventions, and only one is drawn.  ``headings_deg`` is the
    per-waypoint hold setpoint the pattern declares -- the bow direction the
    heading hold actually drives to at each target -- and gets one arrow per
    waypoint.  ``heading_deg`` is the single fallback used when a pattern
    declares no per-waypoint headings, and gets one arrow at the centroid.
    When both are supplied the per-waypoint arrows win, because drawing the
    fallback alongside them would show a setpoint nothing commands.

    ``align_states`` recolors the per-waypoint arrows from the guidance's
    settle verdict, index-matched to ``waypoints``: 1 turns the arrow
    green (the vehicle settled on this heading inside the ring), 2 turns
    it orange (the settle timed out), anything else keeps the default.
    ``visited`` stays the tracker's OWN geometric ring test, so the disc
    fill and the arrow color remain independent witnesses.
    """

    if H_mat is None or rect_wh is None:
        return

    def to_px(points_m):
        return np.rint(world_to_pixels(
            H_mat, K, dist, use_raw_geometry, points_m)).astype(np.int32)

    width, height = float(rect_wh[0]), float(rect_wh[1])
    rect = np.array([(0.0, 0.0), (width, 0.0), (width, height),
                     (0.0, height)])
    cv2.polylines(frame, [to_px(rect).reshape(-1, 1, 2)], True,
                  (34, 59, 194), 2, cv2.LINE_AA)

    if margin_m > 0.0 and 2 * margin_m < min(width, height):
        inset = np.array([
            (margin_m, margin_m), (width - margin_m, margin_m),
            (width - margin_m, height - margin_m),
            (margin_m, height - margin_m)])
        cv2.polylines(frame, [to_px(inset).reshape(-1, 1, 2)], True,
                      (34, 123, 194), 1, cv2.LINE_AA)

    points = np.asarray(waypoints, dtype=float).reshape(-1, 2)
    if len(points) >= 2:
        cv2.polylines(frame, [to_px(points).reshape(-1, 1, 2)], True,
                      (178, 111, 31), 2, cv2.LINE_AA)
    visited_set = {int(i) for i in visited}
    for index, waypoint in enumerate(points):
        pixel = to_px([waypoint])[0]
        ring_px = None
        if arrival_radius_m > 0.0:
            # Project the radius rather than assuming a pixel scale: the
            # overhead view is perspective, so the ring is an ellipse and
            # its size changes across the pool.
            ring = waypoint + arrival_radius_m * np.column_stack([
                np.cos(np.linspace(0, 2 * np.pi, 24)),
                np.sin(np.linspace(0, 2 * np.pi, 24))])
            ring_px = to_px(ring)
        # Filled first so the outline, index and arrow stay legible on top.
        if index in visited_set:
            fill = ring_px
            if fill is None:
                angles = np.linspace(0, 2 * np.pi, 20)
                fill = np.rint(np.asarray(pixel, dtype=float) + 9.0
                               * np.column_stack([np.cos(angles),
                                                  np.sin(angles)])
                               ).astype(np.int32)
            fill_translucent(frame, fill, (58, 157, 42), 0.35)
        cv2.circle(frame, tuple(pixel), 9, (58, 157, 42), 2, cv2.LINE_AA)
        if ring_px is not None:
            cv2.polylines(frame, [ring_px.reshape(-1, 1, 2)], True,
                          (58, 157, 42), 1, cv2.LINE_AA)
        cv2.putText(frame, str(index), (pixel[0] + 12, pixel[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (58, 157, 42), 2,
                    cv2.LINE_AA)

    per_waypoint = (headings_deg is not None
                    and len(headings_deg) == len(points) and len(points))
    if per_waypoint:
        # One arrow per waypoint, from the waypoint outward along its own
        # hold setpoint.  Drawn a little longer than the arrival ring so the
        # two never read as one blob.  The color is the settle verdict:
        # green once the vehicle held this heading inside the ring, orange
        # if the settle timed out, the default purple while pending.
        length = max(0.28, 1.6 * float(arrival_radius_m))
        for index, (waypoint, heading) in enumerate(
                zip(points, headings_deg)):
            state = int(align_states[index]) if index < len(align_states) \
                else 0
            color = {1: (60, 220, 60), 2: (0, 165, 255)}.get(
                state, (138, 75, 91))
            psi = math.radians(float(heading))
            tip = waypoint + length * np.array(
                [math.cos(psi), math.sin(psi)])
            base_px, tip_px = to_px([waypoint, tip])
            cv2.arrowedLine(frame, tuple(base_px), tuple(tip_px),
                            color, 4, cv2.LINE_AA, tipLength=0.30)
    elif heading_deg is not None and len(points):
        psi = math.radians(float(heading_deg))
        centre = points.mean(axis=0)
        tip = centre + 0.35 * np.array([math.cos(psi), math.sin(psi)])
        base_px, tip_px = to_px([centre, tip])
        cv2.arrowedLine(frame, tuple(base_px), tuple(tip_px),
                        (138, 75, 91), 3, cv2.LINE_AA, tipLength=0.3)


def distort_pixels(undistorted_pixels, K, dist):
    """Convert undistorted pixel coordinates back to raw display pixels."""

    pixels = np.asarray(undistorted_pixels, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack(
        [pixels, np.ones(len(pixels), dtype=float)])
    normalized = (np.linalg.inv(K) @ homogeneous.T).T
    rays = np.column_stack([
        normalized[:, 0] / normalized[:, 2],
        normalized[:, 1] / normalized[:, 2],
        np.ones(len(normalized), dtype=float),
    ])
    raw, _ = cv2.projectPoints(
        rays, np.zeros(3), np.zeros(3), K, dist)
    return raw.reshape(-1, 2)


def pool_heading_image_direction(H, raw_anchor_px, K, dist, heading_rad):
    """Unit image direction of a pool-frame bearing at a raw-pixel anchor.

    Projected through the homography rather than rotated in image space.
    The overhead view is perspective, so a rotation in the pool plane is
    not a rotation in the image plane: spinning the pixel vector instead
    would tilt the arrow by several degrees away from the map centre, and
    the drawn angle would stop agreeing with the printed one.
    """

    step = np.array([math.cos(heading_rad), math.sin(heading_rad)])
    if H is None:
        # Preview mode has no pool frame.  Image y runs downward, which is
        # the convention the preview heading is measured in already.
        return np.array([step[0], -step[1]])
    try:
        inverse_H = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.array([1.0, 0.0])
    anchor_undistorted = undistort_pts([raw_anchor_px], K, dist)
    anchor_world = project(H, anchor_undistorted)[0]
    pool_points = np.asarray([anchor_world, anchor_world + step])
    undistorted_pixels = project(inverse_H, pool_points)
    raw_pixels = distort_pixels(undistorted_pixels, K, dist)
    direction = raw_pixels[1] - raw_pixels[0]
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length <= 1e-9:
        return np.array([1.0, 0.0])
    return direction / length


def pool_x_image_direction(H, raw_anchor_px, K, dist):
    """Unit image direction of local pool +x at a raw-pixel anchor."""

    return pool_heading_image_direction(H, raw_anchor_px, K, dist, 0.0)


def reference_positions(ref_seen, H, K, dist):
    """Return live camera-frame and optional pool-frame reference positions."""

    result = {}
    for tag, detection in ref_seen.items():
        camera_xyz = np.asarray(detection["tvec"], dtype=float).reshape(3)
        pool_xy = np.full(2, np.nan)
        if H is not None:
            centre_undistorted = undistort_pts(
                [detection["center_px"]], K, dist)
            pool_xy = project(H, centre_undistorted)[0]
        result[tag] = {
            "camera_xyz_m": camera_xyz,
            "pool_xy_m": pool_xy,
            "range_m": float(detection["range_m"]),
            "pose_valid": bool(detection.get("pose_valid", False)),
            "pnp_reproj_rms_px": float(
                detection.get("pnp_reproj_rms_px", np.nan)),
        }
    return result


def reference_baseline(ref_positions, ref_world):
    """Return IDs plus live PnP and expected distance for the first two refs."""

    common = [tag for tag in ref_world if tag in ref_positions]
    if len(common) < 2:
        return None, None, np.nan, np.nan
    tag_a, tag_b = common[:2]
    camera_a = ref_positions[tag_a]["camera_xyz_m"]
    camera_b = ref_positions[tag_b]["camera_xyz_m"]
    measured = np.nan
    if (ref_positions[tag_a].get("pose_valid", False)
            and ref_positions[tag_b].get("pose_valid", False)
            and np.isfinite(camera_a).all()
            and np.isfinite(camera_b).all()):
        measured = float(np.linalg.norm(camera_b - camera_a))
    expected = float(np.linalg.norm(
        np.asarray(ref_world[tag_b], dtype=float)
        - np.asarray(ref_world[tag_a], dtype=float)))
    return tag_a, tag_b, measured, expected


def csv_float(value, digits=6):
    """Format one finite scalar for CSV, otherwise return an empty field."""

    value = float(value)
    return f"{value:.{digits}f}" if np.isfinite(value) else ""


def calibration_debug_header(ref_ids, n_dist_coeffs):
    """Column names for replayable per-frame calibration diagnostics.

    Schema 4 changed ``ref_tag_size_m`` from a single number to the per-tag
    ``id:size;...`` spec, because the pool frame mixes 0.20 m and 0.40 m
    references and one number can no longer describe a run.
    """

    header = [
        "calibration_debug_schema_version",
        "frame_width_px",
        "frame_height_px",
        "ref_tag_size_m",
        "visible_ref_ids",
        "candidate_available",
        "candidate_accepted",
        "candidate_reject_reason",
        "candidate_method",
        "candidate_world_rms_m",
        "candidate_pnp_baseline_error_m",
        "pnp_baseline_tag_a",
        "pnp_baseline_tag_b",
        "pnp_baseline_measured_m",
        "pnp_baseline_expected_m",
        "map_scale_source",
        "map_baseline_used_m",
        "map_scale_vs_configured",
        "accepted_calibration_samples",
        "max_ref_world_error_m",
        "max_ref_range_error_m",
        "max_ref_pose_reproj_error_px",
        "geometry_distortion_mode",
    ]
    header.extend(
        f"camera_K_{row}{column}"
        for row in range(3)
        for column in range(3)
    )
    header.extend(
        f"camera_dist_{index}"
        for index in range(n_dist_coeffs)
    )
    header.extend(
        f"geometry_dist_{index}"
        for index in range(n_dist_coeffs)
    )
    for tag in ref_ids:
        prefix = f"ref_{tag}"
        header.extend([
            f"{prefix}_detected",
            f"{prefix}_target_x_m",
            f"{prefix}_target_y_m",
            f"{prefix}_center_px_x",
            f"{prefix}_center_px_y",
            f"{prefix}_pnp_pose_valid",
            f"{prefix}_pnp_reproj_rms_px",
            f"{prefix}_pnp_center_px_x",
            f"{prefix}_pnp_center_px_y",
        ])
        for corner in range(4):
            header.extend([
                f"{prefix}_corner{corner}_px_x",
                f"{prefix}_corner{corner}_px_y",
            ])
        header.extend([
            f"{prefix}_cam_x_m",
            f"{prefix}_cam_y_m",
            f"{prefix}_cam_z_m",
            f"{prefix}_range_m",
            f"{prefix}_map_x_m",
            f"{prefix}_map_y_m",
        ])
    for name in ("candidate_H", "active_map_H"):
        header.extend(
            f"{name}_{row}{column}"
            for row in range(3)
            for column in range(3)
        )
    return header


def calibration_debug_row(
        ref_ids, ref_world, ref_seen, ref_live_positions,
        candidate_H, active_map_H, candidate_quality, active_map_quality,
        candidate_accepted, candidate_reject_reason,
        baseline_a, baseline_b, baseline_measured, baseline_expected,
        accepted_samples, max_ref_world_error, max_ref_range_error,
        frame_shape, ref_tag_size, K, camera_dist, geometry_dist,
        geometry_distortion_mode):
    """Values matching :func:`calibration_debug_header`."""

    map_quality = (
        active_map_quality if active_map_H is not None else candidate_quality)
    row = [
        4,
        int(frame_shape[1]),
        int(frame_shape[0]),
        (";".join(f"{tag}:{float(size):.4f}"
                  for tag, size in sorted(ref_tag_size.items()))
         if isinstance(ref_tag_size, dict) else csv_float(ref_tag_size)),
        ";".join(str(tag) for tag in ref_ids if tag in ref_seen),
        int(candidate_H is not None),
        int(candidate_accepted),
        candidate_reject_reason,
        candidate_quality["method"],
        csv_float(candidate_quality["world_rms_m"]),
        csv_float(candidate_quality["range_rms_m"]),
        "" if baseline_a is None else baseline_a,
        "" if baseline_b is None else baseline_b,
        csv_float(baseline_measured),
        csv_float(baseline_expected),
        map_quality.get("map_scale_source", "unavailable"),
        csv_float(map_quality.get("map_baseline_m", np.nan)),
        csv_float(map_quality.get("map_scale_factor", np.nan)),
        accepted_samples,
        csv_float(max_ref_world_error),
        csv_float(max_ref_range_error),
        csv_float(MAX_REF_POSE_REPROJ_PX),
        geometry_distortion_mode,
    ]
    row.extend(
        csv_float(value, 10)
        for value in np.asarray(K, dtype=float).reshape(9)
    )
    row.extend(
        csv_float(value, 10)
        for value in np.asarray(camera_dist, dtype=float).reshape(-1)
    )
    row.extend(
        csv_float(value, 10)
        for value in np.asarray(geometry_dist, dtype=float).reshape(-1)
    )
    for tag in ref_ids:
        target = np.asarray(ref_world[tag], dtype=float)
        if tag not in ref_seen:
            row.extend([
                0, csv_float(target[0]), csv_float(target[1]),
                *([""] * 20),
            ])
            continue
        detection = ref_seen[tag]
        live = ref_live_positions[tag]
        center = np.asarray(detection["center_px"], dtype=float)
        pnp_center = np.asarray(
            detection["pnp_center_px"], dtype=float).reshape(2)
        corners = np.asarray(detection["corners_px"], dtype=float).reshape(4, 2)
        camera = np.asarray(detection["tvec"], dtype=float).reshape(3)
        mapped = np.asarray(live["pool_xy_m"], dtype=float).reshape(2)
        row.extend([
            1,
            csv_float(target[0]),
            csv_float(target[1]),
            csv_float(center[0], 3),
            csv_float(center[1], 3),
            int(detection["pose_valid"]),
            csv_float(detection["pnp_reproj_rms_px"], 3),
            csv_float(pnp_center[0], 3),
            csv_float(pnp_center[1], 3),
        ])
        for corner in corners:
            row.extend([csv_float(corner[0], 3), csv_float(corner[1], 3)])
        row.extend([
            csv_float(camera[0]),
            csv_float(camera[1]),
            csv_float(camera[2]),
            csv_float(detection["range_m"]),
            csv_float(mapped[0]),
            csv_float(mapped[1]),
        ])
    for matrix in (candidate_H, active_map_H):
        if matrix is None:
            row.extend([""] * 9)
        else:
            row.extend(
                csv_float(value, 10)
                for value in np.asarray(matrix, dtype=float).reshape(9)
            )
    return row


def draw_text_block(frame, lines, anchor, color):
    """Draw a compact readable label near a detected reference tag."""

    lines = [str(line) for line in lines if line]
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    padding = 5
    line_height = 19
    widths = [
        cv2.getTextSize(line, font, scale, thickness)[0][0]
        for line in lines
    ]
    block_width = max(widths) + 2 * padding
    block_height = len(lines) * line_height + 2 * padding
    frame_h, frame_w = frame.shape[:2]
    x = int(np.clip(anchor[0] + 12, 2,
                    max(2, frame_w - block_width - 2)))
    y = anchor[1] + 12
    if y + block_height > frame_h - 2:
        y = anchor[1] - block_height - 12
    y = int(np.clip(y, 2, max(2, frame_h - block_height - 2)))
    cv2.rectangle(
        frame, (x, y), (x + block_width, y + block_height),
        (20, 20, 20), cv2.FILLED)
    for index, line in enumerate(lines):
        baseline = y + padding + (index + 1) * line_height - 4
        cv2.putText(
            frame, line, (x + padding, baseline), font, scale, color,
            thickness, cv2.LINE_AA)


def quadrilateral_center(corners):
    """Projective centre of a four-corner marker (diagonal intersection)."""

    points = np.asarray(corners, dtype=float).reshape(4, 2)
    system = np.column_stack((points[2] - points[0],
                              -(points[3] - points[1])))
    try:
        fraction = np.linalg.solve(system, points[1] - points[0])[0]
    except np.linalg.LinAlgError:
        return points.mean(axis=0)
    return points[0] + fraction * (points[2] - points[0])


def segment_similarity(source_a, source_b, target_a, target_b):
    """Return a 2-D similarity taking one directed segment onto another."""

    source_a = np.asarray(source_a, dtype=float)
    source_b = np.asarray(source_b, dtype=float)
    target_a = np.asarray(target_a, dtype=float)
    target_b = np.asarray(target_b, dtype=float)
    source_delta = source_b - source_a
    target_delta = target_b - target_a
    source_length = float(np.linalg.norm(source_delta))
    target_length = float(np.linalg.norm(target_delta))
    if source_length <= 1e-12 or target_length <= 1e-12:
        return None

    cosine = float(source_delta @ target_delta
                   / (source_length * target_length))
    cross = (source_delta[0] * target_delta[1]
             - source_delta[1] * target_delta[0])
    sine = float(cross / (source_length * target_length))
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=float)
    scale = target_length / source_length
    translation = target_a - scale * rotation @ source_a
    similarity = np.eye(3, dtype=float)
    similarity[:2, :2] = scale * rotation
    similarity[:2, 2] = translation
    return similarity


def estimate_two_tag_homography(
        common, ref_seen, ref_world, tag_obj, K, dist,
        baseline_scale_source="configured"):
    """Estimate the plane map from two complete, coplanar square detections.

    Each square first supplies an independent image-to-plane homography.
    Mapping the detected centres through each local plane and aligning the
    directed centre baseline to the measured world baseline removes dependence
    on the printed in-plane rotation of either tag.  A short alternating fit
    then estimates each tag's in-plane rotation and refits one homography to
    all eight corners.  This joint fit is much better conditioned across the
    pool than extrapolating either small square by itself.
    """

    failure_scale = {
        "source": "unavailable", "baseline_m": np.nan,
        "factor": np.nan,
    }
    if tag_obj is None:
        return None, np.nan, np.nan, failure_scale

    tag_a, tag_b = common
    target_a = np.asarray(ref_world[tag_a], dtype=float)
    target_b = np.asarray(ref_world[tag_b], dtype=float)
    configured_baseline = float(np.linalg.norm(target_b - target_a))
    if configured_baseline <= 1e-12:
        return None, np.nan, np.nan, {
            "source": "invalid-configured-baseline",
            "baseline_m": np.nan,
            "factor": np.nan,
        }

    tvec_a = np.asarray(ref_seen[tag_a]["tvec"], dtype=float)
    tvec_b = np.asarray(ref_seen[tag_b]["tvec"], dtype=float)
    pose_a_valid = bool(ref_seen[tag_a].get(
        "pose_valid", np.isfinite(tvec_a).all() and tvec_a[2] > 0.0))
    pose_b_valid = bool(ref_seen[tag_b].get(
        "pose_valid", np.isfinite(tvec_b).all() and tvec_b[2] > 0.0))
    pnp_baseline = np.nan
    if (pose_a_valid and pose_b_valid
            and np.isfinite(tvec_a).all() and np.isfinite(tvec_b).all()):
        pnp_baseline = float(np.linalg.norm(tvec_b - tvec_a))

    map_scale_source = "configured"
    map_baseline = configured_baseline
    map_scale_factor = 1.0
    if (baseline_scale_source == "camera"
            and np.isfinite(pnp_baseline) and pnp_baseline > 1e-12):
        map_scale_source = "camera-pnp"
        map_baseline = pnp_baseline
        map_scale_factor = pnp_baseline / configured_baseline
        # Preserve the configured pool origin and axis directions, but let the
        # independently recovered camera baseline set the metric map scale.
        target_a = target_a * map_scale_factor
        target_b = target_b * map_scale_factor
    target_centres = {tag_a: target_a, tag_b: target_b}
    # Per-tag squares: the two references need not be the same size.
    local_corners = {}
    for tag in common:
        marker_obj = tag_object_points(tag_obj, tag)
        if marker_obj is None:
            return None, np.nan, np.nan, failure_scale
        local_corners[tag] = np.asarray(marker_obj, dtype=float)[:, :2]
    undistorted_corners = {
        tag: undistort_pts(ref_seen[tag]["corners_px"], K, dist)
        for tag in common
    }
    undistorted_centres = {
        tag: quadrilateral_center(undistorted_corners[tag])
        for tag in common
    }

    candidates = []
    for anchor in common:
        local_H, _ = cv2.findHomography(
            undistorted_corners[anchor], local_corners[anchor], method=0)
        if local_H is None:
            continue
        source_a, source_b = project(
            local_H,
            [undistorted_centres[tag_a], undistorted_centres[tag_b]],
        )
        similarity = segment_similarity(
            source_a, source_b, target_a, target_b)
        if similarity is None:
            continue
        candidate = similarity @ local_H
        if abs(candidate[2, 2]) > 1e-12:
            candidate /= candidate[2, 2]
        candidates.append(candidate)

    if len(candidates) != 2:
        return None, np.nan, np.nan, failure_scale

    probes = np.vstack([undistorted_corners[tag] for tag in common])
    H = average_homographies(candidates)

    def fitted_world_corners(current_H):
        targets = []
        for tag in common:
            mapped = project(current_H, undistorted_corners[tag])
            target_center = target_centres[tag]
            centred_mapped = mapped - target_center
            left, _, right_t = np.linalg.svd(
                local_corners[tag].T @ centred_mapped)
            row_rotation = left @ right_t
            if np.linalg.det(row_rotation) < 0.0:
                left = left.copy()
                left[:, -1] *= -1.0
                row_rotation = left @ right_t
            targets.append(
                target_center + local_corners[tag] @ row_rotation)
        return np.vstack(targets)

    world_corner_targets = fitted_world_corners(H)
    for _ in range(15):
        refined_H, _ = cv2.findHomography(
            probes, world_corner_targets, method=0)
        if refined_H is None:
            return None, np.nan, np.nan, failure_scale
        if abs(refined_H[2, 2]) > 1e-12:
            refined_H /= refined_H[2, 2]
        change = float(np.linalg.norm(refined_H - H))
        H = refined_H
        world_corner_targets = fitted_world_corners(H)
        if change < 1e-10:
            break

    # One final fit makes H correspond exactly to the most recent rotations.
    H, _ = cv2.findHomography(probes, world_corner_targets, method=0)
    if H is None:
        return None, np.nan, np.nan, failure_scale
    if abs(H[2, 2]) > 1e-12:
        H /= H[2, 2]
    corner_fit_rms = float(np.sqrt(np.mean(np.sum(
        (project(H, probes) - world_corner_targets) ** 2, axis=1))))

    baseline_error = np.nan
    if np.isfinite(pnp_baseline):
        baseline_error = abs(pnp_baseline - configured_baseline)
    return H, corner_fit_rms, baseline_error, {
        "source": map_scale_source,
        "baseline_m": map_baseline,
        "factor": map_scale_factor,
    }


def estimate_reference_homography(
        ref_seen, ref_world, K, dist, tag_obj=None,
        baseline_scale_source="configured"):
    """Estimate undistorted-pixel -> reference-plane metres.

    Four or more centres directly constrain a projective homography.  With
    exactly three centres, calibrated SQPnP supplies the candidate camera
    poses and the independent per-tag IPPE ranges select the physical branch.
    With exactly two reference tags, all eight detected square corners supply
    two independent baseline-aligned plane homographies.
    """

    common = [tag for tag in ref_world if tag in ref_seen]
    if len(common) < 2:
        return None, {"method": "insufficient", "world_rms_m": np.nan,
                      "range_rms_m": np.nan}

    world_xy = np.asarray([ref_world[tag] for tag in common], np.float64)
    raw_px = np.asarray(
        [ref_seen[tag]["center_px"] for tag in common], np.float64)
    und_px = undistort_pts(raw_px, K, dist)

    two_tag_corner_fit = np.nan
    map_scale = {"source": "configured", "baseline_m": np.nan,
                 "factor": 1.0}
    if len(common) == 2:
        H, two_tag_corner_fit, range_rms, map_scale = (
            estimate_two_tag_homography(
                common, ref_seen, ref_world, tag_obj, K, dist,
                baseline_scale_source=baseline_scale_source))
        if H is None:
            return None, {"method": "2-tag-corner-failed",
                          "world_rms_m": np.nan, "range_rms_m": np.nan}
        method = "2-tag corner homography"
    elif len(common) >= 4:
        world_to_pixel, inliers = cv2.findHomography(
            world_xy, und_px, cv2.RANSAC, 2.0)
        if world_to_pixel is None:
            return None, {"method": "homography-failed",
                          "world_rms_m": np.nan, "range_rms_m": np.nan}
        H = np.linalg.inv(world_to_pixel)
        method = f"{len(common)}-tag homography"
        range_rms = np.nan
    else:
        object_points = np.column_stack(
            [world_xy, np.zeros(len(world_xy), dtype=np.float64)])
        result = cv2.solvePnPGeneric(
            object_points, raw_px, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not result[0]:
            return None, {"method": "SQPnP-failed",
                          "world_rms_m": np.nan, "range_rms_m": np.nan}

        observed_ranges = np.asarray(
            [ref_seen[tag]["range_m"] for tag in common], dtype=float)
        candidates = []
        for rvec, tvec in zip(result[1], result[2]):
            rotation, _ = cv2.Rodrigues(rvec)
            tvec = np.asarray(tvec, dtype=float).reshape(3)
            camera_world = -rotation.T @ tvec
            camera_points = (
                rotation @ object_points.T + tvec.reshape(3, 1)).T
            if camera_world[2] <= 0.0 or np.any(camera_points[:, 2] <= 0.0):
                continue
            predicted_ranges = np.linalg.norm(
                object_points - camera_world, axis=1)
            valid_ranges = np.isfinite(observed_ranges)
            if not valid_ranges.any():
                continue
            range_rms_candidate = float(np.sqrt(np.mean(
                (predicted_ranges[valid_ranges]
                 - observed_ranges[valid_ranges]) ** 2)))
            world_to_pixel = K @ np.column_stack(
                (rotation[:, 0], rotation[:, 1], tvec))
            try:
                candidate_H = np.linalg.inv(world_to_pixel)
            except np.linalg.LinAlgError:
                continue
            candidates.append(
                (range_rms_candidate, candidate_H, camera_world))

        if not candidates:
            return None, {"method": "SQPnP-no-physical-branch",
                          "world_rms_m": np.nan, "range_rms_m": np.nan}
        range_rms, H, camera_world = min(
            candidates, key=lambda candidate: candidate[0])
        method = "3-tag calibrated SQPnP+range"

    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    recovered = project(H, und_px)
    fit_world_xy = world_xy * float(map_scale["factor"])
    world_rms = float(np.sqrt(np.mean(np.sum(
        (recovered - fit_world_xy) ** 2, axis=1))))
    if np.isfinite(two_tag_corner_fit):
        world_rms = max(world_rms, two_tag_corner_fit)
    quality = {
        "method": method,
        "world_rms_m": world_rms,
        "range_rms_m": range_rms,
        "map_scale_source": map_scale["source"],
        "map_baseline_m": map_scale["baseline_m"],
        "map_scale_factor": map_scale["factor"],
        "pnp_valid": all(bool(ref_seen[tag].get(
            "pose_valid",
            np.isfinite(ref_seen[tag]["tvec"]).all()
            and np.asarray(ref_seen[tag]["tvec"])[2] > 0.0))
            for tag in common),
    }
    if len(common) == 3:
        quality["camera_world"] = camera_world
    if len(common) >= 4:
        quality["n_inliers"] = int(np.count_nonzero(inliers))
    return H, quality


def save_pool_map(path, H, K, dist, ref_world, ref_tag_sizes, frame_shape,
                  geometry_distortion_mode, quality, n_samples):
    """Persist a frozen pool map so a later run can reuse it.

    Tag detection can fail for reasons that have nothing to do with the
    camera having moved -- low sun, glare off the water, a wet tag, someone
    standing in front of one.  Without a saved map such a run cannot start at
    all.  With one it can, PROVIDED the camera has not moved, which is why
    everything needed to re-verify that is stored alongside the homography:
    the intrinsics, the layout, the frame size and the distortion mode it was
    computed under.  :func:`load_pool_map` refuses a map whose context
    differs, because silently reusing one taken under a different geometry
    would put a plausible-looking but wrong position into the controller.
    """

    import json

    payload = {
        "schema": 1,
        "created_unix": time.time(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "H": np.asarray(H, dtype=float).tolist(),
        "camera_matrix": np.asarray(K, dtype=float).tolist(),
        "dist_coeffs": np.asarray(dist, dtype=float).ravel().tolist(),
        "frame_width": int(frame_shape[1]),
        "frame_height": int(frame_shape[0]),
        "geometry_distortion_mode": str(geometry_distortion_mode),
        "ref_world": {str(t): [float(x), float(y)]
                      for t, (x, y) in ref_world.items()},
        "ref_tag_size_m": {str(t): float(s)
                           for t, s in ref_tag_sizes.items()},
        "accepted_samples": int(n_samples),
        "method": str(quality.get("method", "")),
        "world_rms_m": float(quality.get("world_rms_m", float("nan"))),
        "map_scale_source": str(quality.get("map_scale_source", "")),
        "map_scale_factor": float(quality.get("map_scale_factor", 1.0)),
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)          # atomic: never leave a half-written map
    return payload


def load_pool_map(path, K, dist, ref_world, ref_tag_sizes, frame_shape,
                  geometry_distortion_mode):
    """Load a saved pool map, refusing one taken under different conditions.

    Returns ``(H, payload, warnings)``.  Raises ``ValueError`` when the map
    cannot legitimately be applied to this run.
    """

    import json

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("schema", 0)) != 1:
        raise ValueError(f"{path}: unsupported pool-map schema")

    saved_world = {int(t): tuple(v)
                   for t, v in payload["ref_world"].items()}
    if saved_world.keys() != set(ref_world):
        raise ValueError(
            f"{path}: saved for tags {sorted(saved_world)}, this run uses "
            f"{sorted(ref_world)}")
    for tag, (x, y) in saved_world.items():
        if (abs(x - ref_world[tag][0]) > 1e-4
                or abs(y - ref_world[tag][1]) > 1e-4):
            raise ValueError(
                f"{path}: tag {tag} was at {saved_world[tag]}, this run uses "
                f"{tuple(ref_world[tag])} -- the layout changed, so the map "
                "describes a different frame")
    if (int(payload["frame_width"]) != int(frame_shape[1])
            or int(payload["frame_height"]) != int(frame_shape[0])):
        raise ValueError(
            f"{path}: saved at {payload['frame_width']}x"
            f"{payload['frame_height']}, this run captures "
            f"{frame_shape[1]}x{frame_shape[0]} -- pixel coordinates are not "
            "comparable")
    if str(payload["geometry_distortion_mode"]) != str(
            geometry_distortion_mode):
        raise ValueError(
            f"{path}: saved under distortion mode "
            f"{payload['geometry_distortion_mode']!r}, this run uses "
            f"{geometry_distortion_mode!r} -- the map maps DIFFERENT pixels")

    warnings = []
    saved_K = np.asarray(payload["camera_matrix"], dtype=float)
    if not np.allclose(saved_K, np.asarray(K, dtype=float), atol=1e-6):
        warnings.append(
            "camera_matrix differs from the saved one (recalibrated since?)")
    saved_dist = np.asarray(payload["dist_coeffs"], dtype=float).ravel()
    live_dist = np.asarray(dist, dtype=float).ravel()
    if (saved_dist.shape != live_dist.shape
            or not np.allclose(saved_dist, live_dist, atol=1e-6)):
        warnings.append("dist_coeffs differ from the saved one")
    saved_sizes = {int(t): float(s)
                   for t, s in payload.get("ref_tag_size_m", {}).items()}
    if saved_sizes and saved_sizes != {int(t): float(s)
                                       for t, s in ref_tag_sizes.items()}:
        warnings.append("reference tag sizes differ from the saved ones")

    age_s = time.time() - float(payload.get("created_unix", 0.0))
    warnings.append(
        f"map is {age_s / 3600.0:.1f} h old and assumes the camera has NOT "
        "moved since; verify against any visible tag before trusting it")
    H = np.asarray(payload["H"], dtype=float)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError(f"{path}: homography is not a finite 3x3")
    return H, payload, warnings


def average_homographies(homographies):
    """Robustly combine normalized static-camera homography samples."""

    normalized = []
    for H in homographies:
        H = np.asarray(H, dtype=float)
        scale = H[2, 2] if abs(H[2, 2]) > 1e-12 else np.linalg.norm(H)
        normalized.append(H / scale)
    H = np.median(np.stack(normalized), axis=0)
    return H / H[2, 2]


def pick_pair_metric(right_blobs, reds, H, K, dist, *, previous_mid,
                     previous_t, now, spacing_estimate, max_spacing_m,
                     max_speed_m_s, jump_slack_m, bounds):
    """Select a color pair using pool-plane geometry instead of pixel gates."""

    best = None
    best_cost = None
    xmin, xmax, ymin, ymax = bounds
    for right in right_blobs:
        for red in reds:
            und = undistort_pts(
                [(right[0], right[1]), (red[0], red[1])], K, dist)
            right_world, red_world = project(H, und)
            spacing = float(np.linalg.norm(right_world - red_world))
            if not np.isfinite(spacing) or spacing > max_spacing_m:
                continue
            midpoint = (right_world + red_world) / 2.0
            if not (xmin <= midpoint[0] <= xmax
                    and ymin <= midpoint[1] <= ymax):
                continue

            jump = 0.0
            if previous_mid is not None and previous_t is not None:
                dt = max(now - previous_t, 0.0)
                jump = float(np.linalg.norm(midpoint - previous_mid))
                if jump > jump_slack_m + max_speed_m_s * dt:
                    continue
            if spacing_estimate is None:
                spacing_cost = spacing
            else:
                spacing_cost = 3.0 * abs(spacing - spacing_estimate)
            cost = spacing_cost + jump
            if best_cost is None or cost < best_cost:
                best = (right, red, right_world, red_world, spacing)
                best_cost = cost
    return best


@dataclass(frozen=True)
class MarkerQualityDecision:
    """Auditable result of the pre-derivative marker-identity gate."""

    accepted: bool
    reason: str
    reset_motion: bool
    position_jump_m: float = np.nan
    heading_jump_rad: float = np.nan
    spacing_error_m: float = np.nan


class MarkerObservationGate:
    """Reject color-pair identity discontinuities before differentiation.

    The blob association stage is intentionally permissive enough to retain a
    raw candidate for diagnostics.  This gate applies the rigid-bar geometry
    and last-good identity constraints.  Rejections never update its state, so
    one bad association cannot become the reference for the next frame.
    """

    def __init__(
        self,
        *,
        min_spacing_m=QUALITY_MIN_COLOR_SPACING_M,
        max_spacing_m=QUALITY_MAX_COLOR_SPACING_M,
        max_spacing_error_m=QUALITY_MAX_SPACING_ERROR_M,
        max_heading_step_rad=math.radians(
            QUALITY_MAX_HEADING_STEP_DEG),
        max_speed_m_s=QUALITY_MAX_SPEED_M_S,
        jump_slack_m=QUALITY_JUMP_SLACK_M,
        identity_memory_s=QUALITY_IDENTITY_MEMORY_S,
        spacing_alpha=0.05,
    ):
        values = (
            min_spacing_m, max_spacing_m, max_spacing_error_m,
            max_heading_step_rad, max_speed_m_s, jump_slack_m,
            identity_memory_s, spacing_alpha,
        )
        if not np.isfinite(values).all():
            raise ValueError("marker quality thresholds must be finite")
        if min_spacing_m < 0.0 or max_spacing_m <= min_spacing_m:
            raise ValueError(
                "marker spacing limits must satisfy 0 <= min < max")
        if max_spacing_error_m <= 0.0:
            raise ValueError("maximum marker spacing error must be positive")
        if not 0.0 < max_heading_step_rad <= math.pi:
            raise ValueError(
                "maximum marker heading step must be in (0, pi]")
        if max_speed_m_s < 0.0 or jump_slack_m < 0.0:
            raise ValueError(
                "marker speed and position-jump slack must be nonnegative")
        if identity_memory_s <= 0.0:
            raise ValueError("marker identity memory must be positive")
        if not 0.0 < spacing_alpha <= 1.0:
            raise ValueError("marker spacing alpha must be in (0, 1]")

        self.min_spacing_m = float(min_spacing_m)
        self.max_spacing_m = float(max_spacing_m)
        self.max_spacing_error_m = float(max_spacing_error_m)
        self.max_heading_step_rad = float(max_heading_step_rad)
        self.max_speed_m_s = float(max_speed_m_s)
        self.jump_slack_m = float(jump_slack_m)
        self.identity_memory_s = float(identity_memory_s)
        self.spacing_alpha = float(spacing_alpha)
        self._last = None
        self._spacing_estimate_m = None

    @property
    def spacing_estimate_m(self):
        return self._spacing_estimate_m

    def reset(self, *, keep_spacing=False):
        """Clear pose identity; optionally retain the learned rigid spacing."""

        self._last = None
        if not keep_spacing:
            self._spacing_estimate_m = None

    @staticmethod
    def _wrapped_heading_step(current, previous):
        return abs((current - previous + math.pi) % (2.0 * math.pi)
                   - math.pi)

    def evaluate(self, *, t, x_m, y_m, heading_rad, spacing_m):
        """Evaluate and, only on acceptance, commit one raw observation."""

        values = np.asarray(
            [t, x_m, y_m, heading_rad, spacing_m], dtype=float)
        if not np.isfinite(values).all():
            return MarkerQualityDecision(
                False, "nonfinite_geometry", True)

        t, x_m, y_m, heading_rad, spacing_m = (
            float(value) for value in values)
        spacing_error = (
            np.nan if self._spacing_estimate_m is None
            else abs(spacing_m - self._spacing_estimate_m))

        position_jump = np.nan
        heading_jump = np.nan
        continuity_expired = False
        if self._last is not None:
            previous_t, previous_x, previous_y, previous_heading = self._last
            dt = t - previous_t
            if dt <= 0.0:
                return MarkerQualityDecision(
                    False, "nonmonotonic_time", True,
                    spacing_error_m=spacing_error)
            if dt > self.identity_memory_s:
                self._last = None
                continuity_expired = True
            else:
                position_jump = math.hypot(
                    x_m - previous_x, y_m - previous_y)
                heading_jump = self._wrapped_heading_step(
                    heading_rad, previous_heading)

        if spacing_m < self.min_spacing_m:
            return MarkerQualityDecision(
                False, "spacing_below_min", True,
                position_jump, heading_jump, spacing_error)
        if spacing_m > self.max_spacing_m:
            return MarkerQualityDecision(
                False, "spacing_above_max", True,
                position_jump, heading_jump, spacing_error)
        if (np.isfinite(spacing_error)
                and spacing_error > self.max_spacing_error_m):
            return MarkerQualityDecision(
                False, "spacing_discontinuity", True,
                position_jump, heading_jump, spacing_error)

        if self._last is not None:
            dt = t - self._last[0]
            maximum_jump = self.jump_slack_m + self.max_speed_m_s * dt
            if position_jump > maximum_jump:
                return MarkerQualityDecision(
                    False, "position_discontinuity", True,
                    position_jump, heading_jump, spacing_error)
            if heading_jump > self.max_heading_step_rad:
                return MarkerQualityDecision(
                    False, "heading_discontinuity", True,
                    position_jump, heading_jump, spacing_error)

        self._last = (t, x_m, y_m, heading_rad)
        if self._spacing_estimate_m is None:
            self._spacing_estimate_m = spacing_m
        else:
            self._spacing_estimate_m += self.spacing_alpha * (
                spacing_m - self._spacing_estimate_m)
        return MarkerQualityDecision(
            True, "accepted", continuity_expired,
            position_jump, heading_jump, spacing_error)


class MotionEstimator:
    """Gap-safe short-window velocity and yaw-rate estimator."""

    def __init__(self, window_s, max_gap_s, cutoff_hz):
        self.window_s = window_s
        self.max_gap_s = max_gap_s
        self.cutoff_hz = cutoff_hz
        self.samples = deque()
        self.rate_filtered = None
        self.rate_t = None

    def reset(self):
        self.samples.clear()
        self.rate_filtered = None
        self.rate_t = None

    def update(self, t, x, y, heading):
        reset = False
        if self.samples and t - self.samples[-1][0] > self.max_gap_s:
            self.reset()
            reset = True

        if self.samples:
            previous_heading = self.samples[-1][3]
            while heading - previous_heading > math.pi:
                heading -= 2.0 * math.pi
            while heading - previous_heading < -math.pi:
                heading += 2.0 * math.pi

        # Keep an unwrapped angle only in the private regression history so
        # crossing 359 -> 0 deg does not create a false yaw-rate spike.  The
        # value returned to the controller/display is canonical: carrying an
        # accumulated 362 deg outside this estimator is an unnecessary and
        # error-prone representation of the same physical heading as 2 deg.
        self.samples.append((t, x, y, heading))
        reported_heading = canonical_heading_rad(heading)
        while self.samples and t - self.samples[0][0] > self.window_s:
            self.samples.popleft()

        if (len(self.samples) < 3
                or self.samples[-1][0] - self.samples[0][0]
                < min(0.08, self.window_s / 2.0)):
            return np.full(3, np.nan), reported_heading, reset

        values = np.asarray(self.samples, dtype=float)
        time_centered = values[:, 0] - values[:, 0].mean()
        denominator = float(time_centered @ time_centered)
        if denominator <= 1e-12:
            return np.full(3, np.nan), reported_heading, reset
        rates = time_centered @ values[:, 1:4] / denominator

        if self.rate_filtered is None:
            self.rate_filtered = rates
        else:
            dt = max(t - self.rate_t, 0.0)
            alpha = 1.0 - math.exp(
                -2.0 * math.pi * self.cutoff_hz * dt)
            self.rate_filtered += alpha * (rates - self.rate_filtered)
        self.rate_t = t
        return self.rate_filtered.copy(), reported_heading, reset


def canonical_heading_rad(value):
    """Represent a finite physical heading in [0, 2*pi)."""

    value = float(value)
    return value % (2.0 * math.pi) if math.isfinite(value) else value


def canonical_heading_deg(value):
    """Represent a finite physical heading in [0, 360)."""

    value = float(value)
    return value % 360.0 if math.isfinite(value) else value


def make_velocity_sample(
    *, frame, host_monotonic_ns, t_s, tracking_valid,
    x_m=np.nan, y_m=np.nan, heading_rad=np.nan,
    vx_m_s=np.nan, vy_m_s=np.nan, u_m_s=np.nan,
    v_port_m_s=np.nan, r_rad_s=np.nan,
    quality_accepted=None, quality_reject_reason="",
    motion_reset=False, color_spacing_m=np.nan,
):
    """Build one IPC-safe sample using the BlueROV starboard-positive v sign."""

    finite_velocity = bool(np.isfinite(
        [u_m_s, v_port_m_s, r_rad_s]).all())
    if quality_accepted is None:
        quality_accepted = bool(tracking_valid)
    tracking_valid = bool(tracking_valid and quality_accepted)
    return {
        "frame": int(frame),
        "host_monotonic_ns": int(host_monotonic_ns),
        "t_s": float(t_s),
        "tracking_valid": tracking_valid,
        "velocity_valid": bool(tracking_valid and finite_velocity),
        "x_m": float(x_m),
        "y_m": float(y_m),
        "heading_rad": canonical_heading_rad(heading_rad),
        "vx_m_s": float(vx_m_s),
        "vy_m_s": float(vy_m_s),
        "u_m_s": float(u_m_s),
        "v_port_m_s": float(v_port_m_s),
        "v_starboard_m_s": float(-v_port_m_s),
        "r_rad_s": float(r_rad_s),
        "quality_accepted": bool(quality_accepted),
        "quality_reject_reason": str(quality_reject_reason),
        "motion_reset": bool(motion_reset),
        "color_spacing_m": float(color_spacing_m),
    }


def is_quit_key(key):
    """Only Esc closes the top-view window; q remains a vehicle-control key."""

    return (int(key) & 0xFF) == 27


def main(argv=None, *, sample_callback=None, stop_event=None,
         mpc_engaged=None, waypoint_align=None, status_text=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=DEFAULT_DEVICE,
                    help="V4L2 index or path (default: bench camera by-id)")
    ap.add_argument("--video", default="",
                    help="process a recorded video file instead of the live camera")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--fps", type=float, default=90.0,
                    help="requested camera capture rate (default: 90; the "
                         "driver default of 30 measures ~15 fps in practice)")
    ap.add_argument("--record-fps", type=float, default=30.0,
                    help="MP4 cadence for live capture, independent of --fps "
                         "(default: 30). Offline replay always keeps one "
                         "output frame per source frame.")
    ap.add_argument("--display-fps", type=float, default=15.0,
                    help="max preview redraws per second (default: 15); the "
                         "preview costs ~6 ms a frame at 1920x1200")
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument(
        "--ref-layout", default=DEFAULT_REF_LAYOUT,
        help="reference centres as id:x,y;... in metres (default: the "
             "four-tag pool rectangle from pool_layout_4tag.py, origin at "
             "the near-left tag)")
    ap.add_argument(
        "--ref-tag-size", default=DEFAULT_REF_TAG_SIZE_M,
        help="AprilTag black-square edge in metres: one number for every "
             "reference, or a per-tag map like '100:0.20;500:0.40' when the "
             "references are mixed sizes (default: from pool_layout_4tag.py)")
    ap.add_argument(
        "--pool-width", type=float, default=DEFAULT_POOL_WIDTH_M,
        help="distance from an equal-y two-tag baseline into the pool along "
             "+y, metres; unused for diagonal or 3+ tag layouts "
             "(default: 1.975)")
    ap.add_argument(
        "--ref-init-frames", type=int, default=N_REF_FRAMES,
        help="accepted transform samples before freezing (default: 30)")
    ap.add_argument(
        "--baseline-scale-source", choices=("configured", "camera"),
        default="configured",
        help="metric scale for a two-tag map: configured reference spacing, "
             "or the independently recovered camera-PnP spacing; camera "
             "requires valid PnP poses for both reference tags "
             "(default: configured)")
    ap.add_argument(
        "--max-ref-range-error", type=float, default=0.60,
        help="maximum three-tag range disagreement or two-tag independent-PnP "
             "baseline error when configured scale is selected, metres "
             "(default: 0.60)")
    ap.add_argument(
        "--max-ref-world-error", type=float, default=0.05,
        help="maximum reference-centre reprojection or two-tag joint corner-fit "
             "RMS, metres (default: 0.05)")
    ap.add_argument(
        "--ref-roi", action=argparse.BooleanOptionalAction, default=True,
        help="search the reference tags inside padded boxes around their last "
             "sighting instead of sweeping the whole frame; any frame that "
             "does not find every cached tag falls back to a full sweep "
             "(default: enabled)")
    ap.add_argument(
        "--ref-roi-pad", type=float, default=0.4,
        help="ROI padding as a fraction of the tag's detected size; the box "
             "tolerates a camera shift of that fraction before a full sweep "
             "is needed, and larger boxes give the speed back (default: 0.4)")
    ap.add_argument(
        "--ref-roi-refresh-frames", type=int, default=150,
        help="force a full-frame sweep at least this often so a missing "
             "reference can rejoin (default: 150)")
    ap.add_argument(
        "--ref-drift-m", type=float, default=0.12,
        help="frozen-map RMS error that indicates camera movement (default: 0.12)")
    ap.add_argument(
        "--ref-drift-frames", type=int, default=5,
        help="consecutive drift frames before automatic remapping (default: 5)")
    ap.add_argument(
        "--ref-drift-reacquire", action="store_true",
        help=("clear the frozen map and re-accumulate when drift persists "
              "(the pre-2026-08-05 behavior).  Default is to LATCH: the "
              "first frozen map is kept for the entire run and sustained "
              "drift is reported instead of remapped.  Run 20260805_190808 "
              "lost metric tracking twice for 1.4 s mid-flight to "
              "reacquisitions triggered at 0.122-0.130 m RMS with all four "
              "tags visible; the velocity MPC flies on this feedback, and a "
              "mid-run remap also moves the world frame under the "
              "controller"))
    ap.add_argument(
        "--waypoints", default="",
        help=("overlay a waypoint pattern once the map is frozen: "
              "'x,y;x,y;...' in pool metres.  Drawn through the same "
              "homography the tracker uses for positions, so it moves with "
              "the calibration instead of being an independent redrawing"))
    ap.add_argument(
        "--waypoint-arrival-radius", type=float, default=0.0,
        help=("draw the arrival ring around each waypoint, metres "
              "(0 disables); projected, so it is an ellipse in the "
              "perspective view"))
    ap.add_argument(
        "--waypoint-heading", type=float, default=None,
        help=("draw the held crab heading as ONE arrow at the pattern "
              "centroid, degrees CCW from +x.  Ignored when "
              "--waypoint-headings is given"))
    ap.add_argument(
        "--waypoint-headings", default="",
        help=("draw the per-waypoint hold setpoint as an arrow at each "
              "waypoint: 'deg;deg;...' CCW from +x, one per --waypoints "
              "entry.  This is the bow direction the heading hold drives "
              "to on arrival, taken from the same pattern module the "
              "guidance reads"))
    ap.add_argument(
        "--safe-region-margin", type=float, default=0.0,
        help=("inset the reference rectangle by this many metres and draw "
              "it as the safe operating region (0 disables).  Distinct from "
              "--pool-margin, which is a tracking gate allowing positions "
              "OUTSIDE the bounds rather than a drawn inset"))
    # Backward-compatible four-corner rectangle input.  If --ref-ids is
    # supplied it overrides --ref-layout.
    ap.add_argument("--ref-ids", default="",
                    help="legacy four-tag order BL,BR,TR,TL")
    ap.add_argument("--ref-w", type=float, default=None,
                    help="legacy rectangle width, centre-to-centre metres")
    ap.add_argument("--ref-h", type=float, default=None,
                    help="legacy rectangle height, centre-to-centre metres")
    ap.add_argument(
        "--marker-pair", choices=tuple(MARKER_PAIR_CONFIGS),
        default=DEFAULT_MARKER_PAIR,
        help="physical left/right color markers: red-lime, red-green or "
             "red-white; the heading axis always runs red->right "
             "(default: red-lime; use red-green to replay pre-2026-08-17 "
             "runs)")
    ap.add_argument("--heading-offset", type=float, default=90.0,
                    help="degrees added to the red->right marker heading so it "
                         "equals the bow direction (default: 90 for the "
                         "current transverse marker layout)")
    ap.add_argument("--cam-height", type=float, default=0.0,
                    help="camera height above water, m (refraction correction)")
    ap.add_argument("--tag-height", type=float, default=0.0,
                    help="reference tag plane height above water, m")
    ap.add_argument("--rov-depth", type=float, default=0.0,
                    help="constant ROV depth below surface, m; if depth varies, "
                         "leave at 0 and correct in post from depth telemetry")
    ap.add_argument(
        "--velocity-cutoff", "--ema-cutoff", dest="velocity_cutoff",
        type=float, default=1.5,
        help="time-aware velocity/yaw-rate low-pass cutoff, Hz (default: 1.5)")
    ap.add_argument(
        "--velocity-window", "--ma-window", dest="velocity_window",
        type=float, default=0.35,
        help="causal regression window for velocity/yaw rate, s (default: 0.35)")
    ap.add_argument(
        "--max-track-gap", type=float, default=0.25,
        help="accepted-detection gap that resets motion estimation, s "
             "(default: 0.25)")
    ap.add_argument(
        "--max-color-spacing", type=float, default=0.80,
        help="maximum red/right separation on the tag plane, m (default: 0.80)")
    ap.add_argument(
        "--max-speed", type=float, default=2.0,
        help="maximum midpoint speed used by association, m/s (default: 2.0)")
    ap.add_argument(
        "--jump-slack", type=float, default=0.15,
        help="position-association allowance beyond speed*dt, m (default: 0.15)")
    ap.add_argument(
        "--quality-min-color-spacing", type=float,
        default=QUALITY_MIN_COLOR_SPACING_M,
        help="minimum physical red/right marker separation accepted before "
             "velocity estimation, m (default: 0.10)")
    ap.add_argument(
        "--quality-max-color-spacing", type=float,
        default=QUALITY_MAX_COLOR_SPACING_M,
        help="maximum physical red/right marker separation accepted before "
             "velocity estimation, m (default: 0.35)")
    ap.add_argument(
        "--quality-max-spacing-error", type=float,
        default=QUALITY_MAX_SPACING_ERROR_M,
        help="maximum separation change from the learned rigid-bar spacing, "
             "m (default: 0.08)")
    ap.add_argument(
        "--quality-max-heading-step", type=float,
        default=QUALITY_MAX_HEADING_STEP_DEG,
        help="maximum wrapped heading change from the last-good identity, "
             "degrees (default: 45)")
    ap.add_argument(
        "--quality-max-speed", type=float,
        default=QUALITY_MAX_SPEED_M_S,
        help="maximum last-good midpoint speed for the pre-derivative "
             "identity gate, m/s (default: 1.0)")
    ap.add_argument(
        "--quality-jump-slack", type=float,
        default=QUALITY_JUMP_SLACK_M,
        help="position allowance beyond quality-max-speed*dt, m "
             "(default: 0.05)")
    ap.add_argument(
        "--quality-identity-memory", type=float,
        default=QUALITY_IDENTITY_MEMORY_S,
        help="retain the last-good marker identity across short detection "
             "losses for this many seconds (default: 1.0)")
    ap.add_argument(
        "--pool-margin", type=float, default=0.30,
        help="allowed margin outside reference-coordinate pool bounds, m "
             "(default: 0.30)")
    ap.add_argument(
        "--distortion-mode",
        choices=("auto", "raw", "calibrated"),
        default="auto",
        help="geometry pixel model: auto uses raw pixels for a two-tag layout "
             "and saved distortion for 3+ tags; raw bypasses saved distortion; "
             "calibrated always applies it (default: auto)")
    ap.add_argument("--dynamic-homography", action="store_true",
                    help="re-estimate the homography every frame instead of "
                         "freezing the robust initial estimate")
    ap.add_argument("--show-mask", action="store_true",
                    help="show the HSV masks for threshold tuning")
    ap.add_argument("--no-display", action="store_true",
                    help="disable GUI windows (useful for offline replay)")
    ap.add_argument("--out", default="",
                    help="output CSV (default usb_cam/rov_track_<timestamp>.csv)")
    ap.add_argument("--out-video", default="", help="write annotated video")
    ap.add_argument(
        "--pool-map-out", default="",
        help="write the frozen pool map here as JSON (default: "
             "<out CSV>_pool_map.json). It also mirrors to --pool-map-latest "
             "so the next run can fall back to it")
    ap.add_argument(
        "--pool-map-latest",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "pool_map_latest.json"),
        help="stable path the newest frozen map is mirrored to; pass '' to "
             "disable the mirror")
    ap.add_argument(
        "--pool-map", default="",
        help="start from a SAVED pool map instead of acquiring one. For a "
             "run where the tags cannot be detected (glare, low sun, a "
             "blocked tag). Valid ONLY if the camera has not moved since the "
             "map was saved: pass '' (the default) to always re-acquire")
    ap.add_argument(
        "--pool-map-verify-m", type=float, default=0.10,
        help="with --pool-map, the world RMS above which visible reference "
             "tags are declared inconsistent with the loaded map, meaning "
             "the camera moved (default: 0.10 m)")
    ap.add_argument("--window-x", type=int, default=None,
                    help="optional display-window x position")
    ap.add_argument("--window-y", type=int, default=None,
                    help="optional display-window y position")
    args = ap.parse_args(argv)
    marker_config = marker_pair_config(args.marker_pair)

    if args.ref_init_frames < 1:
        ap.error("--ref-init-frames must be at least 1")
    if args.ref_roi_pad <= 0:
        ap.error("--ref-roi-pad must be positive")
    if args.ref_roi_refresh_frames < 1:
        ap.error("--ref-roi-refresh-frames must be at least 1")
    if args.pool_width <= 0:
        ap.error("--pool-width must be positive")
    try:
        quality_gate = MarkerObservationGate(
            min_spacing_m=args.quality_min_color_spacing,
            max_spacing_m=args.quality_max_color_spacing,
            max_spacing_error_m=args.quality_max_spacing_error,
            max_heading_step_rad=math.radians(
                args.quality_max_heading_step),
            max_speed_m_s=args.quality_max_speed,
            jump_slack_m=args.quality_jump_slack,
            identity_memory_s=args.quality_identity_memory,
        )
    except ValueError as exc:
        ap.error(str(exc))
    if args.ref_ids:
        ref_ids = [int(value) for value in args.ref_ids.split(",")]
        if len(ref_ids) != 4 or args.ref_w is None or args.ref_h is None:
            ap.error("legacy --ref-ids requires four IDs plus --ref-w and --ref-h")
        ref_world = {
            ref_ids[0]: (0.0, 0.0), ref_ids[1]: (args.ref_w, 0.0),
            ref_ids[2]: (args.ref_w, args.ref_h),
            ref_ids[3]: (0.0, args.ref_h),
        }
    else:
        try:
            ref_world = parse_ref_layout(args.ref_layout)
        except ValueError as exc:
            ap.error(str(exc))
        ref_ids = list(ref_world)
    # Resolved only now: a per-tag size map has to be checked against the
    # layout that is actually in force, including the legacy --ref-ids one.
    try:
        ref_tag_sizes = parse_ref_tag_sizes(args.ref_tag_size, ref_world)
    except ValueError as exc:
        ap.error(str(exc))
    ref_xy = np.asarray(list(ref_world.values()), dtype=float)
    if len(ref_world) == 2 and abs(ref_xy[1, 1] - ref_xy[0, 1]) <= 1e-9:
        # Edge pair along world x: the pool extends +y by --pool-width.
        xmin = float(ref_xy[:, 0].min() - args.pool_margin)
        xmax = float(ref_xy[:, 0].max() + args.pool_margin)
        baseline_y = float(ref_xy[0, 1])
        ymin = baseline_y - args.pool_margin
        ymax = baseline_y + args.pool_width + args.pool_margin
    else:
        # 3+ tags, or a diagonal two-tag pair whose centres already span
        # the tracking rectangle: bounds come from the tag span itself.
        xmin, ymin = ref_xy.min(axis=0) - args.pool_margin
        xmax, ymax = ref_xy.max(axis=0) + args.pool_margin
    association_bounds = (xmin, xmax, ymin, ymax)

    scale = 1.0
    if args.cam_height > 0 and args.rov_depth > 0:
        scale = ((args.cam_height + args.rov_depth / 1.33)
                 / (args.cam_height - args.tag_height))
        print(f"[refraction] applying constant scale {scale:.4f} "
              f"(H={args.cam_height} m, h_tag={args.tag_height} m, "
              f"d={args.rov_depth} m)")
    else:
        print("[refraction] scale=1.0 — coordinates are TAG-PLANE metres; "
              "correct with depth telemetry in post (see docstring)")

    if not args.out:
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f"rov_track_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    if not args.pool_map_out:
        # Next to the run's own CSV, so the map travels with the recording it
        # describes rather than only existing at a shared path.
        args.pool_map_out = os.path.splitext(args.out)[0] + "_pool_map.json"
    pool_map_targets = [p for p in (args.pool_map_out, args.pool_map_latest)
                        if p]
    for output_path in (args.out, args.out_video, *pool_map_targets):
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)),
                        exist_ok=True)

    d = np.load(args.calib)
    K0, dist = d["camera_matrix"].astype(np.float64), d["dist_coeffs"].astype(np.float64)
    cw, ch = int(d["image_w"]), int(d["image_h"])
    use_raw_geometry = (
        args.distortion_mode == "raw"
        or (args.distortion_mode == "auto" and len(ref_world) == 2)
    )
    geometry_dist = np.zeros_like(dist) if use_raw_geometry else dist.copy()
    geometry_distortion_mode = (
        "raw-zero-distortion" if use_raw_geometry else "calibrated")
    if use_raw_geometry:
        print(
            "[geometry] raw-pixel mapping/PnP enabled: saved distortion is "
            "bypassed for geometry but retained in the debug CSV")
    else:
        print("[geometry] applying saved camera distortion coefficients"
              + (" (auto: 3+ reference tags, so the two-tag raw-pixel "
                 "fallback no longer applies)"
                 if args.distortion_mode == "auto" and len(ref_world) >= 3
                 else ""))

    if args.video:
        cap = cv2.VideoCapture(args.video)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        cap = open_camera(args.device, args.width, args.height, args.fps)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not cap.isOpened():
        raise SystemExit(f"cannot open {'video ' + args.video if args.video else args.device}")

    # Capture runs as fast as the sensor allows; the MP4 keeps its own, slower
    # cadence.  Tying the two together would pad the file with duplicates of
    # the same annotated frame -- tracking cannot produce 90 distinct overlays
    # per second -- for 3x the size and encode cost and no extra information.
    record_fps = float(args.record_fps) if args.record_fps else src_fps
    grabber = None
    if not args.video:
        grabber = FrameGrabber(cap)
        grabber.start()
        if not grabber.wait_first():
            grabber.close()
            cap.release()
            raise SystemExit(
                f"no frames from {args.device} within 5 s"
                + (f" ({grabber.error})" if grabber.error else ""))
        print(f"[run] capture {src_fps:.0f} fps requested, "
              f"MP4 cadence {record_fps:.0f} fps")

    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT)
    det_params = cv2.aruco.DetectorParameters()
    det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, det_params)
    roi_detector = (
        ReferenceRoiDetector(
            detector,
            pad_ratio=args.ref_roi_pad,
            refresh_frames=args.ref_roi_refresh_frames,
        )
        if args.ref_roi else None
    )
    tag_obj = make_tag_object_map(ref_tag_sizes)

    # Overlay geometry, in pool metres.  The rectangle is derived from the
    # reference tags themselves rather than configured separately, so the
    # drawn frame is by construction the frame positions are reported in.
    reference_xy = np.asarray(list(ref_world.values()), dtype=float)
    pool_rect_wh = (
        (float(reference_xy[:, 0].max()), float(reference_xy[:, 1].max()))
        if len(reference_xy) else None)
    overlay_waypoints = []
    if args.waypoints:
        for chunk in str(args.waypoints).split(";"):
            item = chunk.strip()
            if not item:
                continue
            parts = item.split(",")
            if len(parts) != 2:
                raise SystemExit(
                    f"--waypoints: {item!r} is not 'x,y' in metres")
            overlay_waypoints.append((float(parts[0]), float(parts[1])))
    overlay_headings = None
    if args.waypoint_headings:
        overlay_headings = [
            float(chunk) for chunk in
            str(args.waypoint_headings).replace(",", ";").split(";")
            if chunk.strip()]
        if len(overlay_headings) != len(overlay_waypoints):
            raise SystemExit(
                f"--waypoint-headings has {len(overlay_headings)} entries "
                f"but --waypoints has {len(overlay_waypoints)}; they must "
                "match one-to-one")
    # Waypoints whose arrival ring the vehicle has entered, filled in the
    # overlay so the operator can see progress at a glance.  Decided here
    # from the tracker's own accepted fix rather than from the controller's
    # target index: the controller is a separate process and the only IPC
    # runs the other way (velocity samples out).  The two can disagree in
    # one case worth knowing about -- passing through a ring out of order
    # marks it visited here while the guidance has not targeted it yet.
    visited_waypoints: set[int] = set()

    K = None                    # rescaled on first frame
    H_mat = None                # ready/frozen homography
    pool_map_loaded = None      # payload when started from a saved map
    pool_map_verified = False   # a visible tag has confirmed it since load
    H_samples = []
    H_scale_samples = []
    H_quality = {"method": "none", "world_rms_m": np.nan,
                 "range_rms_m": np.nan,
                 "map_scale_source": "configured",
                 "map_baseline_m": np.nan,
                 "map_scale_factor": 1.0}
    motion = MotionEstimator(
        args.velocity_window, args.max_track_gap, args.velocity_cutoff)
    previous_mid = None
    previous_pair_t = None
    association_spacing_estimate = None
    preview_mid = None
    drift_count = 0
    drift_warned = False
    rows = []
    trail = []
    writer = None
    n_hit = 0
    frame_idx = 0
    fps_est, t_fps, n_fps = 0.0, time.time(), 0
    t0 = time.monotonic()
    display_created = False
    display_period = 1.0 / max(float(args.display_fps), 0.1)
    next_display = 0.0

    layout_text = ", ".join(
        f"{tag}=({xy[0]:.3f},{xy[1]:.3f})"
        for tag, xy in ref_world.items())
    print(f"[ref] world centres [m]: {layout_text}")
    print("[ref] tag black-square edge [m]: "
          + ", ".join(f"{tag}={size:.3f}"
                      for tag, size in sorted(ref_tag_sizes.items()))
          + f"; freezing after {args.ref_init_frames} accepted samples")
    print(f"[ref] two-tag metric scale source: "
          f"{args.baseline_scale_source}")
    print(f"[marker] {args.marker_pair}: left=red, "
          f"right={marker_config.right_name}, heading axis "
          f"red->{marker_config.right_name}")
    print("[run] Esc quits. Waiting for frames ...")

    last_sequence = 0
    try:
        while stop_event is None or not stop_event.is_set():
            if grabber is None:
                ok, frame = cap.read()
                frame_host_ns = time.monotonic_ns()
                if not ok:
                    print("[run] stream ended / frame grab failed")
                    break
            else:
                # Latest-frame slot: block only until a frame newer than the one
                # just processed arrives, so the loop never re-analyses a frame
                # and never waits on the camera it already outran.
                frame, frame_host_ns, sequence = grabber.snapshot()
                if sequence == last_sequence:
                    if grabber.stopped:
                        print("[run] stream ended / frame grab failed"
                              + (f" ({grabber.error})" if grabber.error else ""))
                        break
                    time.sleep(0.001)
                    continue
                last_sequence = sequence
            # mpc_engaged is a shared Value owned by the control process; it
            # colors the overlay (red trail + MPC MODE banner) while engaged.
            engaged = bool(mpc_engaged.value) if mpc_engaged is not None else False
            # waypoint_align is the matching shared Array: the guidance's
            # per-waypoint settle states (0 pending, 1 aligned, 2 timeout),
            # rendered as the arrow color.  Slicing takes the array lock.
            align_states = (
                waypoint_align[:len(overlay_waypoints)]
                if waypoint_align is not None and overlay_waypoints else ())
            t = (frame_idx / src_fps) if args.video else (time.monotonic() - t0)

            if K is None:
                fh, fw = frame.shape[:2]
                K = K0.copy()
                if (cw, ch) != (fw, fh):
                    K[0] *= fw / cw
                    K[1] *= fh / ch
                print(f"[run] {fw}x{fh}, calib {'rescaled' if (cw, ch) != (fw, fh) else 'native'}")

                if args.pool_map:
                    # Deliberately fatal rather than falling back to normal
                    # acquisition: the operator asked for this map because
                    # the tags cannot be seen, so a silent fallback would
                    # just hang on "CALIBRATING" with no explanation.
                    H_mat, pool_map_loaded, map_warnings = load_pool_map(
                        args.pool_map, K, geometry_dist, ref_world,
                        ref_tag_sizes, frame.shape, geometry_distortion_mode)
                    H_quality = {
                        "method": f"loaded:{pool_map_loaded['method']}",
                        "world_rms_m": float(pool_map_loaded["world_rms_m"]),
                        "range_rms_m": np.nan,
                        "map_scale_source": pool_map_loaded[
                            "map_scale_source"],
                        "map_baseline_m": np.nan,
                        "map_scale_factor": float(
                            pool_map_loaded["map_scale_factor"]),
                    }
                    print(f"[ref] pool map LOADED from {args.pool_map} "
                          f"(saved {pool_map_loaded['created_utc']}, "
                          f"{pool_map_loaded['method']})")
                    for warning in map_warnings:
                        print(f"[ref] WARNING: {warning}")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- reference tags -> homography -------------------------------
            if roi_detector is not None:
                corners, ids = roi_detector.detect(gray, ref_world)
            else:
                corners, ids, _ = detector.detectMarkers(gray)
            ref_seen = detect_references(
                corners, ids, ref_world, tag_obj, K, geometry_dist)

            candidate_H, candidate_quality = estimate_reference_homography(
                ref_seen, ref_world, K, geometry_dist, tag_obj,
                baseline_scale_source=args.baseline_scale_source)
            candidate_reject_reasons = []
            candidate_ok = candidate_H is not None
            if not candidate_ok:
                candidate_reject_reasons.append(candidate_quality["method"])
            else:
                if (len(ref_world) == 2
                        and not candidate_quality.get("pnp_valid", False)):
                    candidate_reject_reasons.append("reference_pnp_invalid")
                world_rms = candidate_quality["world_rms_m"]
                if not np.isfinite(world_rms):
                    candidate_reject_reasons.append("world_rms_nonfinite")
                elif world_rms > args.max_ref_world_error:
                    candidate_reject_reasons.append(
                        f"world_rms>{args.max_ref_world_error:.3f}")
                range_rms = candidate_quality["range_rms_m"]
                if (args.baseline_scale_source == "configured"
                        and np.isfinite(range_rms)
                        and range_rms > args.max_ref_range_error):
                    candidate_reject_reasons.append(
                        f"pnp_baseline_error>{args.max_ref_range_error:.3f}")
                candidate_ok = not candidate_reject_reasons
            candidate_reject_reason = ";".join(candidate_reject_reasons)

            if candidate_ok:
                if args.dynamic_homography:
                    H_mat = candidate_H
                    H_quality = candidate_quality
                elif H_mat is None:
                    H_samples.append(candidate_H)
                    H_scale_samples.append(float(
                        candidate_quality.get("map_scale_factor", 1.0)))
                    if len(H_samples) >= args.ref_init_frames:
                        H_mat = average_homographies(H_samples)
                        H_quality = dict(candidate_quality)
                        H_quality["map_scale_factor"] = float(
                            np.median(H_scale_samples))
                        if len(ref_world) == 2:
                            configured_baseline = float(
                                np.linalg.norm(ref_xy[1] - ref_xy[0]))
                            H_quality["map_baseline_m"] = (
                                configured_baseline
                                * H_quality["map_scale_factor"])
                        geometry_error = candidate_quality["range_rms_m"]
                        if candidate_quality["method"].startswith("2-tag"):
                            geometry_label = "PnP baseline error"
                        else:
                            geometry_label = "range RMS"
                        print(
                            f"[ref] homography frozen from {len(H_samples)} "
                            f"samples ({candidate_quality['method']}; "
                            f"{geometry_label}={geometry_error:.3f} m; "
                            f"scale={H_quality['map_scale_source']} "
                            f"x{H_quality['map_scale_factor']:.4f})")
                        motion.reset()
                        quality_gate.reset()
                        previous_mid = previous_pair_t = None
                        association_spacing_estimate = None
                        preview_mid = None
                        # Persist immediately, not at shutdown: a run that
                        # crashes or is killed still leaves the next one a
                        # usable fallback.
                        for map_path in pool_map_targets:
                            try:
                                save_pool_map(
                                    map_path, H_mat, K, geometry_dist,
                                    ref_world, ref_tag_sizes, frame.shape,
                                    geometry_distortion_mode, H_quality,
                                    len(H_samples))
                                print(f"[ref] pool map saved to {map_path}")
                            except OSError as exc:
                                print(f"[ref] could not save pool map to "
                                      f"{map_path}: {exc}")

            ref_world_rms = np.nan
            if H_mat is not None and ref_seen:
                seen_ids = list(ref_seen)
                seen_und = undistort_pts(
                    [ref_seen[tag]["center_px"] for tag in seen_ids],
                    K, geometry_dist)
                seen_world = project(H_mat, seen_und)
                active_scale_factor = float(
                    H_quality.get("map_scale_factor", 1.0))
                expected_world = np.asarray(
                    [ref_world[tag] for tag in seen_ids], dtype=float
                ) * active_scale_factor
                ref_world_rms = float(np.sqrt(np.mean(np.sum(
                    (seen_world - expected_world) ** 2, axis=1))))

                # A loaded map is an assumption ("the camera has not moved")
                # until a tag confirms it.  Any tag that IS visible settles
                # that question on the first frame it appears, so report the
                # verdict once rather than letting the run proceed on trust.
                if pool_map_loaded is not None and not pool_map_verified:
                    pool_map_verified = True
                    if ref_world_rms <= args.pool_map_verify_m:
                        print(f"[ref] loaded map VERIFIED against "
                              f"{len(seen_ids)} visible tag(s): world RMS "
                              f"{ref_world_rms:.3f} m "
                              f"(gate {args.pool_map_verify_m:.3f})")
                    else:
                        print(f"[ref] loaded map REJECTED: world RMS "
                              f"{ref_world_rms:.3f} m > "
                              f"{args.pool_map_verify_m:.3f} m against "
                              f"{len(seen_ids)} visible tag(s) — the camera "
                              "moved since the map was saved. Clearing it "
                              "and re-acquiring.")
                        H_mat = None
                        H_samples.clear()
                        H_scale_samples.clear()
                        pool_map_loaded = None

            # Sustained disagreement between the frozen map and >=2 visible
            # references.  Since 2026-08-05 the map is LATCHED by default:
            # once frozen it serves the whole run, and drift is reported
            # (console + overlay + the per-row ref_world_rms_m column)
            # rather than remapped.  A mid-run remap costs a ~1.4 s metric
            # blackout while the MPC flies on this feedback and then moves
            # the world frame under the controller; run 20260805_190808 hit
            # both, twice, from RMS hovering just past the gate with all
            # four tags visible.  --ref-drift-reacquire restores clearing
            # for bench use where the camera really may be bumped.
            if (H_mat is not None and not args.dynamic_homography
                    and len(ref_seen) >= 2
                    and ref_world_rms > args.ref_drift_m):
                drift_count += 1
            else:
                drift_count = 0
                drift_warned = False
            if drift_count >= args.ref_drift_frames:
                if args.ref_drift_reacquire:
                    print(
                        f"[ref] camera/reference drift {ref_world_rms:.3f} "
                        f"m for {drift_count} frames; clearing map and "
                        "reacquiring")
                    H_mat = None
                    H_samples.clear()
                    H_scale_samples.clear()
                    H_quality = {"method": "reacquiring",
                                 "world_rms_m": np.nan,
                                 "range_rms_m": np.nan,
                                 "map_scale_source": "configured",
                                 "map_baseline_m": np.nan,
                                 "map_scale_factor": 1.0}
                    motion.reset()
                    quality_gate.reset()
                    previous_mid = previous_pair_t = None
                    association_spacing_estimate = None
                    preview_mid = None
                    drift_count = 0
                elif not drift_warned:
                    print(
                        f"[ref] reference drift {ref_world_rms:.3f} m past "
                        f"{args.ref_drift_m:.3f} for {drift_count} frames; "
                        "KEEPING the latched map for this run "
                        "(--ref-drift-reacquire restores remapping)")
                    drift_warned = True

            # During startup, show the accepted per-frame candidate while the
            # robust frozen map is accumulating.  Once frozen, always display
            # reference positions through that fixed map so drift is visible.
            display_H = H_mat
            if display_H is None and candidate_ok:
                display_H = candidate_H
            ref_live_positions = reference_positions(
                ref_seen, display_H, K, geometry_dist)
            baseline_a, baseline_b, baseline_measured, baseline_expected = (
                reference_baseline(ref_live_positions, ref_world))

            # --- color tags --------------------------------------------------
            # Color segmentation and a pixel-space preview remain visible during
            # reference calibration. Metric pairing, logging, and motion
            # estimation still wait for the frozen pool map.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            right_blobs, right_mask = find_blobs(
                hsv, marker_config.right_lo, marker_config.right_hi)
            reds, rmask = find_blobs(hsv, RED_LO, RED_HI)
            preview_pair = pick_pair_pixel(right_blobs, reds, preview_mid)
            visual_pair = preview_pair if H_mat is None else None
            visual_heading_deg = np.nan
            visual_heading_suffix = " (image preview)"
            if visual_pair is not None:
                preview_right, preview_red = visual_pair
                preview_mid = (
                    (preview_right[0] + preview_red[0]) / 2.0,
                    (preview_right[1] + preview_red[1]) / 2.0,
                )
                visual_heading_deg = canonical_heading_deg(math.degrees(
                    math.atan2(
                        -(preview_right[1] - preview_red[1]),
                        preview_right[0] - preview_red[0])))

            pair = None
            if H_mat is not None:
                active_scale_factor = float(
                    H_quality.get("map_scale_factor", 1.0))
                active_association_bounds = tuple(
                    float(value) * active_scale_factor
                    for value in association_bounds)
                pair = pick_pair_metric(
                    right_blobs, reds, H_mat, K, geometry_dist,
                    previous_mid=previous_mid,
                    previous_t=previous_pair_t,
                    now=t,
                    spacing_estimate=association_spacing_estimate,
                    max_spacing_m=args.max_color_spacing,
                    max_speed_m_s=args.max_speed,
                    jump_slack_m=args.jump_slack,
                    bounds=active_association_bounds,
                )

            # t_unix aligns rows with MAVLink logs (SCALED_PRESSURE2 depth) so the
            # per-sample refraction scale can be applied in post.
            row = [frame_idx, frame_host_ns, f"{t:.4f}",
                   "" if args.video else f"{time.time():.3f}", len(ref_seen)]
            pair_spacing_plane_m = np.nan
            color_spacing_m = np.nan
            right_x_m = np.nan
            right_y_m = np.nan
            motion_reset = False
            quality_decision = MarkerQualityDecision(
                False, "no_color_pair", False)
            sample = make_velocity_sample(
                frame=frame_idx,
                host_monotonic_ns=frame_host_ns,
                t_s=t,
                tracking_valid=False,
                quality_accepted=False,
                quality_reject_reason=quality_decision.reason,
            )
            if pair is not None:
                (right, red, right_world, red_world,
                 pair_spacing_plane_m) = pair
                midpoint_plane = (right_world + red_world) / 2.0
                (right_x_m, right_y_m), (rx, ry) = (
                    np.asarray([right_world, red_world]) * scale)
                cx = (right_x_m + rx) / 2.0
                cy = (right_y_m + ry) / 2.0
                color_spacing_m = pair_spacing_plane_m * scale
                raw_heading = (
                    math.atan2(right_y_m - ry, right_x_m - rx)
                    + math.radians(args.heading_offset))
                visual_pair = (right, red)
                visual_heading_deg = canonical_heading_deg(
                    math.degrees(raw_heading))
                visual_heading_suffix = ""
                quality_decision = quality_gate.evaluate(
                    t=t,
                    x_m=cx,
                    y_m=cy,
                    heading_rad=raw_heading,
                    spacing_m=color_spacing_m,
                )
                raw_geometry = [
                    f"{right_x_m:.4f}", f"{right_y_m:.4f}",
                    f"{rx:.4f}", f"{ry:.4f}",
                    f"{cx:.4f}", f"{cy:.4f}",
                    f"{canonical_heading_deg(math.degrees(raw_heading)):.2f}",
                ]
                if quality_decision.accepted:
                    previous_mid = midpoint_plane
                    previous_pair_t = t
                    association_spacing_estimate = (
                        pair_spacing_plane_m
                        if association_spacing_estimate is None
                        else 0.95 * association_spacing_estimate
                        + 0.05 * pair_spacing_plane_m
                    )
                    if quality_decision.reset_motion:
                        motion.reset()
                    rates, heading, estimator_reset = motion.update(
                        t, cx, cy, raw_heading)
                    motion_reset = bool(
                        quality_decision.reset_motion or estimator_reset)
                    vx, vy, r_s = rates
                    cpsi, spsi = math.cos(heading), math.sin(heading)
                    u = vx * cpsi + vy * spsi       # surge (forward)
                    v_port = -vx * spsi + vy * cpsi # sway (port-positive)
                    n_hit += 1

                    # Only an ACCEPTED fix can mark a waypoint visited, so a
                    # rejected jump cannot light one up spuriously.
                    if overlay_waypoints and args.waypoint_arrival_radius > 0:
                        for w_index, (wx, wy) in enumerate(overlay_waypoints):
                            if math.hypot(cx - wx, cy - wy) <= \
                                    args.waypoint_arrival_radius:
                                visited_waypoints.add(w_index)

                    hd = canonical_heading_deg(math.degrees(heading))
                    visual_heading_deg = hd
                    rate_values = [vx, vy, u, v_port]
                    rate_text = [
                        f"{value:.4f}" if np.isfinite(value) else ""
                        for value in rate_values]
                    row += raw_geometry + rate_text + [
                            f"{-v_port:.4f}" if np.isfinite(v_port) else "",
                            (f"{math.degrees(r_s):.2f}"
                             if not math.isnan(r_s) else ""),
                            f"{r_s:.6f}" if not math.isnan(r_s) else "",
                            f"{scale:.4f}"]
                    sample = make_velocity_sample(
                        frame=frame_idx,
                        host_monotonic_ns=frame_host_ns,
                        t_s=t,
                        tracking_valid=True,
                        x_m=cx,
                        y_m=cy,
                        heading_rad=heading,
                        vx_m_s=vx,
                        vy_m_s=vy,
                        u_m_s=u,
                        v_port_m_s=v_port,
                        r_rad_s=r_s,
                        quality_accepted=True,
                        motion_reset=motion_reset,
                        color_spacing_m=color_spacing_m,
                    )

                    if not math.isnan(vx):
                        cv2.putText(
                            frame,
                            f"pos=({cx:+.2f},{cy:+.2f})m psi={hd:6.1f}deg",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)
                        cv2.putText(
                            frame,
                            f"v=({vx:+.2f},{vy:+.2f}) u={u:+.2f} "
                            f"v_sb={-v_port:+.2f} m/s "
                            f"r={math.degrees(r_s):+.1f}deg/s",
                            (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)
                else:
                    # Preserve the raw selected geometry in the CSV, but prevent
                    # it from entering either the regression window or live
                    # fusion.  The next accepted observation must warm the
                    # derivative from an empty history.
                    motion.reset()
                    motion_reset = True
                    row += raw_geometry + [""] * 7 + [f"{scale:.4f}"]
                    sample = make_velocity_sample(
                        frame=frame_idx,
                        host_monotonic_ns=frame_host_ns,
                        t_s=t,
                        tracking_valid=False,
                        x_m=cx,
                        y_m=cy,
                        heading_rad=raw_heading,
                        quality_accepted=False,
                        quality_reject_reason=quality_decision.reason,
                        motion_reset=True,
                        color_spacing_m=color_spacing_m,
                    )
            else:
                row += [""] * 15
            row += [
                int(sample["tracking_valid"]),
                int(sample["velocity_valid"]),
            ]

            # Red/right overlay: colored boxes, yellow midpoint, red->right
            # cyan arrow, local heading, and a yellow midpoint trail after
            # metric tracking becomes active.
            if visual_pair is not None:
                visual_right, visual_red = visual_pair
                right_px, right_py, _, right_bbox = visual_right
                rx_px, ry_px, _, red_bbox = visual_red
                if marker_config.right_name == "white":
                    bx, by, bw, bh = right_bbox
                    cv2.rectangle(
                        frame, (bx, by), (bx + bw, by + bh), (0, 0, 0), 6)
                for (bx, by, bw, bh), color in [
                        (right_bbox, marker_config.right_overlay_bgr),
                        (red_bbox, (0, 0, 255))]:
                    cv2.rectangle(
                        frame, (bx, by), (bx + bw, by + bh), color, 3)
                visual_mid_px = (
                    int((right_px + rx_px) / 2),
                    int((right_py + ry_px) / 2),
                )
                zero_direction = pool_x_image_direction(
                    H_mat, visual_mid_px, K, geometry_dist)
                zero_end = np.rint(
                    np.asarray(visual_mid_px, dtype=float)
                    + ZERO_REFERENCE_LENGTH_PX * zero_direction).astype(int)
                zero_end[0] = int(np.clip(zero_end[0], 0, frame.shape[1] - 1))
                zero_end[1] = int(np.clip(zero_end[1], 0, frame.shape[0] - 1))
                # A plain datum, not an arrow: the only arrow at the
                # vehicle should be the bow, so the two cannot be confused
                # at a glance.
                cv2.line(
                    frame, visual_mid_px, tuple(zero_end),
                    (245, 245, 245), 2, cv2.LINE_AA)
                perpendicular = np.array(
                    [-zero_direction[1], zero_direction[0]])
                zero_label = (
                    np.asarray(visual_mid_px, dtype=float)
                    + 75.0 * zero_direction + 28.0 * perpendicular)
                zero_label[0] = np.clip(
                    zero_label[0], 5, max(5, frame.shape[1] - 190))
                zero_label[1] = np.clip(
                    zero_label[1], 20, max(20, frame.shape[0] - 10))
                zero_frame = "+pool x" if H_mat is not None else "+image x"
                cv2.putText(
                    frame, f"0 deg ({zero_frame})",
                    tuple(np.rint(zero_label).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2,
                    cv2.LINE_AA)
                cv2.circle(frame, visual_mid_px, 6, (0, 255, 255), -1)
                # The bow arrow starts at the marker midpoint -- the point
                # the position fix is taken at -- and runs along the
                # REPORTED heading, which already carries
                # --heading-offset (90 deg for the transverse marker
                # layout).  Deriving it from the angle rather than from
                # the red->right marker vector means the drawn direction
                # and the printed number cannot disagree, and changing the
                # offset moves both together.
                if math.isfinite(visual_heading_deg):
                    bow_direction = pool_heading_image_direction(
                        H_mat, visual_mid_px, K, geometry_dist,
                        math.radians(visual_heading_deg))
                    bow_end = np.rint(
                        np.asarray(visual_mid_px, dtype=float)
                        + HEADING_ARROW_LENGTH_PX
                        * bow_direction).astype(int)
                    bow_end[0] = int(
                        np.clip(bow_end[0], 0, frame.shape[1] - 1))
                    bow_end[1] = int(
                        np.clip(bow_end[1], 0, frame.shape[0] - 1))
                    cv2.arrowedLine(
                        frame, visual_mid_px, tuple(bow_end),
                        (255, 255, 0), 3, cv2.LINE_AA, tipLength=0.22)
                cv2.putText(
                    frame,
                    f"heading {visual_heading_deg:6.1f} deg"
                    f"{visual_heading_suffix}",
                    (visual_mid_px[0] + 20, visual_mid_px[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
                    cv2.LINE_AA)
                if pair is not None and quality_decision.accepted:
                    trail.append((visual_mid_px, engaged))
                elif pair is not None:
                    cv2.putText(
                        frame,
                        f"VELOCITY REJECTED: {quality_decision.reason}",
                        (10, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                        (0, 165, 255), 2, cv2.LINE_AA)

            h_state = (
                "dynamic" if H_mat is not None and args.dynamic_homography
                else "frozen" if H_mat is not None
                else f"acquiring:{len(H_samples)}/{args.ref_init_frames}")
            # The control process receives only this compact latest-value
            # sample, not the diagnostic CSV row below.  Publish the map
            # readiness explicitly so full-pose MPC cannot engage on a
            # loaded homography before visible reference tags verify it.
            sample["homography_state"] = h_state
            sample["pool_map_verified"] = bool(
                H_mat is not None
                and (pool_map_loaded is None or pool_map_verified)
            )
            sample["ref_world_rms_m"] = (
                float(ref_world_rms)
                if np.isfinite(ref_world_rms) else float("nan")
            )
            sample["ref_drift_active"] = bool(
                H_mat is not None
                and drift_count >= args.ref_drift_frames
            )
            row += [
                f"{ref_world_rms:.4f}" if np.isfinite(ref_world_rms) else "",
                (f"{H_quality['range_rms_m']:.4f}"
                 if np.isfinite(H_quality["range_rms_m"]) else ""),
                (f"{color_spacing_m:.4f}"
                 if np.isfinite(color_spacing_m) else ""),
                int(motion_reset),
                h_state,
                H_quality["method"],
                2,
                args.marker_pair,
                (f"{right_x_m:.4f}" if np.isfinite(right_x_m) else ""),
                (f"{right_y_m:.4f}" if np.isfinite(right_y_m) else ""),
                int(pair is not None),
                int(quality_decision.accepted),
                quality_decision.reason,
                (f"{quality_decision.position_jump_m:.4f}"
                 if np.isfinite(quality_decision.position_jump_m) else ""),
                (f"{math.degrees(quality_decision.heading_jump_rad):.2f}"
                 if np.isfinite(quality_decision.heading_jump_rad) else ""),
                (f"{quality_decision.spacing_error_m:.4f}"
                 if np.isfinite(quality_decision.spacing_error_m) else ""),
            ]
            row += calibration_debug_row(
                ref_ids=ref_ids,
                ref_world=ref_world,
                ref_seen=ref_seen,
                ref_live_positions=ref_live_positions,
                candidate_H=candidate_H,
                active_map_H=H_mat,
                candidate_quality=candidate_quality,
                active_map_quality=H_quality,
                candidate_accepted=candidate_ok,
                candidate_reject_reason=candidate_reject_reason,
                baseline_a=baseline_a,
                baseline_b=baseline_b,
                baseline_measured=baseline_measured,
                baseline_expected=baseline_expected,
                accepted_samples=len(H_samples),
                max_ref_world_error=args.max_ref_world_error,
                max_ref_range_error=args.max_ref_range_error,
                frame_shape=frame.shape,
                ref_tag_size=ref_tag_sizes,
                K=K,
                camera_dist=dist,
                geometry_dist=geometry_dist,
                geometry_distortion_mode=geometry_distortion_mode,
            )
            rows.append(row)
            if sample_callback is not None:
                try:
                    sample_callback(sample)
                except Exception as exc:
                    print(f"[ipc] velocity sample publish failed: {exc}")
                    sample_callback = None

            # Pool frame, safe region and waypoint pattern, under the live
            # markers so the vehicle overlay stays readable on top of it.
            draw_pool_overlay(
                frame, H_mat, K, dist, use_raw_geometry,
                rect_wh=pool_rect_wh, margin_m=args.safe_region_margin,
                waypoints=overlay_waypoints,
                arrival_radius_m=args.waypoint_arrival_radius,
                heading_deg=args.waypoint_heading,
                headings_deg=overlay_headings,
                visited=visited_waypoints,
                align_states=align_states)

            reference_good = H_mat is not None and (
                not np.isfinite(ref_world_rms)
                or ref_world_rms <= args.ref_drift_m)
            # The magenta baseline between the two PnP tags used to be drawn
            # here.  It carried no constraint -- the pool bounds and the safe
            # region do -- and it crossed the pattern, so it is gone; the
            # baseline pair is still named in the HUD's PnP readout below.

            reference_color = (0, 255, 0) if reference_good else (0, 165, 255)
            for i in ref_ids:
                if i in ref_seen:
                    p = ref_seen[i]["center_px"].astype(int)
                    marker_corners = np.rint(
                        ref_seen[i]["corners_px"]).astype(np.int32)
                    cv2.polylines(
                        frame, [marker_corners.reshape(-1, 1, 2)], True,
                        reference_color, 2, cv2.LINE_AA)
                    cv2.circle(frame, tuple(p), 7, reference_color, 2)
                    live = ref_live_positions[i]
                    camera_xyz = live["camera_xyz_m"]
                    pool_xy = live["pool_xy_m"]
                    pool_label = (
                        "pool xy" if H_mat is not None else "candidate xy")
                    camera_mode_label = (
                        "raw approx" if use_raw_geometry else "calibrated")
                    camera_text = (
                        f"cam xyz ({camera_mode_label})="
                        f"({camera_xyz[0]:+.2f},"
                        f"{camera_xyz[1]:+.2f},{camera_xyz[2]:+.2f}) m"
                        if live["pose_valid"] and np.isfinite(camera_xyz).all()
                        else f"cam xyz ({camera_mode_label})=PnP invalid")
                    pnp_text = (
                        f"PnP reproj={live['pnp_reproj_rms_px']:.2f} px"
                        if np.isfinite(live["pnp_reproj_rms_px"])
                        else "PnP reproj=unavailable")
                    pool_text = (
                        f"{pool_label}=({pool_xy[0]:+.3f},"
                        f"{pool_xy[1]:+.3f}) m"
                        if np.isfinite(pool_xy).all()
                        else f"{pool_label}=waiting for map")
                    draw_text_block(
                        frame, [f"ID {i}", camera_text, pnp_text, pool_text],
                        p, reference_color)
            # Segments laid down while the MPC was engaged stay red so the
            # recorded video separates MPC trajectory from manual driving.
            for (p, _), (q, q_engaged) in trail_segments(trail):
                cv2.line(frame, p, q,
                         (0, 0, 255) if q_engaged else (0, 255, 255), 2)

            n_fps += 1
            if time.time() - t_fps >= 1.0:
                fps_est, n_fps, t_fps = n_fps / (time.time() - t_fps), 0, time.time()
            if H_mat is not None and args.dynamic_homography:
                status = "H:dynamic"
            elif H_mat is not None:
                status = ("H:frozen DRIFT!"
                          if drift_count >= args.ref_drift_frames
                          else "H:frozen")
            else:
                status = f"H:init {len(H_samples)}/{args.ref_init_frames}"
            stage = (
                ("MPC MODE" if engaged else "ROV TRACKING")
                if H_mat is not None
                else "CALIBRATING REFERENCES - hold camera/tags fixed")
            stage_color = (
                (0, 0, 255) if engaged and H_mat is not None
                else (0, 255, 0) if reference_good else (0, 165, 255))
            cv2.putText(
                frame,
                f"{stage} | refs:{len(ref_seen)}/{len(ref_world)} {status} "
                f"{fps_est:.0f}fps",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                stage_color, 2)
            # Controller status line: shared Array('c') owned by the control
            # process (profile phase + reference, e.g. the step probe's
            # "BURST 3/6 u_ref=0.15").  Empty means nothing to show.
            if status_text is not None:
                try:
                    controller_line = status_text.value.decode(
                        "utf-8", "replace")
                except Exception:
                    controller_line = ""
                if controller_line:
                    cv2.putText(
                        frame, controller_line, (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                        (0, 0, 255) if engaged else (245, 245, 245), 2,
                        cv2.LINE_AA)
            if np.isfinite(baseline_measured):
                baseline_error = baseline_measured - baseline_expected
                baseline_text = (
                    f"{baseline_a}-{baseline_b} PnP({geometry_distortion_mode})="
                    f"{baseline_measured:.3f} m target={baseline_expected:.3f} m "
                    f"error={baseline_error:+.3f} m")
            else:
                baseline_text = (
                    f"{ref_ids[0]}-{ref_ids[1]} "
                    f"PnP({geometry_distortion_mode})="
                    "waiting for two valid poses")
            fit_error = candidate_quality["world_rms_m"]
            if np.isfinite(fit_error):
                baseline_text += f" | corner fit={fit_error:.3f} m"
            display_quality = H_quality if H_mat is not None else candidate_quality
            map_factor = float(display_quality.get(
                "map_scale_factor", np.nan))
            if np.isfinite(map_factor):
                baseline_text += (
                    f" | map scale={display_quality.get('map_scale_source', 'configured')}"
                    f" x{map_factor:.4f}")
            cv2.putText(
                frame, baseline_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.64, reference_color, 2, cv2.LINE_AA)
            cv2.putText(
                frame, f"frame {frame_idx}  t={t:6.2f}s",
                (20, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)

            if args.out_video and writer is None:
                size = (frame.shape[1], frame.shape[0])
                if args.video:
                    # Offline replay preserves the source timeline: one output
                    # frame per source frame, regardless of analysis wall time.
                    writer = cv2.VideoWriter(
                        args.out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                        src_fps, size)
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"could not open MP4 writer: {args.out_video}")
                else:
                    # Live tracking is slower than the camera's nominal FPS.
                    # Emit the latest complete overlay on a wall-clock cadence so
                    # the MP4 duration matches the physical run duration.
                    writer = WallClockVideoWriter(
                        args.out_video, record_fps, size)
                    writer.submit(frame)
                    writer.start()
            if isinstance(writer, WallClockVideoWriter):
                if writer.error is not None:
                    raise RuntimeError(
                        f"top-view MP4 encoder failed: {writer.error}")
                writer.submit(frame)
            elif writer is not None:
                writer.write(frame)

            # Redraw on its own cadence: imshow + waitKey costs ~6 ms a frame at
            # 1920x1200, and a preview refreshed faster than ~15 Hz looks no
            # different while stealing that time from tracking.  The resize is
            # inside the guard so --no-display pays nothing for it.
            if not args.no_display and time.monotonic() >= next_display:
                next_display = time.monotonic() + display_period
                disp = frame if frame.shape[1] <= 1280 else cv2.resize(
                    frame, (1280, int(frame.shape[0] * 1280 / frame.shape[1])))
                window_name = "ROV top-view tracking (Esc to quit)"
                if not display_created:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, disp.shape[1], disp.shape[0])
                    if args.window_x is not None and args.window_y is not None:
                        cv2.moveWindow(
                            window_name, int(args.window_x), int(args.window_y))
                    display_created = True
                cv2.imshow(window_name, disp)
                if args.show_mask:
                    m = cv2.resize(
                        cv2.bitwise_or(right_mask, rmask),
                        (disp.shape[1], disp.shape[0]))
                    cv2.imshow(f"color masks ({args.marker_pair})", m)
                key = cv2.waitKey(1)
                if is_quit_key(key):
                    break

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[run] interrupted")
    finally:
        # Always finalize.  Skipping this on Ctrl-C loses BOTH the
        # MP4 (unwritten moov atom -> unplayable) and every tracked
        # row, which is only flushed to CSV here.
        if grabber is not None:
            grabber.close()
            print(f"[run] camera delivered {grabber.frames_read} frames; "
                  f"tracker processed {frame_idx}")
        if roi_detector is not None:
            searched = roi_detector.roi_frames + roi_detector.full_frames
            share = (
                100.0 * roi_detector.roi_frames / searched if searched else 0.0)
            print(
                f"[ref] ROI reference search on {roi_detector.roi_frames} of "
                f"{searched} frames ({share:.0f}%); "
                f"{roi_detector.full_frames} full-frame sweeps")
        cap.release()
        if isinstance(writer, WallClockVideoWriter):
            writer.close()
            print(
                f"[video] wall-clock paced {writer.frames_written} frames at "
                f"{record_fps:.3f} fps; tracker produced {frame_idx} frames; "
                f"encoder dropped {writer.dropped_seconds:.3f} s")
        elif writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

        with open(args.out, "w", newline="") as f:
            cwri = csv.writer(f)
            header = [
                "frame", "host_monotonic_ns", "t_s", "t_unix_s", "n_ref",
                # Legacy compatibility: green_* contains the selected RIGHT
                # marker for both red-green and red-white runs. New consumers
                # should read marker_pair + right_* below.
                "green_x_m", "green_y_m", "red_x_m", "red_y_m",
                "x_m", "y_m", "heading_deg",
                "vx_m_s", "vy_m_s", "u_m_s", "v_port_m_s",
                "v_starboard_m_s", "r_deg_s", "r_rad_s",
                "scale", "tracking_valid", "velocity_valid",
                "ref_world_rms_m", "ref_range_rms_m",
                "color_spacing_m", "motion_reset", "homography_state",
                "reference_method",
                "tracking_quality_schema_version",
                "marker_pair",
                "right_x_m",
                "right_y_m",
                "color_pair_candidate",
                "quality_accepted",
                "quality_reject_reason",
                "quality_position_jump_m",
                "quality_heading_jump_deg",
                "quality_spacing_error_m",
            ]
            header += calibration_debug_header(ref_ids, np.asarray(dist).size)
            cwri.writerow(header)
            cwri.writerows(rows)
        print(f"done: {frame_idx} frames, {n_hit} tracked "
              f"({100 * n_hit / max(frame_idx, 1):.1f}%)")
        print(f"csv: {args.out}" + (f", video: {args.out_video}" if args.out_video else ""))


if __name__ == "__main__":
    main()
