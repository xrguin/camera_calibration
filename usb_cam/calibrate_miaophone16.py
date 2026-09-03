#!/usr/bin/env python3
"""Calibrate the MiaoPhone 16 as a macOS Continuity Camera, in one command.

    python usb_cam/calibrate_miaophone16.py

No device ID to copy and no flags to remember: the phone is found by name,
the stream is checked before any view is captured, and the result lands in
usb_cam/iphone16wide_continuity/ -- the directory combined_view looks in
before it decides whether it can skip OBS.  Extra arguments pass through to
calibrate_camera.py unchanged.

This calibrates the DIRECT pipeline: macOS publishes the paired iPhone as a
camera in its own right, so DroidCam, OBS and the virtual camera are all out
of the path.  That is a different pipeline from the one behind
usb_cam/iphone16wide/calibration.npz, which is why it is a separate file --
Continuity can select a different lens and crop than DroidCam did, and those
intrinsics on this stream would give wrong poses rather than an error.

A calibration only describes the configuration it was captured with, so
before starting, and identically for every later run:
  - turn Center Stage OFF (System Settings > ... > Video Effects, or Control
    Centre while the camera is live).  It re-crops the frame as people move,
    which changes the intrinsics mid-session.
  - turn off any other video effect (Portrait, Studio Light, Reactions).
  - mount the phone as it will be over the pool and do not zoom afterwards.

Board and model follow calibrate_iphone.sh: the 10x8 board and the rational
k1..k6 model, which a lens this wide needs to stay monotonic to the frame
edge (a plain 5-coefficient fit folds past ~63% radius and corrupts edge
poses).  --auto-capture holds the session to one pair of hands, and
--require-coverage refuses a fit with an empty cell.
"""

import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PROFILE = "iphone16wide_continuity"
OUTDIR = os.path.join(DIR, PROFILE)
# The phone is matched on its name first because that is what the operator
# sees in the camera list; the iPhone 16 family model ID is the fallback for
# a renamed handset.
DEVICE_NAME = "MiaoPhone"
MODEL_PREFIX = "iPhone17,"
EXPECT_W, EXPECT_H = 1920, 1080

BOARD_ARGS = [
    "--cols", "10", "--rows", "8", "--board-units", "auto", "--square", "0.20",
    "--model", "rational",
    "--auto-capture", "1.5",
    "--require-coverage",
    # The phone paces its own frames; the 90 fps request only makes sense on
    # the bench camera.
    "--cam-fps", "0",
    "--cam-width", str(EXPECT_W), "--cam-height", str(EXPECT_H),
]


def find_phone():
    """The avf: identity of the phone, or exit saying what is attached."""

    from usb_cam.track_rov_topview import (
        AVF_DEVICE_PREFIX, macos_video_devices)

    devices = macos_video_devices()
    for match in (lambda d: DEVICE_NAME.lower() in d["name"].lower(),
                  lambda d: d["model_id"].startswith(MODEL_PREFIX)):
        found = [d for d in devices if match(d)]
        if len(found) == 1:
            print(f"[camera] {found[0]['name']} [{found[0]['model_id']}]")
            return AVF_DEVICE_PREFIX + found[0]["unique_id"]
        if len(found) > 1:
            listing = ", ".join(d["name"] for d in found)
            raise SystemExit(
                f"[camera] several cameras match ({listing}). Pass "
                "--device avf:<uniqueID> from  python calibrate_camera.py "
                "--list-cameras")
    listing = ", ".join(
        f"{d['name']} [{d['model_id']}]" for d in devices) or "none"
    raise SystemExit(
        f"[camera] no {DEVICE_NAME} among the attached cameras: {listing}\n"
        "  Wake the phone, keep it unlocked and on the same Apple account, "
        "and check Continuity Camera is enabled on it.")


def preflight(device):
    """One frame, before any views: right size, and not an empty stream."""

    import time

    from usb_cam.track_rov_topview import open_capture

    cap = open_capture(device, width=EXPECT_W, height=EXPECT_H)
    ok, frame = False, None
    # The phone's first frames arrive while auto-exposure is still ramping
    # and are genuinely black, so wait for one with content before judging.
    deadline = time.monotonic() + 6.0
    while cap.isOpened() and time.monotonic() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None and frame.any():
            break
    cap.release()
    if not ok:
        raise SystemExit(
            "[preflight] no frame from the phone. Is it awake and unlocked? "
            "On macOS also check System Settings > Privacy & Security > "
            "Camera for the terminal running this.")
    if not frame.any():
        raise SystemExit(
            "[preflight] the phone delivers an entirely black frame -- the "
            "camera is open but showing nothing.")
    h, w = frame.shape[:2]
    if (w, h) != (EXPECT_W, EXPECT_H):
        # Not fatal: what the calibration records is what it was captured at,
        # and every consumer checks that size later.  It does mean the runs
        # must use this same size, so it is said out loud.
        print(f"[preflight] note: the stream is {w}x{h}, not "
              f"{EXPECT_W}x{EXPECT_H}. The calibration will be written for "
              f"{w}x{h}, and runs must use that same configuration.")
    else:
        print(f"[preflight] {w}x{h} OK")


def main():
    if {"-h", "--help", "--list-cameras"} & set(sys.argv[1:]):
        # Asking what the flags are must not wake the phone and open its
        # camera, so these go straight through.
        import calibrate_camera

        return calibrate_camera.main()

    if sys.platform != "darwin":
        raise SystemExit(
            "Continuity Camera is macOS-only. On Linux the phone reaches "
            "this machine through OBS: ./usb_cam/calibrate_iphone.sh "
            "iphone16wide")

    device = find_phone()
    preflight(device)
    os.makedirs(OUTDIR, exist_ok=True)

    out = os.path.join(OUTDIR, "calibration.npz")
    views = os.path.join(OUTDIR, "calib_views")
    print(f"[calib] writing {out}")
    print("[calib] hold the board still over a red cell and it captures "
          "itself; fill every cell, especially the corners")

    import calibrate_camera

    # Injected defaults first so an explicit flag on the command line wins.
    sys.argv = ([sys.argv[0], "--device", device]
                + BOARD_ARGS
                + ["--out", out, "--save-views", views]
                + sys.argv[1:])
    return calibrate_camera.main()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
