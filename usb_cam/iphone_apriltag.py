#!/usr/bin/env python3
"""AprilTag (tag36h11) detection + pose on the calibrated iPhone stream.

Thin front-end over usb_cam_apriltag.py: identical detection, reference-map
and recording machinery — this file only resolves the iPhone-specific pieces
and hands over:

  * --source: 'auto' finds the OBS Virtual Camera — among /dev/video* on
    Linux, by AVFoundation device name on macOS, where it resolves to an
    avf:<uniqueID> identity (works for both USB and WiFi phone connections
    — OBS owns the transport); under a *_continuity profile it finds the
    phone itself instead. A bare phone IP means DroidCam's own MJPEG server
    (http://IP:4747/video, no OBS); a /dev/videoN path, index, or full URL
    is used as-is.
  * --camera-profile: picks the matching physical phone/camera calibration.
    The default is the iPhone 16 Wide camera in usb_cam/iphone16wide/
    (since 2026-08-14). The earlier iPhone 13 profile remains available in
    usb_cam/iphone13wide/. The camera selected in the DroidCam app MUST match
    the profile — using either phone's stream with the other calibration
    produces confidently wrong poses, not an error. ``--lens`` remains a
    compatibility alias for existing commands.
  * the *_continuity profiles (macOS, since 2026-08-14): the paired iPhone
    is a camera in its own right, so DroidCam, OBS and the virtual camera
    all leave the path. That pipeline can select a different lens or crop,
    hence its own directory (usb_cam/iphone16wide_continuity/) rather than
    a second reader of the OBS calibration. --device-name picks between two
    phones of one model family.
  * a preflight frame-size check: the stream must match the calibration's
    native resolution, because the silent K-rescale in load_calibration()
    is only correct for a true scale, and a mismatch here usually means the
    OBS output settings drifted. Override with --allow-size-mismatch.

Every other flag of usb_cam_apriltag.py passes through unchanged, e.g.:

    python usb_cam/iphone_apriltag.py             # iPhone 16 Wide, via OBS
    python usb_cam/iphone_apriltag.py --record
    python usb_cam/iphone_apriltag.py --camera-profile iphone13wide
    python usb_cam/iphone_apriltag.py --camera-profile iphone16wide_continuity
    python usb_cam/iphone_apriltag.py --source 192.168.1.53 --tag-size 0.20
"""
import argparse
import glob
import os
import sys

import cv2

try:
    from usb_cam import usb_cam_apriltag as base
    from usb_cam.track_rov_topview import (
        AVF_DEVICE_PREFIX, macos_video_devices, open_capture)
except ImportError:  # run as a plain script: usb_cam/ itself is on sys.path
    import usb_cam_apriltag as base
    from track_rov_topview import (
        AVF_DEVICE_PREFIX, macos_video_devices, open_capture)

DIR = os.path.dirname(os.path.abspath(__file__))
DROIDCAM_PORT = 4747
DEFAULT_CAMERA_PROFILE = "iphone16wide"
# A calibration belongs to one capture pipeline, not just to one phone: the
# DroidCam->OBS path and macOS Continuity Camera can pick different lenses
# and crops off the same handset, so each gets its own directory and the
# "_continuity" suffix is the transport, not a second lens.
CONTINUITY_SUFFIX = "_continuity"
CAMERA_PROFILE_DIRS = {
    "iphone13wide": "iphone13wide",
    "iphone16wide": "iphone16wide",
    "iphone13wide_continuity": "iphone13wide_continuity",
    "iphone16wide_continuity": "iphone16wide_continuity",
}
CAMERA_PROFILE_ALIASES = {
    "1x": "iphone13wide",
    "0.5x": "iphone16wide",
    "0p5x": "iphone16wide",
}
# AVFoundation model IDs for the handsets these profiles describe. The
# families overlap the neighbouring model year (iPhone14,7 is an iPhone 14),
# which only matters when two phones of one family are attached at once --
# find_direct_iphone_cam() refuses that rather than guessing.
PROFILE_MODEL_PREFIXES = {
    "iphone13wide": ("iPhone14,",),
    "iphone16wide": ("iPhone17,",),
}


def is_continuity_profile(camera_profile):
    """True when this profile captures the phone directly, without OBS."""

    return camera_profile.endswith(CONTINUITY_SUFFIX)


def base_camera_profile(camera_profile):
    """The handset identity behind a profile, transport suffix removed."""

    if is_continuity_profile(camera_profile):
        return camera_profile[:-len(CONTINUITY_SUFFIX)]
    return camera_profile


def camera_profile_name(value):
    """Normalize compatibility spellings to a physical camera identity."""

    profile = CAMERA_PROFILE_ALIASES.get(value, value)
    if profile not in CAMERA_PROFILE_DIRS:
        choices = ", ".join(sorted(CAMERA_PROFILE_DIRS))
        raise argparse.ArgumentTypeError(
            f"camera profile must be one of: {choices}")
    return profile


def find_obs_virtual_cam():
    """Return the OBS Virtual Camera: a /dev/videoN path, or an avf: token.

    Linux reads the v4l2loopback driver name.  macOS matches the
    AVFoundation device name and returns the ``avf:<uniqueID>`` identity
    that usb_cam/avf_capture.py opens by identity, never by index.
    """
    if sys.platform == "darwin":
        for device in macos_video_devices():
            if "OBS" in device["name"] or "OBS" in device["model_id"]:
                return AVF_DEVICE_PREFIX + device["unique_id"]
        return None
    for name_file in sorted(glob.glob("/sys/class/video4linux/video*/name")):
        with open(name_file) as f:
            name = f.read().strip()
        if "OBS" in name or "Loopback" in name:
            return "/dev/" + name_file.split("/")[-2]
    return None


def find_direct_iphone_cam(camera_profile, name_hint=None):
    """Return the ``avf:`` identity of the phone a profile names, or None.

    macOS publishes a paired iPhone as a Continuity Camera, so the phone can
    be captured with no DroidCam and no OBS in the path.  The physical phone
    is matched on the AVFoundation model ID (``iPhone17,x`` is the 16 family,
    ``iPhone14,x`` the 13) because the device *name* is whatever the owner
    called their phone.  ``name_hint`` is a case-insensitive substring for
    the case that leaves ambiguous -- two phones of the same family, which
    otherwise raises rather than guessing which one is over the pool.
    """
    prefixes = PROFILE_MODEL_PREFIXES.get(base_camera_profile(camera_profile))
    if not prefixes:
        return None
    matches = [d for d in macos_video_devices()
               if d["model_id"].startswith(prefixes)]
    if name_hint:
        wanted = name_hint.lower()
        matches = [d for d in matches if wanted in d["name"].lower()]
    if not matches:
        return None
    if len(matches) > 1:
        listing = ", ".join(f"{d['name']} [{d['model_id']}]" for d in matches)
        raise ValueError(
            f"{camera_profile}: several cameras match ({listing}). Name the "
            "one over the pool with --top-device-name (combined_view) or "
            "--device-name, or pass the device explicitly.")
    return AVF_DEVICE_PREFIX + matches[0]["unique_id"]


def describe_device(device):
    """Human-readable name for a device spec, for the pre-run log line."""

    text = str(device)
    if not text.startswith(AVF_DEVICE_PREFIX):
        return text
    unique_id = text[len(AVF_DEVICE_PREFIX):]
    for entry in macos_video_devices():
        if entry["unique_id"] == unique_id:
            return f"{entry['name']} [{entry['model_id']}]"
    return text


def resolve_source(source, camera_profile=None, name_hint=None):
    if source == "auto":
        if camera_profile and is_continuity_profile(camera_profile):
            dev = find_direct_iphone_cam(camera_profile, name_hint)
            if dev is None:
                raise SystemExit(
                    f"No camera matching {camera_profile} is connected. "
                    "Wake the phone and check it is signed into the same "
                    "Apple account with Continuity Camera enabled, pick the "
                    "OBS profile "
                    f"({base_camera_profile(camera_profile)}) instead, or "
                    "pass --source explicitly.")
            print(f"[source] {describe_device(dev)} direct, no OBS")
            return dev
        dev = find_obs_virtual_cam()
        if dev is None:
            where = ("among the macOS AVFoundation cameras"
                     if sys.platform == "darwin" else "among /dev/video*")
            raise SystemExit(
                f"No OBS Virtual Camera found {where}. Click "
                "'Start Virtual Camera' in OBS first, or pass --source "
                "with a device path/index, phone IP, or URL.")
        print(f"[source] OBS Virtual Camera at {describe_device(dev)}")
        return dev
    if source.startswith(("http://", "rtsp://")) or source.startswith("/"):
        return source
    if source.count(".") == 3:
        url = f"http://{source}:{DROIDCAM_PORT}/video"
        print(f"[source] direct DroidCam stream at {url}")
        return url
    return source  # numeric index


def preflight(source, calib_path, allow_mismatch):
    """One frame off the stream; its size must match the calibration's."""
    import numpy as np

    data = np.load(calib_path)
    cal_w, cal_h = int(data["image_w"]), int(data["image_h"])

    cap = open_capture(source)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    cap.release()
    if not ok:
        hint = (" On macOS also check System Settings > Privacy & Security >"
                " Camera for the terminal running this."
                if sys.platform == "darwin" else "")
        raise SystemExit(
            f"[preflight] no frame from {source!r} — phone connected, app in "
            f"the foreground, and (for OBS) the virtual camera started?{hint}")
    h, w = frame.shape[:2]
    if (w, h) != (cal_w, cal_h):
        msg = (f"[preflight] stream is {w}x{h} but the calibration is "
               f"{cal_w}x{cal_h} native. Usual cause: OBS output settings "
               "drifted, or the phone app resolution changed.")
        if not allow_mismatch:
            raise SystemExit(msg + "\n  Fix the pipeline, or run with "
                             "--allow-size-mismatch to rescale K anyway.")
        print(msg + " Continuing (--allow-size-mismatch).")
    else:
        print(f"[preflight] {source}: {w}x{h} matches calibration")
    return w, h


def launch(base_main, description=None):
    """Resolve the iPhone flags, then hand over to another usb_cam tool.

    Shared by iphone_waypoints.py (and any future iPhone front-end): parses
    --source/--camera-profile/--allow-size-mismatch, injects the resolved
    --device, --width/--height, --fps 0 and --calib into sys.argv, and calls
    the base tool's main(). Unrecognized flags pass through untouched.
    """
    ap = argparse.ArgumentParser(
        description=description or __doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="auto",
                    help="'auto' (the OBS Virtual Camera, or the phone "
                         "itself under a *_continuity profile), camera "
                         "index/path, phone IP, or full URL (default: auto)")
    ap.add_argument(
        "--camera-profile", "--lens", dest="camera_profile",
        type=camera_profile_name, choices=sorted(CAMERA_PROFILE_DIRS),
        default=DEFAULT_CAMERA_PROFILE,
        help="physical iPhone camera whose calibration to load; --lens is "
             "a compatibility alias and legacy values 1x/0.5x are accepted. "
             "A *_continuity profile captures the phone directly on macOS "
             "instead of through OBS (default: iphone16wide)")
    ap.add_argument(
        "--device-name", default=None,
        help="substring of the macOS camera name to use when two phones of "
             "the same model family are attached (*_continuity profiles)")
    ap.add_argument("--allow-size-mismatch", action="store_true",
                    help="Run even if the stream size differs from the "
                         "calibration's native size (K gets rescaled)")
    args, extra = ap.parse_known_args()

    calib = os.path.join(
        DIR, CAMERA_PROFILE_DIRS[args.camera_profile], "calibration.npz")
    if not os.path.exists(calib):
        raise SystemExit(
            f"{calib} not found — calibrate this camera profile first:\n"
            f"    ./usb_cam/calibrate_iphone.sh {args.camera_profile}")

    if "--offline" in extra:
        # No camera involved (usb_cam_waypoints.py --offline): skip source
        # resolution and preflight, only supply the calibration.
        sys.argv = [sys.argv[0], "--calib", calib] + extra
        return base_main()

    source = resolve_source(args.source, args.camera_profile, args.device_name)
    w, h = preflight(source, calib, args.allow_size_mismatch)

    # Injected defaults first, the operator's pass-through flags last, so an
    # explicit --tag-size/--fps/... on our command line still wins.
    sys.argv = [sys.argv[0],
                "--device", str(source),
                "--width", str(w), "--height", str(h),
                "--fps", "0",
                "--calib", calib] + extra
    return base_main()


def main():
    launch(base.main)


if __name__ == "__main__":
    main()
