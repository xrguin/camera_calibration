#!/usr/bin/env python3
"""Waypoint-profile preview on the calibrated iPhone stream.

Thin front-end over usb_cam_waypoints.py, exactly like iphone_apriltag.py is
for usb_cam_apriltag.py: the waypoint pattern, ghost profile, feasibility
report and reference-map machinery all live in the base tool — this file
only resolves the iPhone source and the per-camera calibration, then hands
over. Edit the waypoint pattern in usb_cam_waypoints.py as before.

  * --source: 'auto' finds the OBS Virtual Camera (USB or WiFi phone link —
    OBS owns the transport); a bare phone IP means DroidCam's direct MJPEG
    server; a /dev/videoN path, index, or full URL is used as-is.
  * --camera-profile: defaults to the iPhone 16 Wide calibration in
    usb_cam/iphone16wide/ (since 2026-08-14). The earlier iPhone 13
    calibration remains selectable as
    iphone13wide. This must match the physical phone/camera selected in the
    DroidCam app. ``--lens`` remains a compatibility alias. The
    *_continuity profiles capture the phone directly on macOS, with no
    DroidCam and no OBS in the path, from their own calibration directory.
  * the same preflight refuses a stream whose size differs from the
    calibration's native resolution (--allow-size-mismatch overrides).
  * --offline passes straight through and needs no camera at all.

Examples:

    python usb_cam/iphone_waypoints.py               # iPhone 16 Wide via OBS
    python usb_cam/iphone_waypoints.py --offline     # PNG + report, no camera
    python usb_cam/iphone_waypoints.py --camera-profile iphone13wide
    python usb_cam/iphone_waypoints.py --source 192.168.1.53 --record
"""

try:
    from usb_cam import iphone_apriltag
    from usb_cam import usb_cam_waypoints
except ImportError:  # run as a plain script: usb_cam/ itself is on sys.path
    import iphone_apriltag
    import usb_cam_waypoints


def main():
    iphone_apriltag.launch(usb_cam_waypoints.main, description=__doc__)


if __name__ == "__main__":
    main()
