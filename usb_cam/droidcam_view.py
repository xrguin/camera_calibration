#!/usr/bin/env python3
"""Live view of the iPhone DroidCam stream with OpenCV. Press q or Esc to quit.

One script for every transport — the code path is identical, only the source
string differs:

  auto              scan /dev/video* for the OBS Virtual Camera loopback
                    device (phone -> OBS via WiFi OR USB, OBS decides) [default]
  /dev/video10, 2   an explicit V4L2 device (same OBS route)
  192.168.1.53      a phone IP: opens DroidCam's own MJPEG server at
                    http://<ip>:4747/video, no OBS involved (WiFi)
  http://...        any full stream URL as-is. For DroidCam over USB without
                    OBS: `iproxy 4747 4747` then --source http://127.0.0.1:4747/video

Kept standalone like usb_cam_view.py: a fast, dependency-light way to check
what the phone is actually delivering (resolution, fps, watermark) before it
goes anywhere near the tracking pipeline.
"""
import argparse
import glob
import time

import cv2

DROIDCAM_PORT = 4747


def find_obs_virtual_cam():
    """Return the /dev/videoN whose driver name says it's the OBS loopback."""
    for name_file in sorted(glob.glob("/sys/class/video4linux/video*/name")):
        with open(name_file) as f:
            name = f.read().strip()
        if "OBS" in name or "Loopback" in name:
            return "/dev/" + name_file.split("/")[-2], name
    return None, None


def open_source(source):
    """Return (VideoCapture, description) for any accepted source form."""
    if source == "auto":
        dev, name = find_obs_virtual_cam()
        if dev is None:
            raise SystemExit(
                "No OBS Virtual Camera found among /dev/video*. Click "
                "'Start Virtual Camera' in OBS first, or pass --source "
                "with a device path, phone IP, or URL.")
        return cv2.VideoCapture(dev, cv2.CAP_V4L2), f"{dev} ({name})"

    if source.startswith(("http://", "rtsp://")):
        return cv2.VideoCapture(source), source

    if source.count(".") == 3 and not source.startswith("/"):
        url = f"http://{source}:{DROIDCAM_PORT}/video"
        return cv2.VideoCapture(url), url

    dev = int(source) if source.isdigit() else source
    return cv2.VideoCapture(dev, cv2.CAP_V4L2), str(dev)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="auto",
                    help="'auto', V4L2 index/path, phone IP, or full URL "
                         "(default: auto)")
    args = ap.parse_args()

    cap, desc = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(
            f"Cannot open {desc!r}. WiFi/URL: phone on the same network with "
            "the DroidCam app in the foreground? V4L2: is the device present "
            "and free (v4l2-ctl --list-devices)?")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Opened {desc}: {w}x{h}")

    win = "DroidCam (iPhone)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    t_prev, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed, stopping.")
            break
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        cv2.putText(frame, f"{fps:5.1f} fps  {frame.shape[1]}x{frame.shape[0]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
