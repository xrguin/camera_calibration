#!/usr/bin/env python3
"""Live view of a USB camera with OpenCV. Press q or Esc to quit."""
import argparse
import time

import cv2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device",
                    default=("/dev/v4l/by-id/usb-Global_Shutter_Camera_"
                             "Global_Shutter_Camera_01.00.00-video-index0"),
                    help="V4L2 device index or path (default: stable by-id path)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=90.0,
                    help="Requested capture rate (default 90). Without this "
                         "the driver stays at its 30 fps default, which "
                         "measures ~15 fps.")
    args = ap.parse_args()

    # Kept standalone (no track_rov_topview import) so this stays a fast,
    # dependency-light way to check what the camera is actually delivering.
    # BUFFERSIZE is deliberately left at the driver default: a single buffer
    # makes the camera idle while userspace copies, roughly halving the rate.
    dev = int(args.device) if str(args.device).isdigit() else args.device
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        cap.set(cv2.CAP_PROP_FPS, float(args.fps))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.device!r} — check "
                         "ls /dev/v4l/by-id/ and that nothing else has it open")

    win = "USB cam (bench camera)"
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
        cv2.putText(frame, f"{fps:5.1f} fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
