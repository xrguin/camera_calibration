# USB camera (bench camera, /dev/video4)

Everything specific to the USB bench camera lives in this folder, kept
separate from the BlueROV camera files in the repo root. The BlueROV
calibration (`../camera_calibration.npz`) and detector
(`../apriltag_detector.py`) are never read or written by anything here.

## Files

- `usb_cam_view.py` — plain live viewer (`q`/Esc to quit)
- `usb_cam_apriltag.py` — tag36h11 detection + pose on the USB camera
- `usb_cam_calibration.npz` — intrinsics for THIS camera (created by
  calibration; until it exists, the detector falls back to approximate
  60-deg-FOV intrinsics and distances are not metric)

## Device

The /dev/videoN index shifts when the camera is replugged (it has already
moved 4 -> 5). All scripts therefore default to the stable by-id path:

    /dev/v4l/by-id/usb-Global_Shutter_Camera_Global_Shutter_Camera_01.00.00-video-index0

`--device` accepts either that path or a bare index.

## Commands (from the repo root)

Calibrate (SPACE=capture, c=calibrate+save, q=quit):

    ./usb_cam/calibrate.sh

The launcher bakes in the standard settings: by-id device, 9x7-square board
= 8x6 INNER corners (--cols/--rows count corners, one less than squares),
20 mm squares, output + view PNGs inside usb_cam/. Extra args pass through
and override, e.g. `./usb_cam/calibrate.sh --square 0.025`.

IMPORTANT: views live in memory until you press `c` — quitting with `q`
first discards them (the npz is only written by `c`). With --save-views,
each SPACE also writes view_NN.png + view_NN_corners.png for inspection.

Detect AprilTags with metric pose (tag-size = black-square edge in metres):

    python3 usb_cam/usb_cam_apriltag.py --tag-size 0.10

Plain live view:

    python3 usb_cam/usb_cam_view.py --width 1920 --height 1200

The camera maxes out at 1920x1200 @ 30 fps (MJPG). Its second by-id entry
(video-index1) is the metadata node and produces no frames.
