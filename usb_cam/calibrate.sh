#!/usr/bin/env bash
# Calibrate the USB bench camera with its standard settings:
# stable by-id device, the large 10x8 board with 200 mm squares, output +
# view PNGs kept inside usb_cam/. Extra args pass through, e.g.
#   ./usb_cam/calibrate.sh --require-coverage
#
# WHY THIS EXISTS IN THIS FORM
#
# This camera is ~116 deg horizontal FOV. The previous calibration used a
# 5-coefficient model fitted on views that never reached past 63% of the
# image-corner radius, so the radial polynomial folded beyond that: at the
# frame edges undistortPoints stopped converging and returned its input
# unchanged, and the 0.20 m pool tags sitting there reported poses ~230 px
# wrong while the image centre still looked perfect. Hence:
#
#   --model rational   k1..k6, which stays monotonic out to the frame edge
#   --board-units auto 10x8 is accepted as either squares or inner corners;
#                      whichever matches is locked in and printed
#
# The old calib_views/ and calib_views2/ are NOT seeded here: they are the
# small 8x6 / 20 mm board, and one --square cannot describe two boards. The
# new 2.0 x 1.6 m board covers the centre in a handful of shots anyway. Add
# --load-views explicitly if you ever want them back.
#
# WHAT TO CAPTURE: the live overlay tiles the image into 24 cells and marks
# every cell red until a captured view has put board corners in it. Fill them
# all, and push the radius readout past 90%. The bottom of the frame and the
# far left/right are what the old sessions never reached.
#
# To run:
#   cd ~/Documents/EDMDc_bluerov
#   ./usb_cam/calibrate.sh
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

# cv2/numpy live in the 'rov' conda env, not in miniconda base, and base is
# what a bare `python3` resolves to here.  Check before opening the camera:
# discovering this at the poolside as a bare ModuleNotFoundError wastes a
# session.  Override with PYTHON=/path/to/python ./usb_cam/calibrate.sh
PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c "import cv2, numpy" 2>/dev/null; then
    echo "error: '$PYTHON' cannot import cv2/numpy." >&2
    echo "       Activate the project environment first:" >&2
    echo "           conda activate rov" >&2
    echo "       or point this script at one:" >&2
    echo "           PYTHON=~/miniconda3/envs/rov/bin/python $0" >&2
    exit 1
fi

exec "$PYTHON" "$DIR/../calibrate_camera.py" \
    --device /dev/v4l/by-id/usb-Global_Shutter_Camera_Global_Shutter_Camera_01.00.00-video-index0 \
    --cols 10 --rows 8 --board-units auto --square 0.20 \
    --model rational \
    --out "$DIR/usb_cam_calibration.npz" \
    --save-views "$DIR/calib_views_20cm" \
    "$@"
