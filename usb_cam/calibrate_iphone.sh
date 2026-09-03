#!/usr/bin/env bash
# Calibrate the iPhone (DroidCam) camera, one profile per lens.
#
#   ./usb_cam/calibrate_iphone.sh iphone13wide                 # OBS, auto-found
#   ./usb_cam/calibrate_iphone.sh iphone16wide 192.168.1.53    # direct WiFi
#   ./usb_cam/calibrate_iphone.sh iphone13wide /dev/video10    # explicit V4L2
#   ./usb_cam/calibrate_iphone.sh iphone13wide http://127.0.0.1:4747/video
#   ./usb_cam/calibrate_iphone.sh iphone16wide_continuity      # macOS, no OBS
#
# Extra args pass through to calibrate_camera.py, e.g. --require-coverage.
#
# The iPhone 13 Wide and iPhone 16 camera profiles have different intrinsics,
# so each physical camera gets its own unambiguous folder under usb_cam/:
#
#   iphone13wide/calibration.npz + calib_views/
#   iphone16wide/calibration.npz + calib_views/
#
# A calibration also belongs to one capture pipeline, not just to one phone.
# On macOS a paired iPhone is a Continuity Camera that OpenCV can open with
# no DroidCam and no OBS in the path, and that pipeline chooses its own lens
# and crop -- so it is its own profile and its own folder:
#
#   iphone13wide_continuity/calibration.npz + calib_views/
#   iphone16wide_continuity/calibration.npz + calib_views/
#
# combined_view prefers a _continuity profile automatically once its
# calibration exists and the phone is attached; until then it stays on OBS.
#
# Point the tracking scripts at the matching file with
#   --calib usb_cam/iphone13wide/calibration.npz
#
# Board and model follow calibrate.sh: the large 10x8 board with 200 mm
# squares and the rational k1..k6 model. For the 0.5x ultra-wide the rational
# model is not optional — it is in the same ~120 deg class as the bench
# camera, where the plain 5-coefficient fit folded beyond 63% radius and
# corrupted edge poses (see calibrate.sh).
#
# A calibration is only valid for the exact phone configuration it was
# captured with. The standard setup is 1920x1080 (DroidCam Pro) — a
# preflight below refuses to start on any other frame size, so a phone
# that fell back to the free-tier 640x480 or an OBS canvas mismatch is
# caught before a session is wasted, not after. Before starting, and
# identically for every later use:
#   - select the lens in the DroidCam app (1x or 0.5x), zero digital zoom
#   - set the app to 1920x1080; disable any stabilization it exposes
#   - OBS canvas = output = 1920x1080, no crop/scale/filters on the source
# Override the size gate with EXPECT=WxH (or EXPECT=any) if you knowingly
# calibrate another configuration.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

REQUESTED_PROFILE="${1:-}"
case "$REQUESTED_PROFILE" in
    iphone13wide|1x)
        PROFILE="iphone13wide"
        OUTDIR="$DIR/iphone13wide"
        ;;
    iphone16wide|0.5x|0p5x)
        PROFILE="iphone16wide"
        OUTDIR="$DIR/iphone16wide"
        ;;
    iphone13wide_continuity|iphone16wide_continuity)
        PROFILE="$REQUESTED_PROFILE"
        OUTDIR="$DIR/$REQUESTED_PROFILE"
        ;;
    *)
        echo "usage: $0 <iphone13wide|iphone16wide>[_continuity] [source]" \
             "[extra args]" >&2
        echo "       source: phone IP, /dev/videoN, index, or http/rtsp URL" >&2
        echo "               (default: auto-detect the OBS Virtual Camera)" >&2
        echo "       legacy aliases 1x and 0.5x are still accepted" >&2
        exit 1 ;;
esac
shift

# Second positional arg (if not an --option) is the source.
SOURCE=""
if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then
    SOURCE="$1"
    shift
fi

# cv2/numpy live in the 'rov' conda env, not miniconda base (see calibrate.sh).
# Resolved before the source, because macOS camera discovery runs through it.
PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c "import cv2, numpy" 2>/dev/null; then
    echo "error: '$PYTHON' cannot import cv2/numpy." >&2
    echo "       Activate the project environment first:" >&2
    echo "           conda activate rov" >&2
    echo "       or point this script at one:" >&2
    echo "           PYTHON=~/miniconda3/envs/rov/bin/python $0 $PROFILE" >&2
    exit 1
fi

case "$SOURCE" in
    "")
        if [ "$(uname -s)" = "Darwin" ]; then
            # No /dev/video* on macOS: the same discovery the launcher uses
            # picks the phone itself for a _continuity profile, otherwise the
            # OBS Virtual Camera, and returns an avf:<uniqueID> identity.
            SOURCE="$("$PYTHON" - "$DIR/.." "$PROFILE" <<'EOF'
import sys

sys.path.insert(0, sys.argv[1])
from usb_cam.iphone_apriltag import (
    describe_device, find_direct_iphone_cam, find_obs_virtual_cam,
    is_continuity_profile)

profile = sys.argv[2]
device = (find_direct_iphone_cam(profile) if is_continuity_profile(profile)
          else find_obs_virtual_cam())
if device is None:
    sys.exit(1)
print(device)
print(describe_device(device), file=sys.stderr)
EOF
)" || {
                echo "error: no camera for $PROFILE among the macOS" >&2
                echo "       cameras. For a _continuity profile wake the" >&2
                echo "       phone and check Continuity Camera; otherwise" >&2
                echo "       click 'Start Virtual Camera' in OBS. Or pass a" >&2
                echo "       source: phone IP, device index, or URL." >&2
                exit 1
            }
            echo "[source] $PROFILE at $SOURCE"
        else
            # Find the OBS Virtual Camera loopback device by name.
            for n in /sys/class/video4linux/video*/name; do
                [ -e "$n" ] || continue
                if grep -qE 'OBS|Loopback' "$n"; then
                    SOURCE="/dev/$(basename "$(dirname "$n")")"
                    break
                fi
            done
            if [ -z "$SOURCE" ]; then
                echo "error: no OBS Virtual Camera found among /dev/video*." >&2
                echo "       Click 'Start Virtual Camera' in OBS, or pass a" >&2
                echo "       source: phone IP, device path, or URL." >&2
                exit 1
            fi
            echo "[source] OBS Virtual Camera at $SOURCE"
        fi
        ;;
    http://*|rtsp://*|/dev/*|avf:*|[0-9]*)
        ;;  # already usable as-is
    *)
        # Bare phone IP/hostname -> DroidCam's own MJPEG endpoint.
        SOURCE="http://$SOURCE:4747/video"
        echo "[source] direct DroidCam stream at $SOURCE"
        ;;
esac

mkdir -p "$OUTDIR"

# Preflight: one frame, correct size, before any views are captured.
EXPECT="${EXPECT:-1920x1080}"
if [ "$EXPECT" != "any" ]; then
    "$PYTHON" - "$SOURCE" "$EXPECT" "$DIR/.." <<'EOF'
import sys
import cv2

sys.path.insert(0, sys.argv[3])
from usb_cam.track_rov_topview import capture_backend, resolve_capture_device

source, expect = sys.argv[1], sys.argv[2]
want_w, want_h = (int(v) for v in expect.split("x"))
if source.startswith(("http://", "rtsp://")):
    cap = cv2.VideoCapture(source)
else:
    cap = cv2.VideoCapture(resolve_capture_device(source), capture_backend())
ok, frame = cap.read() if cap.isOpened() else (False, None)
cap.release()
if not ok:
    sys.exit(f"[preflight] no frame from {source!r} — is the phone "
             "connected and (for OBS) the virtual camera started?")
if not frame.any():
    sys.exit(f"[preflight] {source!r} delivers an entirely black frame — the "
             "capture source is running but showing nothing. In OBS, check "
             "the scene renders the phone.")
h, w = frame.shape[:2]
if (w, h) != (want_w, want_h):
    sys.exit(f"[preflight] stream is {w}x{h}, expected {want_w}x{want_h}.\n"
             "  640x480  -> phone still on the free tier, or app resolution unset\n"
             "  other    -> OBS canvas/output not matching the source\n"
             "  Fix the pipeline, or override with EXPECT=WxH / EXPECT=any.")
print(f"[preflight] {source}: {w}x{h} OK")
EOF
fi

# --cam-fps 0 skips the V4L2 rate request: the loopback and HTTP sources set
# their own rate and the 90 fps default only makes sense on the bench camera.
# --auto-capture: hold the board still over a red cell and it captures by
# itself (SPACE still works) — this session is one person holding a phone
# rig AND a board. --require-coverage: an iPhone calibration with edge gaps
# is exactly the folded-model failure calibrate.sh documents, so refuse it.
exec "$PYTHON" "$DIR/../calibrate_camera.py" \
    --device "$SOURCE" \
    --cam-fps 0 \
    --cols 10 --rows 8 --board-units auto --square 0.20 \
    --model rational \
    --auto-capture 1.5 \
    --require-coverage \
    --out "$OUTDIR/calibration.npz" \
    --save-views "$OUTDIR/calib_views" \
    "$@"
