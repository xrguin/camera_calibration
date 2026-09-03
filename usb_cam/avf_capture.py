"""Open a macOS camera by unique ID, because its index is not a name.

OpenCV's AVFoundation backend can only open a camera by index, and macOS
will not hold that index still: the device array reorders on essentially
every camera open, as Continuity Camera devices attach, sleep and wake.
Measured on this project's mac, opening one camera moved the others every
single time -- so "read the list, find our camera at position 2, ask OpenCV
for index 2" is a race that silently hands back a different camera.  That is
not a theoretical risk; it is how a calibration run ended up looking at the
OBS Virtual Camera instead of the phone.

This module removes the index from the picture.  AVFoundation itself can
open a device by ``uniqueID``, so a session is built here and frames are
pulled off an ``AVCaptureVideoDataOutput`` into numpy arrays.  The result is
duck-type compatible with ``cv2.VideoCapture`` -- ``isOpened()``, ``read()``,
``release()``, ``set()``, ``get()`` -- so every caller in this repository
keeps working unchanged.

It also buys format control the OBS path does not have.  The order matters
and is not obvious: ``AVCaptureSession`` applies its preset (1920x1080 at
30 fps) when the configuration is committed, overwriting anything set
before, so the format must be applied AFTER ``startRunning``.  Done in that
order, an iPhone 16 Continuity Camera delivers a measured 60.2 fps at
1920x1440; done before, the same code silently gets 30 fps at 1920x1080.

Requires pyobjc (``pyobjc-framework-AVFoundation`` and
``pyobjc-framework-libdispatch``).  macOS only; every other platform keeps
using cv2.VideoCapture.
"""

import sys
import threading
import time

AVF_DEVICE_PREFIX = "avf:"

# The ObjC delegate class may only be defined once per process, so the
# bindings and the class are built on first use and cached.
_BINDINGS = None
_SINK_CLASS = None
# kCVPixelFormatType_32BGRA: the BGRA byte order OpenCV already expects,
# minus the alpha channel, so no colour conversion is needed per frame.
_PIXEL_FORMAT_32BGRA = 0x42475241


def is_avf_device(device):
    """True for the ``avf:<uniqueID>`` spec this module opens."""

    return str(device).startswith(AVF_DEVICE_PREFIX)


def device_unique_id(device):
    return str(device)[len(AVF_DEVICE_PREFIX):]


def _bindings():
    global _BINDINGS

    if _BINDINGS is None:
        try:
            import AVFoundation
            import libdispatch
            import objc
            from CoreMedia import (
                CMSampleBufferGetImageBuffer, CMTimeMake,
                CMVideoFormatDescriptionGetDimensions)
            from Foundation import NSObject
            from Quartz import (
                CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow,
                CVPixelBufferGetHeight, CVPixelBufferGetWidth,
                CVPixelBufferLockBaseAddress, CVPixelBufferUnlockBaseAddress)
        except ImportError as exc:
            # Naming a camera by identity needs these; they are per-Python
            # environment, so an install into another env does not help the
            # one actually running.
            raise SystemExit(
                f"{exc}. Naming a macOS camera by identity needs pyobjc. "
                f"Install it into THIS environment ({sys.executable}):\n"
                f"    {sys.executable} -m pip install "
                "pyobjc-framework-AVFoundation pyobjc-framework-libdispatch"
            ) from exc

        _BINDINGS = {
            "AVF": AVFoundation, "objc": objc, "NSObject": NSObject,
            "dispatch_queue_create": libdispatch.dispatch_queue_create,
            "CMSampleBufferGetImageBuffer": CMSampleBufferGetImageBuffer,
            "CMTimeMake": CMTimeMake,
            "CMVideoFormatDescriptionGetDimensions":
                CMVideoFormatDescriptionGetDimensions,
            "lock": CVPixelBufferLockBaseAddress,
            "unlock": CVPixelBufferUnlockBaseAddress,
            "base": CVPixelBufferGetBaseAddress,
            "stride": CVPixelBufferGetBytesPerRow,
            "width": CVPixelBufferGetWidth,
            "height": CVPixelBufferGetHeight,
        }
    return _BINDINGS


def _closest_supported_rate(wanted, ranges):
    """The nearest frame rate the format actually offers."""

    if not ranges:
        return wanted
    options = []
    for entry in ranges:
        low, high = entry.minFrameRate(), entry.maxFrameRate()
        options.append(min(max(wanted, low), high))
    return min(options, key=lambda rate: abs(rate - wanted))


def _sink_class():
    """The sample-buffer delegate: newest frame wins, no queue.

    Mirrors FrameGrabber in track_rov_topview.py -- when processing falls
    behind, intermediate frames are dropped rather than queued, and the
    sequence number reveals the skips.
    """
    global _SINK_CLASS

    if _SINK_CLASS is not None:
        return _SINK_CLASS

    import numpy as np

    api = _bindings()
    objc = api["objc"]

    class _AVFSink(api["NSObject"]):
        def init(self):
            self = objc.super(_AVFSink, self).init()
            if self is None:
                return None
            self.condition = threading.Condition()
            self.frame = None
            self.sequence = 0
            return self

        def captureOutput_didOutputSampleBuffer_fromConnection_(
                self, output, sample_buffer, connection):
            pixels = api["CMSampleBufferGetImageBuffer"](sample_buffer)
            if pixels is None:
                return
            api["lock"](pixels, 1)  # kCVPixelBufferLock_ReadOnly
            try:
                width = api["width"](pixels)
                height = api["height"](pixels)
                stride = api["stride"](pixels)
                raw = api["base"](pixels).as_buffer(stride * height)
                rows = np.frombuffer(raw, dtype=np.uint8).reshape(
                    height, stride // 4, 4)
                # Copy: the pixel buffer is recycled as soon as we unlock.
                # Dropping alpha leaves the BGR order OpenCV works in.
                frame = rows[:, :width, :3].copy()
            finally:
                api["unlock"](pixels, 1)
            with self.condition:
                self.frame = frame
                self.sequence += 1
                self.condition.notify_all()

    _SINK_CLASS = _AVFSink
    return _SINK_CLASS


class AVFCamera:
    """A camera named by unique ID, with a cv2.VideoCapture-shaped surface."""

    def __init__(self, device, width=None, height=None, fps=None,
                 read_timeout=5.0):
        api = _bindings()
        AVF = api["AVF"]

        self._api = api
        self._read_timeout = read_timeout
        self._last_sequence = 0
        self._requested = (width, height, fps)
        self._closed = False

        unique_id = device_unique_id(device) if is_avf_device(device) else str(
            device)
        self._device = AVF.AVCaptureDevice.deviceWithUniqueID_(unique_id)
        if self._device is None:
            raise RuntimeError(
                f"no camera with unique ID {unique_id}. Attached now: "
                + (", ".join(
                    f"{d.localizedName()} [{d.uniqueID()}]"
                    for d in AVF.AVCaptureDevice.devicesWithMediaType_("vide"))
                   or "none"))

        self._session = AVF.AVCaptureSession.alloc().init()
        self._session.beginConfiguration()
        device_input, error = (
            AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(
                self._device, None))
        if device_input is None:
            self._session.commitConfiguration()
            raise RuntimeError(
                f"cannot capture {self._device.localizedName()}: {error}. "
                "On macOS check System Settings > Privacy & Security > "
                "Camera for the program running this.")
        self._session.addInput_(device_input)

        output = AVF.AVCaptureVideoDataOutput.alloc().init()
        output.setAlwaysDiscardsLateVideoFrames_(True)
        output.setVideoSettings_({"PixelFormatType": _PIXEL_FORMAT_32BGRA})
        self._sink = _sink_class().alloc().init()
        output.setSampleBufferDelegate_queue_(
            self._sink, api["dispatch_queue_create"](b"avf.capture", None))
        self._session.addOutput_(output)
        self._session.commitConfiguration()
        self._session.startRunning()

        # After startRunning, never before: committing the configuration
        # applies the session preset over any format set earlier.  If this
        # fails, stop the session on the way out -- a running session holds
        # a non-daemon capture thread that keeps the process alive forever.
        try:
            if width or height or fps:
                self._apply_format(width, height, fps)
        except Exception:
            self.release()
            raise

    # -- format ----------------------------------------------------------

    def _format_size(self, fmt):
        dims = self._api["CMVideoFormatDescriptionGetDimensions"](
            fmt.formatDescription())
        return int(dims.width), int(dims.height)

    def _apply_format(self, width, height, fps):
        """Pick the format matching the request; leave it alone if none fits.

        A size that the camera does not offer is reported rather than
        silently approximated, because a calibration is only valid for the
        exact configuration it was captured with.
        """
        best = None
        for fmt in self._device.formats():
            size = self._format_size(fmt)
            if width and height and size != (int(width), int(height)):
                continue
            ranges = list(fmt.videoSupportedFrameRateRanges())
            top = max((r.maxFrameRate() for r in ranges), default=0.0)
            if best is None or top > best[1]:
                best = (fmt, top, ranges)
        if best is None:
            print(f"[avf] {self.name()} has no {width}x{height} format; "
                  f"staying at {self.frame_size()[0]}x{self.frame_size()[1]}. "
                  "Available: "
                  + ", ".join(sorted({
                      f"{w}x{h}" for w, h in
                      (self._format_size(f) for f in self._device.formats())})))
            return

        locked, error = self._device.lockForConfiguration_(None)
        if not locked:
            print(f"[avf] cannot configure {self.name()}: {error}")
            return
        try:
            self._device.setActiveFormat_(best[0])
            if fps:
                # A format advertises RANGES, and a device rejects any
                # duration outside them outright -- the OBS Virtual Camera,
                # for one, supports 60 fps and nothing else.  Clamp into the
                # nearest supported range rather than raising mid-launch.
                rate = _closest_supported_rate(float(fps), best[2])
                duration = self._api["CMTimeMake"](1, int(round(rate)))
                try:
                    self._device.setActiveVideoMinFrameDuration_(duration)
                    self._device.setActiveVideoMaxFrameDuration_(duration)
                except Exception as exc:
                    print(f"[avf] {self.name()} refused {rate:g} fps: {exc}")
                if abs(rate - float(fps)) > 0.01:
                    print(f"[avf] {self.name()} runs this format at "
                          f"{rate:g} fps, not the requested {float(fps):g}")
        finally:
            self._device.unlockForConfiguration()

    # -- identity --------------------------------------------------------

    def name(self):
        return str(self._device.localizedName())

    def model_id(self):
        return str(self._device.modelID())

    def unique_id(self):
        return str(self._device.uniqueID())

    def frame_size(self):
        return self._format_size(self._device.activeFormat())

    def frame_rate(self):
        duration = self._device.activeVideoMinFrameDuration()
        if duration.value:
            return float(duration.timescale) / float(duration.value)
        return 0.0

    def describe(self):
        width, height = self.frame_size()
        return (f"{self.name()} [{self.model_id()}] "
                f"{width}x{height}@{self.frame_rate():g}")

    # -- cv2.VideoCapture surface ---------------------------------------

    def isOpened(self):
        return not self._closed and bool(self._session.isRunning())

    def read(self):
        """Block for the next NEW frame, so a caller cannot re-read one.

        cv2.VideoCapture.read() advances the stream; returning the cached
        frame instead would let FrameGrabber spin at CPU speed and make a
        stalled camera look like a fast one.
        """
        if self._closed:
            return False, None
        deadline = time.monotonic() + self._read_timeout
        with self._sink.condition:
            while self._sink.sequence == self._last_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self._sink.condition.wait(remaining)
            self._last_sequence = self._sink.sequence
            return True, self._sink.frame

    def release(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._session.stopRunning()
        finally:
            with self._sink.condition:
                self._sink.condition.notify_all()

    def set(self, prop, value):
        import cv2

        width, height, fps = self._requested
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            width = int(value)
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            height = int(value)
        elif prop == cv2.CAP_PROP_FPS:
            fps = float(value)
        else:
            return False  # FOURCC/BUFFERSIZE are V4L2 ideas; nothing to do
        self._requested = (width, height, fps)
        if width and height:
            self._apply_format(width, height, fps)
        return True

    def get(self, prop):
        import cv2

        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.frame_size()[0])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.frame_size()[1])
        if prop == cv2.CAP_PROP_FPS:
            return self.frame_rate()
        return 0.0

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


def open_avf_camera(device, width=None, height=None, fps=None):
    """Open ``avf:<uniqueID>``; raises with the attached list if it is gone."""

    if sys.platform != "darwin":
        raise RuntimeError("avf: devices are macOS-only")
    return AVFCamera(device, width=width, height=height, fps=fps)
