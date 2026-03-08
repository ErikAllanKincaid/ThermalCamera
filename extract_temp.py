"""
extract_temp.py -- Extract per-pixel temperature from Thermal Master P2 camera.

Frame layout at 256x386 (confirmed from APK CameraPreviewManager.java + probing):
  Rows   0-191:  YUYV image data (colorized thermal preview, white-hot palette)
  Rows 192-193:  Info rows (ASCII firmware placeholder + sync marker)
  Rows 194-385:  Raw 16-bit temperature data, little-endian uint16, one value per pixel

Temperature formula (confirmed from LibIRTemp.java):
  temp_c = uint16_value / 64.0 - 273.15

Capture method: PyAV packet demux (do NOT decode).
  Packet 1 is empty (device buffer artifact). Packet 2 is the first real frame.
"""

import av
import numpy as np
import cv2
import os
import glob
from pathlib import Path

# SAVE_DIR: resolve to the directory containing this script so saves work
# on any machine regardless of path.
SAVE_DIR = str(Path(__file__).parent)

WIDTH    = 256
HEIGHT   = 386    # full frame: 192 image + 2 info + 192 temp
IR_ROWS  = 192    # thermal image rows


def find_camera(vid='3474', pid='4281', fallback='/dev/video0'):
    """
    Find /dev/videoN for the P2 camera by USB VID:PID (3474:4281).
    Walks /sys/class/video4linux/ sysfs tree to match the USB device.
    Returns the device path, or fallback if not found.
    """
    for node in sorted(glob.glob('/sys/class/video4linux/video*')):
        real = os.path.realpath(node)
        parts = real.split('/')
        for i in range(len(parts), 0, -1):
            check = '/'.join(parts[:i])
            v_file = os.path.join(check, 'idVendor')
            p_file = os.path.join(check, 'idProduct')
            if os.path.isfile(v_file) and os.path.isfile(p_file):
                with open(v_file) as f: v = f.read().strip()
                with open(p_file) as f: p = f.read().strip()
                if v == vid and p == pid:
                    dev = '/dev/' + os.path.basename(node)
                    print(f'  Camera found: {dev}')
                    return dev
                break
    print(f'  Camera not found in sysfs, using fallback {fallback}')
    return fallback


DEVICE = find_camera()


def capture_raw_frame():
    """
    Demux one raw YUYV frame from V4L2 device via PyAV (no decode).
    Returns: uint8 array of shape (386, 512) -- rows x bytes_per_row.
    """
    container = av.open(DEVICE, format='v4l2', options={
        'video_size': f'{WIDTH}x{HEIGHT}',
        'framerate':  '25',
        # input_format ('yuyv422') omitted -- explicit value hangs on kernel 6.8.
        # FFMPEG auto-negotiates correctly on all tested kernels (6.8, 6.14).
    })
    try:
        count = 0
        raw_bytes = None
        for packet in container.demux(video=0):
            count += 1
            if count < 2:
                continue
            raw_bytes = bytes(packet)
            print(f'  Packet {count}: {len(raw_bytes)} bytes (expected {WIDTH * HEIGHT * 2})')
            break
    finally:
        try:
            container.close()
        except Exception:
            pass

    if not raw_bytes or len(raw_bytes) != WIDTH * HEIGHT * 2:
        raise RuntimeError(f'Unexpected packet size: {len(raw_bytes) if raw_bytes else 0}')

    return np.frombuffer(raw_bytes, dtype=np.uint8).reshape(HEIGHT, WIDTH * 2)


def extract_temperature(frame):
    """
    Extract calibrated per-pixel temperature from rows 194-385.

    The temp section bytes are little-endian uint16 pairs.
    .view(np.uint16) reinterprets in-place (no copy, correct on x86 LE).

    Returns:
        temp_c  -- float32 array (192, 256), degrees Celsius
        y_plane -- uint8 array  (192, 256), grayscale image (white-hot)
    """
    y_plane  = frame[:IR_ROWS, 0::2]              # Y bytes at even positions
    temp_raw = frame[194:386, :]                  # shape (192, 512) uint8
    temp_u16 = temp_raw.view(np.uint16)           # shape (192, 256) uint16 LE
    temp_c   = temp_u16.astype(np.float32) / 64.0 - 273.15
    return temp_c, y_plane


def save_temperature_overlay(temp_c, path):
    """
    False-color inferno image with min/max/mean annotation and hotspot markers.
    """
    t_min  = float(temp_c.min())
    t_max  = float(temp_c.max())
    t_mean = float(temp_c.mean())

    norm    = cv2.normalize(temp_c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)

    hot_idx  = np.unravel_index(np.argmax(temp_c), temp_c.shape)
    cold_idx = np.unravel_index(np.argmin(temp_c), temp_c.shape)

    font = cv2.FONT_HERSHEY_SIMPLEX
    s, w = 0.35, (255, 255, 255)
    cv2.putText(colored, f'Min:  {t_min:.1f}C @ ({cold_idx[1]},{cold_idx[0]})', (4, 12), font, s, w, 1)
    cv2.putText(colored, f'Max:  {t_max:.1f}C @ ({hot_idx[1]},{hot_idx[0]})',   (4, 24), font, s, w, 1)
    cv2.putText(colored, f'Mean: {t_mean:.1f}C',                                (4, 36), font, s, w, 1)
    cv2.putText(colored, '(uint16 / 64 - 273.15)',                              (4, 48), font, s, (180, 180, 180), 1)

    cv2.drawMarker(colored, (hot_idx[1],  hot_idx[0]),  (0, 255, 255), cv2.MARKER_CROSS, 10, 1)
    cv2.drawMarker(colored, (cold_idx[1], cold_idx[0]), (255, 100, 0), cv2.MARKER_CROSS, 10, 1)

    cv2.imwrite(path, colored)
    return t_min, t_max, t_mean


# -- Main ----------------------------------------------------------------------

print('=== Capturing frame (256x386) ===')
frame = capture_raw_frame()

print()
print('=== Extracting temperature data (rows 194-385) ===')
temp_c, y_plane = extract_temperature(frame)

print(f'Y-plane range:    min={int(y_plane.min())}  max={int(y_plane.max())}  mean={y_plane.mean():.1f}')
print(f'Temperature:      min={temp_c.min():.1f}C  max={temp_c.max():.1f}C  mean={temp_c.mean():.1f}C')
print(f'Plausible range:  {bool(-20 <= temp_c.min() and temp_c.max() <= 150)}')

print()
print('=== Saving outputs ===')
overlay_path = f'{SAVE_DIR}/temp_overlay.png'
gray_path    = f'{SAVE_DIR}/temp_gray.png'

t_min, t_max, t_mean = save_temperature_overlay(temp_c, overlay_path)
cv2.imwrite(gray_path, y_plane)

print(f'Saved: temp_overlay.png  (false-color inferno, annotated)')
print(f'Saved: temp_gray.png     (white-hot grayscale, Y-channel)')
print()
print(f'Scene: {t_min:.1f}C to {t_max:.1f}C  (mean {t_mean:.1f}C)')

print()
print('=== Info rows ===')
cal_row  = frame[192, :]
sync_row = frame[193, :]
print(f'Row 192 ASCII: {bytes(cal_row[:20].tolist()).decode("ascii", errors="replace")}')
print(f'Row 193 first 8 bytes: {sync_row[:8].tolist()}')
