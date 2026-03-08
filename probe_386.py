"""
probe_386.py -- Open camera at 256x386, extract raw 16-bit temp from bottom half.

The 386-row frame layout (confirmed from APK CameraPreviewManager.java):
  Rows   0-191:  YUYV image data (the colorized thermal preview)
  Rows 192-193:  Info/calibration rows (ASCII metadata + sync marker)
  Rows 194-385:  Raw 16-bit temperature data, little-endian, one uint16 per pixel

Temperature formula (confirmed from LibIRTemp.java line 317):
  temp_c = (uint16_value / 64.0) - 273.15

Why .view(np.uint16) works:
  The temp section bytes are already little-endian uint16 pairs.
  numpy's .view() reinterprets the raw bytes without copying -- fast and correct
  on x86 (which is also little-endian). No manual (high << 8 | low) needed.
"""

import av
import numpy as np
import cv2

DEVICE  = '/dev/video0'
WIDTH   = 256
HEIGHT  = 386   # full frame: 192 image + 2 info + 192 temp
IR_ROWS = 192   # thermal image rows
SAVE_DIR = '/home/erik/code/claude/ThermalCamera'

# --- Capture one raw frame via PyAV ---
# PyAV reads raw V4L2 packets without decoding -- preserves exact YUYV bytes.
# We skip packet 1 (empty buffer artifact from device init).

container = av.open(DEVICE, format='v4l2', options={
    'input_format': 'yuyv422',
    'video_size':   f'{WIDTH}x{HEIGHT}',
    'framerate':    '25',
})

count = 0
raw_bytes = None
for packet in container.demux(video=0):
    count += 1
    if count < 2:
        continue
    raw_bytes = bytes(packet)
    expected  = WIDTH * HEIGHT * 2   # YUYV = 2 bytes per pixel
    print(f'Packet {count}: {len(raw_bytes)} bytes  (expected {expected})')
    break
container.close()

if not raw_bytes or len(raw_bytes) != WIDTH * HEIGHT * 2:
    raise RuntimeError(f'Unexpected packet size: {len(raw_bytes) if raw_bytes else 0}')

# --- Reshape into rows ---
# YUYV at 256x386: each row is 256 pixels * 2 bytes = 512 bytes.
# Reshape gives us (386, 512) -- rows x bytes_per_row.
frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(HEIGHT, WIDTH * 2)

print(f'Frame shape: {frame.shape}  dtype: {frame.dtype}')

# --- Extract image half (rows 0-191): Y bytes for visual check ---
# YUYV layout: Y0 U0 Y1 V0 ...  Y bytes at even positions (0, 2, 4, ...)
y_plane = frame[:IR_ROWS, 0::2]   # shape (192, 256)
print(f'Image Y-plane: min={y_plane.min()}  max={y_plane.max()}  mean={y_plane.mean():.1f}')

# --- Inspect info rows 192-193 ---
cal_row  = frame[192, :]
sync_row = frame[193, :]
print(f'Info row 192 as ASCII: {bytes(cal_row[:20].tolist()).decode("ascii", errors="replace")}')
print(f'Sync row 193 first 8 bytes: {sync_row[:8].tolist()}')

# --- Extract temperature half (rows 194-385) ---
# These 192 rows contain raw 16-bit little-endian sensor values.
# Each row is 256 pixels * 2 bytes = 512 bytes = 256 uint16 values.
# .view(np.uint16) reinterprets the uint8 bytes as uint16 LE in-place.
temp_raw = frame[194:386, :]              # shape (192, 512) uint8
temp_u16 = temp_raw.view(np.uint16)       # shape (192, 256) uint16 LE -- no copy
temp_c   = temp_u16.astype(np.float32) / 64.0 - 273.15   # degrees Celsius

print()
print(f'Raw uint16 range: min={int(temp_u16.min())}  max={int(temp_u16.max())}')
print(f'Temperature:      min={temp_c.min():.1f}C  max={temp_c.max():.1f}C  mean={temp_c.mean():.1f}C')
print(f'Plausible (-20 to 150C): {bool(-20 <= temp_c.min() and temp_c.max() <= 150)}')

# --- Save false-color temperature image ---
# Normalize temp_c to 0-255, apply inferno colormap, annotate with stats.
norm    = cv2.normalize(temp_c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(colored, f'Min:  {temp_c.min():.1f}C',  (4, 12), font, 0.35, (255,255,255), 1)
cv2.putText(colored, f'Max:  {temp_c.max():.1f}C',  (4, 24), font, 0.35, (255,255,255), 1)
cv2.putText(colored, f'Mean: {temp_c.mean():.1f}C', (4, 36), font, 0.35, (255,255,255), 1)
cv2.putText(colored, '(RAW uint16 / 64 - 273.15)',  (4, 48), font, 0.35, (180,180,180), 1)

out_path = f'{SAVE_DIR}/probe_386_temp.png'
cv2.imwrite(out_path, colored)
print(f'\nSaved: probe_386_temp.png')

# Also save the Y-plane grayscale for comparison
cv2.imwrite(f'{SAVE_DIR}/probe_386_gray.png', y_plane)
print(f'Saved: probe_386_gray.png')
