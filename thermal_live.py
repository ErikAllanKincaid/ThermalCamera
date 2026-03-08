"""
thermal_live.py -- Live OpenCV display for Thermal Master P2 camera.

Displays a real-time false-color thermal image with:
  - Per-pixel temperature readout at mouse cursor
  - Min / max / mean overlay
  - Auto hotspot (cyan cross) and coldspot (orange cross) markers
  - FPS counter
  - Keyboard palette cycling

Frame layout (256x386 YUYV, confirmed from probe_386.py):
  Rows   0-191:  grayscale thermal preview (Y bytes = pixel brightness)
  Rows 192-193:  info/sync rows (skip)
  Rows 194-385:  raw 16-bit temperature data, little-endian uint16 per pixel

Temperature formula (from APK LibIRTemp.java):
  temp_c = uint16_value / 64.0 - 273.15

Capture strategy: open the PyAV container ONCE and loop packets continuously.
  DO NOT open/close on each frame -- the camera enters a bad state if you
  reopen after closing. Keep one container alive for the full session.

Controls:
  p       -- cycle through false-color palettes
  s       -- save current frame to disk (temp_snapshot.png)
  q / ESC -- quit
"""

import av
import numpy as np
import cv2
import time
import os
import glob
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WIDTH    = 256
HEIGHT   = 386       # full dual-frame: 192 image + 2 info + 192 temp
IR_ROWS  = 192       # rows of actual thermal image
SCALE    = 3         # display scale factor: 256x192 -> 768x576

# SAVE_DIR: always save next to this script, regardless of where it is run from.
# Path(__file__).parent resolves to the directory containing thermal_live.py.
SAVE_DIR = str(Path(__file__).parent)


def find_camera(vid='3474', pid='4281', fallback='/dev/video0'):
    """
    Locate the /dev/videoN device for the Thermal Master P2 by USB VID:PID.

    Why: on machines with a built-in webcam (e.g. a laptop), the P2 may not
    be /dev/video0. This function walks /sys/class/video4linux/, resolves each
    symlink to its real sysfs path, then climbs the directory tree looking for
    idVendor/idProduct files that identify the USB device.

    VID 3474 / PID 4281 = Raysentek Co.,Ltd (Thermal Master P2 / InfiRay P2L).

    Returns the /dev/videoN path if found, or fallback if not.
    """
    for node in sorted(glob.glob('/sys/class/video4linux/video*')):
        real = os.path.realpath(node)           # follow symlink to actual sysfs path
        parts = real.split('/')
        # Walk up from the deepest directory toward the root, stopping when we
        # find a directory that has both idVendor and idProduct (USB device node).
        for i in range(len(parts), 0, -1):
            check = '/'.join(parts[:i])
            v_file = os.path.join(check, 'idVendor')
            p_file = os.path.join(check, 'idProduct')
            if os.path.isfile(v_file) and os.path.isfile(p_file):
                with open(v_file) as f: v = f.read().strip()
                with open(p_file) as f: p = f.read().strip()
                if v == vid and p == pid:
                    dev = '/dev/' + os.path.basename(node)
                    print(f'  Found camera at {dev}  (VID:{v} PID:{p})')
                    return dev
                break   # found VID/PID files but no match; stop climbing for this node
    print(f'  Camera VID:{vid}/PID:{pid} not found in sysfs, using fallback {fallback}')
    return fallback


DEVICE = find_camera()

# ---------------------------------------------------------------------------
# Palette definitions
# Each entry: (display name, OpenCV colormap constant or None for grayscale)
# None means render as grayscale (white-hot or black-hot).
# ---------------------------------------------------------------------------

# Why these palettes?
#   INFERNO  -- perceptually uniform, dark-to-bright, good for subtle detail
#   HOT      -- classic thermal look (black -> red -> yellow -> white)
#   JET      -- familiar rainbow; easy to read temperature gradient at a glance
#   BONE     -- cool blue-gray; clinical / medical feel
#   white-hot / black-hot -- grayscale, no false color; traditional IR styles

PALETTES = [
    ('Inferno',    cv2.COLORMAP_INFERNO),
    ('Hot',        cv2.COLORMAP_HOT),
    ('Jet',        cv2.COLORMAP_JET),
    ('Bone',       cv2.COLORMAP_BONE),
    ('White-hot',  None),   # inverted grayscale: cold=black, hot=white
    ('Black-hot',  'invert'),  # grayscale: cold=white, hot=black
]

palette_idx = 0   # current palette index, changed by 'p' keypress

# ---------------------------------------------------------------------------
# Mouse state -- updated by the OpenCV mouse callback
# ---------------------------------------------------------------------------

# Why a global dict instead of two separate globals?
# Python's scoping rules mean you can mutate a dict from inside a nested
# function without the 'global' keyword, making the callback simpler.
mouse = {'x': 0, 'y': 0}


def on_mouse(event, x, y, flags, param):
    """
    OpenCV mouse callback. Called by cv2 whenever the mouse moves over the window.
    We only care about the cursor position (x, y) -- we ignore click events here.
    x, y are in DISPLAY pixel coordinates (scaled up by SCALE).
    """
    mouse['x'] = x
    mouse['y'] = y


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_frame(temp_c, y_plane):
    """
    Convert a float32 temperature array into an annotated BGR display image.

    Arguments:
        temp_c  -- float32 array (192, 256) in degrees Celsius
        y_plane -- uint8 array  (192, 256) grayscale Y-channel (for white/black-hot)

    Returns: BGR uint8 display image at 768x576 (or SCALE * 256 x SCALE * 192)

    Steps:
      1. Normalize temp_c to 0-255 range (or use y_plane for grayscale modes)
      2. Apply colormap (or grayscale pass-through)
      3. Scale up with INTER_NEAREST (keeps crisp pixel edges; INTER_LINEAR would blur)
      4. Annotate: stats overlay, cursor crosshair + temp, hotspot/coldspot markers
    """
    name, cmap = PALETTES[palette_idx]

    # -- Step 1+2: build the colored base image --
    if cmap is None:
        # White-hot: cold pixels are dark, hot pixels are bright.
        # The Y-plane is already a brightness image from the camera firmware,
        # but it is a preview render, not a linear temp map. For true white-hot
        # based on actual temps, normalize temp_c.
        norm = cv2.normalize(temp_c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Convert grayscale to BGR so we can draw colored annotations on top
        base = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

    elif cmap == 'invert':
        # Black-hot: invert the white-hot normalization
        norm = cv2.normalize(temp_c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm = 255 - norm   # flip: hot pixels become dark, cold pixels become bright
        base = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

    else:
        # Standard false-color palette via OpenCV colormap
        # cv2.normalize maps the float32 temp range to uint8 0-255, then
        # applyColorMap maps each 0-255 value to a BGR color from the palette LUT.
        norm = cv2.normalize(temp_c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        base = cv2.applyColorMap(norm, cmap)

    # -- Step 3: scale up for display --
    # INTER_NEAREST: each source pixel becomes a SCALE x SCALE block with no blending.
    # This keeps the thermal pixel grid sharp and visible, which is useful for
    # understanding the actual sensor resolution (256x192 is not many pixels).
    display = cv2.resize(base, (WIDTH * SCALE, IR_ROWS * SCALE),
                         interpolation=cv2.INTER_NEAREST)

    # -- Step 4: annotations --

    # Map mouse display coordinates back to sensor pixel coordinates.
    # Clamp to [0, WIDTH-1] and [0, IR_ROWS-1] so we never index out of bounds.
    sx = min(max(mouse['x'] // SCALE, 0), WIDTH - 1)
    sy = min(max(mouse['y'] // SCALE, 0), IR_ROWS - 1)
    cursor_temp = float(temp_c[sy, sx])

    # Statistics
    t_min  = float(temp_c.min())
    t_max  = float(temp_c.max())
    t_mean = float(temp_c.mean())

    # Hotspot and coldspot locations (argmax/argmin on the flat array, then convert
    # back to (row, col) with unravel_index)
    hot_idx  = np.unravel_index(np.argmax(temp_c), temp_c.shape)  # (row, col)
    cold_idx = np.unravel_index(np.argmin(temp_c), temp_c.shape)

    # Scale hotspot/coldspot coords to display space (multiply by SCALE,
    # center in the middle of the scaled block by adding SCALE//2)
    hot_pt  = (hot_idx[1]  * SCALE + SCALE // 2, hot_idx[0]  * SCALE + SCALE // 2)
    cold_pt = (cold_idx[1] * SCALE + SCALE // 2, cold_idx[0] * SCALE + SCALE // 2)

    # Text overlay: stats bar at top-left
    font = cv2.FONT_HERSHEY_SIMPLEX
    WHITE  = (255, 255, 255)
    CYAN   = (255, 255, 0)    # BGR: yellow-green, reads well on inferno
    GRAY   = (180, 180, 180)
    ORANGE = (0, 140, 255)    # BGR for orange

    # Semi-transparent black bar behind text improves readability on bright palettes.
    # cv2 does not have native transparency, so we draw a filled rectangle then
    # blend it with the display image using addWeighted.
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (WIDTH * SCALE, 72), (0, 0, 0), -1)
    # alpha=0.5: 50% transparency. overlay has the black bar; display has original.
    cv2.addWeighted(overlay, 0.45, display, 0.55, 0, display)

    cv2.putText(display, f'Min: {t_min:.1f}C   Max: {t_max:.1f}C   Mean: {t_mean:.1f}C',
                (6, 16), font, 0.45, WHITE, 1, cv2.LINE_AA)
    cv2.putText(display, f'Cursor: {cursor_temp:.1f}C  @ sensor ({sx}, {sy})',
                (6, 34), font, 0.45, CYAN, 1, cv2.LINE_AA)
    cv2.putText(display, f'Palette: {name}   [p] cycle   [s] save   [q] quit',
                (6, 52), font, 0.38, GRAY, 1, cv2.LINE_AA)

    # Cursor crosshair at mouse position in display space
    cx, cy = mouse['x'], mouse['y']
    cv2.drawMarker(display, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 24, 1, cv2.LINE_AA)

    # Hotspot marker (cyan) and coldspot marker (orange)
    cv2.drawMarker(display, hot_pt,  (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 1, cv2.LINE_AA)
    cv2.drawMarker(display, cold_pt, ORANGE,        cv2.MARKER_TILTED_CROSS, 16, 1, cv2.LINE_AA)

    # Label the hotspot and coldspot with their temperatures.
    # Offset label slightly so it does not sit on top of the marker.
    cv2.putText(display, f'{t_max:.1f}C',
                (hot_pt[0] + 6,  hot_pt[1] - 6),  font, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display, f'{t_min:.1f}C',
                (cold_pt[0] + 6, cold_pt[1] - 6), font, 0.38, ORANGE,        1, cv2.LINE_AA)

    return display


# ---------------------------------------------------------------------------
# Main capture + display loop
# ---------------------------------------------------------------------------

print('Opening camera...')

# Open the PyAV container ONCE. We will iterate its packets in the loop below.
# Why not cv2.VideoCapture? It times out with CONVERT_RGB=0 on this camera.
# PyAV raw demux (no decode) is the only method confirmed to work reliably.
container = av.open(DEVICE, format='v4l2', options={
    'video_size': f'{WIDTH}x{HEIGHT}',
    'framerate':  '25',
    # input_format ('yuyv422') omitted -- explicit value hangs on kernel 6.8.
    # FFMPEG auto-negotiates correctly on all tested kernels (6.8, 6.14).
})

# Create the display window and register the mouse callback.
# WINDOW_NORMAL allows the user to resize the window manually.
WIN = 'Thermal P2'
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, WIDTH * SCALE, IR_ROWS * SCALE)
cv2.setMouseCallback(WIN, on_mouse)

print(f'Streaming at {WIDTH}x{IR_ROWS} (display: {WIDTH*SCALE}x{IR_ROWS*SCALE})')
print('Controls: [p] palette  [s] save  [q/ESC] quit')

# FPS tracking: measure wall-clock time per frame
fps        = 0.0
frame_count = 0
t_start    = time.time()

try:
    count = 0  # packet counter -- we skip packet 1 (empty device buffer artifact)

    for packet in container.demux(video=0):
        count += 1
        if count < 2:
            # Packet 1 is always empty (V4L2 device init artifact).
            # Skip it silently.
            continue

        # Read raw packet bytes -- these are the YUYV bytes from the camera.
        # We do NOT call packet.decode() because the YUYV format is not a standard
        # compressed video; decoding would fail or produce wrong results.
        raw_bytes = bytes(packet)

        # Sanity check: the packet must be exactly WIDTH * HEIGHT * 2 bytes.
        # YUYV = 2 bytes per pixel. If the size is wrong, the camera may be
        # in a bad state or the resolution negotiation failed.
        if len(raw_bytes) != WIDTH * HEIGHT * 2:
            print(f'  Unexpected packet size: {len(raw_bytes)}, skipping')
            continue

        # -- Parse the raw bytes into a numpy array --
        # reshape to (386 rows, 512 bytes per row)
        # Each row has 256 pixels * 2 bytes = 512 bytes in YUYV format.
        frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(HEIGHT, WIDTH * 2)

        # -- Extract the Y (luminance) plane for optional grayscale display --
        # YUYV layout per macropixel: Y0 U0 Y1 V0
        # Y bytes sit at even byte positions (0, 2, 4, ...) in each row.
        # Slicing [:, 0::2] takes every other byte starting at 0 -> Y channel.
        y_plane = frame[:IR_ROWS, 0::2]   # shape (192, 256) uint8

        # -- Extract the temperature data from rows 194-385 --
        # These rows contain raw 16-bit sensor values packed as YUYV bytes,
        # but the YUYV interpretation does not apply here: every pair of bytes
        # IS a uint16 little-endian temperature value (not Y+U).
        # .view(np.uint16) reinterprets the uint8 buffer in-place as uint16 LE.
        # This works on x86 (little-endian) with no byte swapping.
        temp_raw = frame[194:386, :]              # shape (192, 512) uint8
        temp_u16 = temp_raw.view(np.uint16)       # shape (192, 256) uint16 LE -- no copy
        temp_c   = temp_u16.astype(np.float32) / 64.0 - 273.15  # degrees Celsius

        # -- Render and display --
        display = render_frame(temp_c, y_plane)

        # FPS: count frames, compute rate once per second
        frame_count += 1
        elapsed = time.time() - t_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            t_start = time.time()

        # Draw FPS in top-right corner
        fps_text = f'{fps:.1f} fps'
        (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(display, fps_text,
                    (WIDTH * SCALE - tw - 6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(WIN, display)

        # -- Handle keypresses --
        # cv2.waitKey(1): process window events for 1ms.
        # Without this call, the window freezes and never updates.
        # The return value is the ASCII code of any key pressed, or -1 if none.
        # '& 0xFF' masks to the low byte (needed on some Linux/OpenCV builds).
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:    # q or ESC
            print('Quit.')
            break

        elif key == ord('p'):
            # Cycle to next palette
            palette_idx = (palette_idx + 1) % len(PALETTES)
            print(f'Palette: {PALETTES[palette_idx][0]}')

        elif key == ord('s'):
            # Save current rendered frame to disk
            snap_path = f'{SAVE_DIR}/temp_snapshot.png'
            cv2.imwrite(snap_path, display)
            # Also save the raw temperature array as a numpy file for later analysis
            np_path = f'{SAVE_DIR}/temp_snapshot.npy'
            np.save(np_path, temp_c)
            print(f'Saved: temp_snapshot.png  temp_snapshot.npy')
            print(f'  Scene: {temp_c.min():.1f}C to {temp_c.max():.1f}C  mean {temp_c.mean():.1f}C')

finally:
    # Always close the container and destroy the window on exit,
    # whether we quit normally or an exception was raised.
    try:
        container.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
    print('Camera closed.')
