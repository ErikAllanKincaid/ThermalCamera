"""
thermal_server.py -- Flask + MJPEG live thermal camera server for Thermal Master P2.

Architecture overview:
  - ONE capture thread: opens the PyAV container, loops packets, renders frames,
    stores the latest JPEG + temperature array in shared state.
  - ONE Flask thread: serves HTTP. Reads from shared state to stream frames
    and answer API requests. Never touches the camera directly.
  - Shared state protected by threading.Lock() to prevent race conditions.

Why Flask + MJPEG instead of cv2.imshow():
  - cv2.imshow() requires a local display server (X11/Wayland). Headless servers
    and SSH sessions have no display. PyPI opencv-python wheels are also compiled
    for specific Linux distributions and GUI backends often fail silently.
  - MJPEG over HTTP works in any browser on any device on the same network.
    No display server needed on the server side at all.

MJPEG (Motion JPEG) streaming:
  - HTTP content type: multipart/x-mixed-replace; boundary=frame
  - Each frame is a JPEG image preceded by a MIME boundary marker.
  - The browser's <img> tag receives an infinite stream of JPEG replacements,
    creating the appearance of live video. This is the same technique used by
    IP cameras and many embedded systems.

Routes:
  GET  /                    -- HTML UI (browser opens this)
  GET  /video_feed          -- MJPEG stream (connected to <img src="/video_feed">)
  GET  /api/temp_at?x=N&y=N -- temperature at display pixel (x, y) in Celsius
  GET  /api/stats           -- min / max / mean / fps as JSON
  POST /api/palette         -- cycle to next false-color palette
  POST /api/save            -- save current frame as PNG + numpy array to disk

Temperature formula (confirmed from APK LibIRTemp.java):
  temp_c = uint16_value / 64.0 - 273.15
Frame layout:
  Rows   0-191: YUYV grayscale thermal image
  Rows 192-193: info/sync rows
  Rows 194-385: raw 16-bit temperature data (little-endian uint16 per pixel)
"""

import av
import numpy as np
import cv2
import time
import threading
import os
import glob
from pathlib import Path
from flask import Flask, Response, jsonify, request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WIDTH    = 256
HEIGHT   = 386        # full dual-frame: 192 image + 2 info + 192 temp
IR_ROWS  = 192        # rows of actual thermal image
SCALE    = 3          # render at 3x: 256x192 -> 768x576
PORT     = 5800

# SAVE_DIR: resolve to the directory containing this script.
# Path(__file__).parent works on any machine regardless of where the script lives.
SAVE_DIR = str(Path(__file__).parent)

# ---------------------------------------------------------------------------
# Camera auto-detection
# ---------------------------------------------------------------------------

def find_camera(vid='3474', pid='4281', fallback='/dev/video0'):
    """
    Locate /dev/videoN for the Thermal Master P2 by USB VID:PID (3474:4281).

    Scans /sys/class/video4linux/ and climbs the sysfs directory tree from each
    video node looking for idVendor / idProduct files that identify the USB device.
    This works on any Linux machine regardless of how many other cameras are attached.

    Returns the /dev/videoN device path, or fallback if not found.
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
                    print(f'  Found camera: {dev}  (VID:{v} PID:{p})')
                    return dev
                break
    print(f'  Camera VID:{vid}/PID:{pid} not found, using fallback {fallback}')
    return fallback


DEVICE = find_camera()

# ---------------------------------------------------------------------------
# False-color palettes
# ---------------------------------------------------------------------------

# Each entry: (display name, OpenCV colormap constant or special string)
# None    = white-hot (normalized grayscale, cold=dark, hot=bright)
# 'invert'= black-hot (inverted grayscale, cold=bright, hot=dark)
PALETTES = [
    ('Inferno',   cv2.COLORMAP_INFERNO),
    ('Hot',       cv2.COLORMAP_HOT),
    ('Jet',       cv2.COLORMAP_JET),
    ('Bone',      cv2.COLORMAP_BONE),
    ('White-hot', None),
    ('Black-hot', 'invert'),
]

# ---------------------------------------------------------------------------
# Shared state between capture thread and Flask thread
# ---------------------------------------------------------------------------

# Why a dict with a lock instead of global variables?
# Both the capture thread and Flask request threads read/write this data
# concurrently. Without a lock, one thread could read a partially-written
# value. threading.Lock() ensures only one thread accesses state at a time.

state = {
    'jpeg':      None,         # latest rendered frame as JPEG bytes
    'temp_c':    None,         # float32 (192, 256) temperature array -- always Celsius internally
    'stats':     {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'fps': 0.0},
    'lock':      threading.Lock(),
    'palette':   0,            # index into PALETTES list
    'unit':      'C',          # display unit: 'C' or 'F' -- conversion happens at output boundary
    'running':   True,         # set to False to stop the capture thread
    'frame_seq': 0,            # increments on every good frame; client uses this to detect stale frames
}


def c_to_unit(c, unit):
    """
    Convert a Celsius value to the current display unit.

    We always store temperatures internally in Celsius (the native sensor unit).
    Conversion to Fahrenheit happens only at output: frame annotation, API responses.
    This way there is one source of truth and no accumulated rounding errors.

    F = C * 9/5 + 32  -- exact formula, no approximation.
    """
    return c * 9.0 / 5.0 + 32.0 if unit == 'F' else c

# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_jpeg(temp_c, palette_idx, unit):
    """
    Convert a float32 temperature array to an annotated JPEG image (bytes).

    Pure function: all state (palette_idx, unit) is passed as arguments.
    The caller reads these from state under the lock before calling, so this
    function never needs to acquire the lock itself.

    Steps:
      1. Normalize temp_c to 0-255 (cv2.normalize handles the float range)
      2. Apply false-color palette (or grayscale for white/black-hot)
      3. Scale up 3x with INTER_NEAREST (sharp pixel edges, no blurring)
      4. Annotate: min/max/mean text, hotspot marker, coldspot marker
      5. Encode as JPEG (quality 85 -- good balance of quality vs. bandwidth)

    Returns: bytes (JPEG-encoded image)
    """
    name, cmap = PALETTES[palette_idx]

    # -- Normalize and apply palette --
    if cmap is None:
        # White-hot: cold=dark, hot=bright
        norm = cv2.normalize(temp_c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        base = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    elif cmap == 'invert':
        # Black-hot: invert the brightness
        norm = cv2.normalize(temp_c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        base = cv2.cvtColor(255 - norm, cv2.COLOR_GRAY2BGR)
    else:
        norm = cv2.normalize(temp_c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        base = cv2.applyColorMap(norm, cmap)

    # -- Scale up --
    # INTER_NEAREST: each sensor pixel becomes a sharp SCALE x SCALE block.
    # Never use INTER_LINEAR here -- blurring thermal pixels is misleading.
    display = cv2.resize(base, (WIDTH * SCALE, IR_ROWS * SCALE),
                         interpolation=cv2.INTER_NEAREST)

    # -- Statistics (convert to display unit at annotation time) --
    t_min  = c_to_unit(float(temp_c.min()),  unit)
    t_max  = c_to_unit(float(temp_c.max()),  unit)
    t_mean = c_to_unit(float(temp_c.mean()), unit)

    hot_idx  = np.unravel_index(np.argmax(temp_c), temp_c.shape)
    cold_idx = np.unravel_index(np.argmin(temp_c), temp_c.shape)

    # Map sensor pixel (row, col) to display pixel (cx, cy)
    # Center in the scaled block by adding SCALE // 2
    hot_pt  = (hot_idx[1]  * SCALE + SCALE // 2, hot_idx[0]  * SCALE + SCALE // 2)
    cold_pt = (cold_idx[1] * SCALE + SCALE // 2, cold_idx[0] * SCALE + SCALE // 2)

    # -- Semi-transparent annotation bar --
    # Draw filled rectangle on a copy, then blend back at partial opacity.
    # cv2 has no native alpha drawing; addWeighted is the standard workaround.
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (WIDTH * SCALE, 54), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    WHITE = (255, 255, 255)
    GRAY  = (170, 170, 170)
    CYAN  = (255, 255, 0)    # BGR
    ORG   = (0,  140, 255)   # orange BGR

    cv2.putText(display, f'Min: {t_min:.1f}{unit}   Max: {t_max:.1f}{unit}   Mean: {t_mean:.1f}{unit}',
                (6, 16), font, 0.45, WHITE, 1, cv2.LINE_AA)
    cv2.putText(display, f'Palette: {name}',
                (6, 34), font, 0.38, GRAY, 1, cv2.LINE_AA)
    cv2.putText(display, f'Hot: {t_max:.1f}{unit}',
                (6, 50), font, 0.35, CYAN, 1, cv2.LINE_AA)

    # Hotspot (cyan tilted cross) and coldspot (orange tilted cross)
    cv2.drawMarker(display, hot_pt,  CYAN, cv2.MARKER_TILTED_CROSS, 16, 1, cv2.LINE_AA)
    cv2.drawMarker(display, cold_pt, ORG,  cv2.MARKER_TILTED_CROSS, 16, 1, cv2.LINE_AA)

    # -- JPEG encode --
    # cv2.imencode returns (success_bool, numpy_array_of_bytes).
    # [1] gets the byte array; .tobytes() converts to Python bytes.
    # Quality 85: imperceptible loss at ~5-10x smaller than quality 100.
    ok, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


# ---------------------------------------------------------------------------
# Capture thread
# ---------------------------------------------------------------------------

def open_camera():
    """
    Open the PyAV container for the thermal camera.
    Separated from capture_loop so auto-reconnect can call it again cleanly.
    """
    return av.open(DEVICE, format='v4l2', options={
        'video_size': f'{WIDTH}x{HEIGHT}',
        'framerate':  '25',
        # input_format ('yuyv422') intentionally omitted.
        # Specifying it explicitly causes an indefinite hang on kernel 6.8 (Dell XPS).
        # FFMPEG auto-negotiates the format correctly on all tested kernels (6.8, 6.14).
    })


def capture_loop():
    """
    Runs in a background thread. Captures frames and writes them to shared state.

    Auto-reconnect design:
      Earlier versions opened the container once and looped forever. This caused
      a persistent glitch: when the USB stream lost sync (misaligned packet
      boundaries), every subsequent frame was corrupted until the server restarted.

      Now the loop detects bad frames by:
        1. Checking the sync row (row 193 must be [255, 0, 255, 0, ...]).
           This row is a firmware-defined marker. If it is wrong, the frame
           buffer is misaligned -- temperature rows are not where we expect.
        2. Checking temperature plausibility (must be within -50C to 200C).
           Values outside this range indicate garbage data in the temp rows.

      If N consecutive bad frames are detected, the container is closed and
      reopened. This resets the USB stream and camera internal state, exactly
      what a manual server restart used to accomplish.

    Thread safety:
      Writes to state['jpeg'], state['temp_c'], state['stats'] under the lock.
      Flask threads only READ these values under the same lock.
    """
    print(f'Capture thread started. Device: {DEVICE}')

    fps_count      = 0
    fps_timer      = time.time()
    fps            = 0.0
    BAD_FRAME_LIMIT = 8   # consecutive bad frames before reconnect

    while state['running']:
        container = None
        try:
            print('Opening camera container...')
            container = open_camera()
            print('Camera container open. Streaming.')

            count       = 0
            bad_frames  = 0

            for packet in container.demux(video=0):
                if not state['running']:
                    return

                count += 1
                if count < 2:
                    # First packet is always empty (V4L2 device init artifact).
                    continue

                raw_bytes = bytes(packet)
                if len(raw_bytes) != WIDTH * HEIGHT * 2:
                    bad_frames += 1
                    if bad_frames >= BAD_FRAME_LIMIT:
                        raise RuntimeError(f'Wrong packet size {len(raw_bytes)} for {BAD_FRAME_LIMIT} consecutive frames')
                    continue

                # Reshape YUYV bytes to (386 rows, 512 bytes/row)
                frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(HEIGHT, WIDTH * 2)

                # --- Validate sync row (row 193) ---
                # The camera firmware writes a fixed [255, 0, 255, 0, ...] pattern
                # into row 193 as a frame sync marker. If this is wrong, the packet
                # is misaligned and we must not trust the temperature data.
                sync = frame[193, :]
                if not (sync[0] == 255 and sync[1] == 0 and sync[2] == 255 and sync[3] == 0):
                    bad_frames += 1
                    print(f'  Bad sync row: {sync[:8].tolist()} (frame {count})')
                    if bad_frames >= BAD_FRAME_LIMIT:
                        raise RuntimeError('Lost frame sync -- reconnecting')
                    continue

                # Extract temperature data from rows 194-385 (uint16 LE).
                temp_raw = frame[194:386, :]
                temp_u16 = temp_raw.view(np.uint16)
                temp_c   = temp_u16.astype(np.float32) / 64.0 - 273.15

                # --- Validate temperature plausibility ---
                # Values outside [-50, 200]C indicate corrupted temperature data.
                # The camera spec is -20 to 150C; we use wider bounds to tolerate
                # edge cases without false positives.
                t_min = float(temp_c.min())
                t_max = float(temp_c.max())
                if t_min < -50.0 or t_max > 200.0:
                    bad_frames += 1
                    print(f'  Implausible temps {t_min:.1f}C to {t_max:.1f}C (frame {count})')
                    if bad_frames >= BAD_FRAME_LIMIT:
                        raise RuntimeError(f'Persistent bad temperatures -- reconnecting')
                    continue

                # Good frame -- reset bad frame counter
                bad_frames = 0

                # Read display settings under lock before rendering.
                # render_jpeg is a pure function; we snapshot the settings here
                # so the render call is lock-free (CV ops can be slow).
                with state['lock']:
                    palette_idx = state['palette']
                    unit        = state['unit']

                jpeg = render_jpeg(temp_c, palette_idx, unit)

                # FPS tracking
                fps_count += 1
                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    fps       = fps_count / elapsed
                    fps_count = 0
                    fps_timer = time.time()

                with state['lock']:
                    state['jpeg']      = jpeg
                    state['temp_c']    = temp_c
                    state['frame_seq'] = state['frame_seq'] + 1
                    state['stats']     = {
                        'min':  round(t_min,  1),
                        'max':  round(t_max,  1),
                        'mean': round(float(temp_c.mean()), 1),
                        'fps':  round(fps, 1),
                    }

        except Exception as e:
            print(f'Capture error: {e}')
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
                print('Camera container closed.')

        if state['running']:
            print('Reconnecting in 2s...')
            time.sleep(2)

    print('Capture thread stopped.')


# ---------------------------------------------------------------------------
# HTML page (served at /)
# ---------------------------------------------------------------------------

# The HTML is embedded directly in the Python file.
# For a small single-file app this is fine -- no templates directory needed.
# The image tag src="/video_feed" connects to the MJPEG stream endpoint.
# JS mousemove sends cursor coordinates to /api/temp_at and displays the result.

HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Thermal P2 Live</title>
  <style>
    body        { background:#111; color:#eee; font-family:monospace; margin:0; padding:12px; }
    h1          { font-size:1em; margin:0 0 8px; color:#aaa; }
    #wrap       { position:relative; display:inline-block; }
    #feed       { display:block; width:768px; height:576px; cursor:crosshair; background:#000; }
    #cursor_box {
      position:absolute; top:4px; right:4px;
      background:rgba(0,0,0,0.65); color:#0ff;
      padding:4px 10px; font-size:1em; pointer-events:none;
      border:1px solid #333;
    }
    #controls   { margin-top:10px; }
    button {
      background:#222; color:#ccc; border:1px solid #444;
      padding:6px 16px; cursor:pointer; font-family:monospace; margin-right:6px;
    }
    button:hover { background:#333; }
    #stats      { margin-top:8px; font-size:0.82em; color:#888; }
    #msg        { font-size:0.8em; color:#555; margin-top:4px; min-height:1em; }
  </style>
</head>
<body>
  <h1>Thermal Master P2 // Live</h1>
  <div id="wrap">
    <canvas id="feed" width="768" height="576"></canvas>
    <div id="cursor_box">--</div>
  </div>
  <div id="controls">
    <button onclick="cyclePalette()" title="keyboard: p">Palette [p]</button>
    <button onclick="toggleUnit()"  title="keyboard: u" id="unitBtn">C / F [u]</button>
    <button onclick="saveSnapshot()" title="keyboard: s">Save [s]</button>
  </div>
  <div id="stats">connecting...</div>
  <div id="msg"></div>

  <script>
    const feed      = document.getElementById('feed');
    const ctx       = feed.getContext('2d');  // canvas 2D drawing context
    const cursorBox = document.getElementById('cursor_box');
    const statsDiv  = document.getElementById('stats');
    const msgDiv    = document.getElementById('msg');

    // --- JS frame polling (replaces MJPEG) ---
    //
    // Why double-buffering?
    //   If we just set feed.src = '/api/frame?t=...' directly, the browser
    //   clears the displayed image immediately, causing a visible flicker
    //   between frames. Double-buffering loads the next frame into a hidden
    //   Image object first; only when it is fully decoded do we swap it onto
    //   the visible <img>. The result is tear-free, glitch-free display.
    //
    // Why onload -> request next (sequential)?
    //   We request the next frame only after the current one is displayed.
    //   This self-throttles to the server's render rate and prevents a
    //   backlog of in-flight requests from building up.
    //
    // Why ?t=Date.now() on the URL?
    //   Prevents the browser from serving a cached response. Each URL is unique.
    //   The server also sends Cache-Control: no-store for the same reason.

    // --- Canvas double-buffer frame loop ---
    //
    // Why canvas instead of <img src="...">?
    //   Rapidly swapping img.src triggers a Chromium compositing bug where the
    //   previous frame's GPU layer persists while the new one loads, causing
    //   visible "ghosting" (two frames layered on screen simultaneously).
    //   ctx.drawImage() writes directly to the canvas pixel buffer in one atomic
    //   operation -- the old frame is overwritten completely. Layering is impossible.
    //
    // Why fetch() + createObjectURL instead of offscreen.src = url?
    //   fetch() gives us a Blob (raw bytes). createObjectURL() wraps it in a
    //   local blob:// URL. Setting offscreen.src to a blob:// URL loads from
    //   memory -- no second HTTP request, no caching ambiguity.
    //   We revoke the old blob URL after drawing to free the memory immediately.

    let offscreen    = new Image();
    let currentBlob  = null;
    let lastFrameSeq = -1;

    function fetchFrame() {
      fetch('/api/frame')
        .then(function(r) {
          if (r.status === 204) { setTimeout(fetchFrame, 100); return null; }

          // X-Frame-Seq: detect if the server is sending duplicate frames.
          // This does NOT skip rendering (we always show the latest), but
          // logs a warning so we can diagnose a stalled capture thread.
          const seq = parseInt(r.headers.get('X-Frame-Seq') || '-1', 10);
          if (seq !== -1 && seq === lastFrameSeq) {
            console.warn('Duplicate frame seq=' + seq + ' -- capture thread may be stalled');
          }
          lastFrameSeq = seq;

          return r.blob();
        })
        .then(function(blob) {
          if (!blob) return;
          const url = URL.createObjectURL(blob);

          offscreen.onload = function() {
            // requestAnimationFrame: schedule the draw to the browser's next
            // vsync paint cycle. Drawing outside the paint cycle can cause
            // the old GPU texture layer to persist until the next repaint,
            // producing a "ghost" of the previous frame visible beneath the
            // new one. rAF eliminates this by guaranteeing the draw happens
            // at the correct point in the compositing pipeline.
            requestAnimationFrame(function() {
              // clearRect: explicitly zero the canvas before drawing.
              // Redundant with a full-canvas drawImage, but prevents any
              // partial-paint artifacts if the new frame is slightly smaller
              // or if the GPU compositor has a stale layer cached.
              ctx.clearRect(0, 0, feed.width, feed.height);
              ctx.drawImage(offscreen, 0, 0, feed.width, feed.height);
              if (currentBlob) URL.revokeObjectURL(currentBlob);
              currentBlob = url;
              offscreen = new Image();
              fetchFrame();
            });
          };

          offscreen.onerror = function() {
            URL.revokeObjectURL(url);
            setTimeout(fetchFrame, 200);
          };

          offscreen.src = url;
        })
        .catch(function() { setTimeout(fetchFrame, 200); });
    }

    fetchFrame();  // start the loop

    // --- Cursor temperature readout ---
    // getBoundingClientRect() works identically on <canvas> as on <img>.
    // offsetX/offsetY would also work here but clientX - rect.left is more portable.
    feed.addEventListener('mousemove', function(e) {
      const rect = this.getBoundingClientRect();
      const x = Math.round(e.clientX - rect.left);
      const y = Math.round(e.clientY - rect.top);
      fetch('/api/temp_at?x=' + x + '&y=' + y)
        .then(r => r.json())
        .then(d => {
          // d.unit comes from the server so the label always matches the frame annotation
          if (d.temp !== null) {
            cursorBox.textContent = d.temp.toFixed(1) + '\u00b0' + d.unit;
          }
        })
        .catch(() => {});
    });

    feed.addEventListener('mouseleave', function() {
      cursorBox.textContent = '--';
    });

    // --- Palette cycling ---
    function cyclePalette() {
      fetch('/api/palette', {method: 'POST'})
        .then(r => r.json())
        .then(d => { msgDiv.textContent = 'Palette: ' + d.palette; })
        .catch(() => {});
    }

    // --- Unit toggle ---
    // Sends POST to /api/unit which flips C<->F server-side.
    // Server returns the new unit so we can update the button label immediately.
    // The frame annotation and all API responses update automatically on the next poll.
    function toggleUnit() {
      fetch('/api/unit', {method: 'POST'})
        .then(r => r.json())
        .then(d => {
          document.getElementById('unitBtn').textContent = '\u00b0' + d.unit + ' [u]';
          msgDiv.textContent = 'Unit: \u00b0' + d.unit;
        })
        .catch(() => {});
    }

    // --- Save snapshot ---
    function saveSnapshot() {
      msgDiv.textContent = 'Saving...';
      fetch('/api/save', {method: 'POST'})
        .then(r => r.json())
        .then(d => { msgDiv.textContent = 'Saved: ' + d.path; })
        .catch(() => { msgDiv.textContent = 'Save failed.'; });
    }

    // --- Stats bar (polls /api/stats every second) ---
    // Using polling (setInterval) rather than WebSocket keeps the server simple.
    // At 1 Hz this has negligible overhead.
    function updateStats() {
      fetch('/api/stats')
        .then(r => r.json())
        .then(d => {
          // d.unit is included in the stats response so the label always matches
          const u = '\u00b0' + d.unit;
          statsDiv.textContent =
            'Min: ' + d.min + u + '   ' +
            'Max: ' + d.max + u + '   ' +
            'Mean: ' + d.mean + u + '   ' +
            'FPS: ' + d.fps;
        })
        .catch(() => { statsDiv.textContent = 'stats unavailable'; });
    }
    setInterval(updateStats, 1000);
    updateStats();

    // --- Keyboard shortcuts ---
    document.addEventListener('keydown', function(e) {
      if (e.key === 'p') cyclePalette();
      if (e.key === 'u') toggleUnit();
      if (e.key === 's') saveSnapshot();
    });
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Suppress Flask's per-request access log -- it would spam the terminal
# at 25fps. We still see startup messages and errors.
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)


@app.route('/')
def index():
    """Serve the HTML UI. Cache-Control: no-store prevents the browser from
    loading a cached copy of the old MJPEG-based HTML, which would open a
    /video_feed connection in parallel with the canvas feed and cause ghosting."""
    return Response(HTML, mimetype='text/html',
                    headers={'Cache-Control': 'no-store, must-revalidate'})


def generate_mjpeg():
    """
    Generator function for the MJPEG stream.

    MJPEG protocol:
      HTTP Content-Type: multipart/x-mixed-replace; boundary=frame
      Each part:
          --frame\r\n
          Content-Type: image/jpeg\r\n
          \r\n
          {jpeg bytes}
          \r\n

    The browser's <img> tag interprets the stream and replaces the displayed
    image each time a new JPEG part arrives, creating live video.

    We poll the shared state at up to 25fps. If no new frame is ready we
    sleep briefly to avoid burning CPU.
    """
    while True:
        with state['lock']:
            jpeg = state['jpeg']
        if jpeg is not None:
            # Content-Length is critical for MJPEG stability.
            # Without it the browser guesses frame boundaries by scanning for the
            # next '--frame' marker. If it misparses, it layers frames on top of
            # each other (ghosting). Content-Length makes framing unambiguous.
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(jpeg)).encode() + b'\r\n'
                   b'\r\n' + jpeg + b'\r\n')
        time.sleep(0.04)   # ~25fps poll


@app.route('/video_feed')
def video_feed():
    """MJPEG stream endpoint. Kept for reference; UI now uses /api/frame polling."""
    return Response(
        generate_mjpeg(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/frame')
def api_frame():
    """
    Single JPEG frame for JS polling.

    Why polling instead of MJPEG:
      MJPEG relies on the browser correctly parsing a long-running multipart HTTP
      response. Browsers have well-known bugs with this: after some time they
      start layering frames on top of each other (ghosting) instead of replacing
      them. Adding Content-Length helps but does not fully eliminate the problem.

      JS polling fetches individual JPEG frames via normal HTTP requests.
      Each frame is a clean, complete HTTP transaction. The browser never has to
      parse a multipart stream -- it just loads a JPEG, which every browser
      handles perfectly. Double-buffering in JS eliminates flicker.

    Cache-Control: no-store prevents the browser from serving a cached frame
    even if the cache-busting timestamp in the URL is somehow reused.
    """
    with state['lock']:
        jpeg     = state['jpeg']
        frame_seq = state['frame_seq']
    if jpeg is None:
        return '', 204   # No Content -- client will retry
    return Response(jpeg, mimetype='image/jpeg',
                    headers={
                        'Cache-Control': 'no-store',
                        'X-Frame-Seq':   str(frame_seq),
                    })


@app.route('/api/temp_at')
def temp_at():
    """
    Return temperature at display pixel (x, y).
    x, y are in display image coordinates (0-767, 0-575).
    Server maps to sensor coordinates by dividing by SCALE (3).
    """
    x = int(request.args.get('x', 0))
    y = int(request.args.get('y', 0))
    sx = min(max(x // SCALE, 0), WIDTH - 1)
    sy = min(max(y // SCALE, 0), IR_ROWS - 1)
    with state['lock']:
        temp_c = state['temp_c']
        unit   = state['unit']
    if temp_c is None:
        return jsonify({'temp': None, 'unit': unit})
    return jsonify({'temp': round(c_to_unit(float(temp_c[sy, sx]), unit), 1), 'unit': unit})


@app.route('/api/stats')
def api_stats():
    """
    Return min / max / mean / fps as JSON, converted to the current display unit.
    Stats are stored internally in Celsius; conversion happens here at the output boundary.
    The 'unit' field in the response tells the browser which symbol to display.
    """
    with state['lock']:
        s    = state['stats']
        unit = state['unit']
    return jsonify({
        'min':  round(c_to_unit(s['min'],  unit), 1),
        'max':  round(c_to_unit(s['max'],  unit), 1),
        'mean': round(c_to_unit(s['mean'], unit), 1),
        'fps':  s['fps'],
        'unit': unit,
    })


@app.route('/api/palette', methods=['POST'])
def api_palette():
    """Cycle to the next false-color palette. Returns new palette name."""
    with state['lock']:
        state['palette'] = (state['palette'] + 1) % len(PALETTES)
        name = PALETTES[state['palette']][0]
    return jsonify({'palette': name})


@app.route('/api/unit', methods=['POST'])
def api_unit():
    """
    Toggle temperature display unit between Celsius and Fahrenheit.
    The internal temp_c array always stays in Celsius -- only the display changes.
    Returns the new unit so the browser can update its labels.
    """
    with state['lock']:
        state['unit'] = 'F' if state['unit'] == 'C' else 'C'
        u = state['unit']
    return jsonify({'unit': u})


@app.route('/api/save', methods=['POST'])
def api_save():
    """
    Save the current frame as PNG and the temperature array as .npy.
    PNG: lossless, suitable for archiving.
    npy: raw float32 temperature data, loadable with np.load() for analysis.
    """
    with state['lock']:
        jpeg   = state['jpeg']
        temp_c = state['temp_c']

    if jpeg is None or temp_c is None:
        return jsonify({'error': 'no frame yet'}), 503

    ts = time.strftime('%Y%m%d_%H%M%S')
    png_path = os.path.join(SAVE_DIR, f'snapshot_{ts}.png')
    npy_path = os.path.join(SAVE_DIR, f'snapshot_{ts}.npy')

    # Decode JPEG back to BGR and save as lossless PNG
    buf = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    cv2.imwrite(png_path, img)
    np.save(npy_path, temp_c)

    print(f'Saved: {png_path}  {npy_path}')
    return jsonify({'path': f'snapshot_{ts}.png'})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f'Starting capture thread...')
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    # Give the capture thread a moment to get the first frame before Flask starts.
    # Not strictly required, but avoids a brief "no frame" state on first connect.
    time.sleep(1.0)

    print(f'Thermal server running.')
    print(f'Open in browser:')
    print(f'  Local:   http://localhost:{PORT}')
    print(f'  Network: http://{os.uname().nodename}:{PORT}')
    print(f'Press Ctrl+C to stop.')

    # threaded=True: Flask handles each request in its own thread.
    # This is important because generate_mjpeg() is a slow streaming generator
    # and would block all other requests if Flask ran single-threaded.
    app.run(host='0.0.0.0', port=PORT, threaded=True)
