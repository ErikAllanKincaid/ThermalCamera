# Thermal Master P2 -- Linux

Linux driver and live web interface for the Raysentek Thermal Master P2 USB thermal camera.

The camera presents as a standard UVC device. No kernel module or proprietary driver is required.

---

## Hardware

| Field | Value |
|-------|-------|
| Manufacturer | Raysentek Co., Ltd |
| USB VID:PID | 3474:4281 |
| Kernel driver | uvcvideo (auto-loads) |
| Sensor resolution | 192 x 256 pixels |
| Temperature range | -20C to 150C |
| Frame rate | 25 fps |

---

## How It Works

The camera streams YUYV 4:2:2 video at 256x386 pixels. The frame is split into two halves:

```
Rows   0-191:  8-bit grayscale thermal image (white-hot palette, pre-rendered by firmware)
Rows 192-193:  Firmware sync/info rows (row 193 is a fixed [255, 0, 255, 0, ...] marker)
Rows 194-385:  Raw 16-bit temperature data, little-endian uint16 per pixel
```

Temperature formula (confirmed from APK source, `LibIRTemp.java`):

```
temp_celsius = uint16_value / 64.0 - 273.15
```

The values in rows 194-385 are NUC-corrected by the camera firmware before streaming. No host-side calibration is required.

---

## Files

| File | Description |
|------|-------------|
| `thermal_server.py` | Main application. Flask live server, port 7700. |
| `extract_temp.py` | Single-shot capture. Saves `temp_overlay.png` and `temp_gray.png`. |
| `probe_386.py` | Frame layout discovery tool. Used during initial reverse engineering. |
| `thermal_live.py` | OpenCV window live display. Superseded by Flask server; kept for reference. |
| `Dockerfile` | Container build. Ubuntu 24.04 + uv + opencv-python-headless + ffmpeg 6.x. |
| `docker-compose.yml` | Compose config. Device passthrough, recordings volume, auto-restart. |
| `start.sh` | Wrapper for `docker compose` that resolves the udev symlink before passing the device to Docker. Use this instead of `docker compose` directly. |
| `99-thermal-cam.rules` | Udev rule. Creates `/dev/thermal_cam` stable symlink by VID:PID. |
| `requirements.txt` | Python dependencies for Docker build and bare-metal use. |

---

## Run (Bare Metal)

```bash
cd ThermalCamera
uv venv
uv pip install -r requirements.txt
uv run python3 thermal_server.py           # default port 7700
uv run python3 thermal_server.py --port 9000   # custom port
THERMAL_PORT=9000 uv run python3 thermal_server.py     # via env var
```

Open `http://localhost:7700` in a browser.

---

## Run (Docker)

**Step 1: Install the udev rule (one-time, on the host)**

The camera exposes two V4L2 interfaces: a video capture interface (index 0) and a metadata
interface (index 1). Both share the same VID:PID. The udev rule uses `ATTR{index}=="0"` to
select only the capture interface, creating a stable `/dev/thermal_cam` symlink.

Without this filter, the symlink can land on the metadata node, which rejects all capture
ioctls with `ENOTTY (Inappropriate ioctl for device)`.

```bash
sudo cp 99-thermal-cam.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

Unplug and replug the camera, then verify:

```bash
ls -la /dev/thermal_cam
# should show -> videoN (the capture node, not the metadata node)
v4l2-ctl -d /dev/thermal_cam --info | grep 'Video Capture'
# should show "Video Capture" under Device Caps
```

**Step 2: Build and run**

Use `start.sh` instead of `docker compose` directly. Docker does not follow udev symlinks
when creating device cgroups -- it sees the symlink as a plain file rather than a char device,
which causes the passthrough to fail. `start.sh` resolves `/dev/thermal_cam` to the real
`/dev/videoN` path before handing it to Docker.

```bash
./start.sh up --build          # build image and start (first run)
./start.sh up -d               # start detached (subsequent runs)
./start.sh down                # stop
THERMAL_PORT=9000 ./start.sh up -d    # custom port
```

Open `http://localhost:7700` (or the host IP/hostname from another machine on the network).

Snapshots and recordings are saved to `./recordings/` on the host.

**Base image note**

The Dockerfile uses `ubuntu:24.04` (ffmpeg 6.1.1). Debian Bookworm (`python:3.12-slim`)
ships ffmpeg 7.x, which has a breaking change in its V4L2 demuxer: `VIDIOC_G_INPUT` failure
is treated as fatal. This camera does not implement `VIDIOC_G_INPUT`. Ffmpeg 6.x ignores the
failure and continues; 7.x aborts. Ubuntu 24.04 avoids this.

PyAV ships its own bundled ffmpeg in its PyPI wheel (currently 8.x), which also has this
behavior, so the system ffmpeg version matters for the ffmpeg binary path but not for PyAV
itself. Both will fail on Debian Bookworm due to the cgroup issue being a separate layer.

**Without the udev rule**

If you skip the udev rule, find the capture node manually and pass it explicitly:

```bash
# Find the capture node: look for the node where "Video Capture" appears under Device Caps
for n in /dev/video*; do
    if v4l2-ctl -d $n --info 2>/dev/null | grep -q 'Video Capture'; then
        echo "$n"
    fi
done

THERMAL_DEVICE_HOST=/dev/video4 ./start.sh up -d
```

---

## Web Interface

| Control | Action |
|---------|--------|
| Move cursor over image | Live temperature readout at cursor |
| `p` / Palette button | Cycle false-color palette (Inferno, Hot, Jet, Bone, White-hot, Black-hot) |
| `u` / C/F button | Toggle Celsius / Fahrenheit |
| `s` / Save button | Save current frame as PNG + raw `.npy` temperature array |
| `r` / Record button | Start/stop video recording (MP4, 768x576, 25 fps) |

Annotations on the live image:

- Min / Max / Mean temperature (top bar)
- Cyan cross: hotspot (maximum temperature pixel)
- Orange cross: coldspot (minimum temperature pixel)

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `THERMAL_DEVICE` | (auto-detect) | Camera device path inside the container. Set to skip VID:PID sysfs scan. |
| `THERMAL_DEVICE_HOST` | (from udev symlink) | Real device path on the host, set by `start.sh`. Override if needed. |
| `SAVE_DIR` | Script directory | Where snapshots and recordings are written. |
| `THERMAL_PORT` | 7700 | Port to listen on. Overridden by `--port` flag if both are set. |

---

## Auto-Reconnect

The capture thread validates every frame:

1. Packet size must equal `256 x 386 x 2` bytes.
2. Sync row (row 193) must begin with `[255, 0, 255, 0, ...]`.
3. Temperature values must fall within -50C to 200C.

After 8 consecutive bad frames the camera container is closed and reopened.
Physically replugging the camera is not required for software-level stream loss,
but is required if the USB device itself enters a bad state (visible in `dmesg`).

---

## Known Limitations

- **Absolute temperature accuracy**: The firmware performs NUC correction internally before streaming. Host-side recalibration is not possible without implementing the proprietary IRCMD USB protocol (native `.so` library, not decompilable). The `/64.0 - 273.15` formula gives accurate readings under normal conditions.
- **Single camera**: The server handles one camera. Multi-camera split view is not implemented.
- **256x386 mode required**: The 256x194 mode (thermal image only, no temp data) does not expose the raw temperature rows and is not supported by this server.
