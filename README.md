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
| `thermal_server.py` | Main application. Flask + MJPEG live server, port 5800. |
| `extract_temp.py` | Single-shot capture. Saves `temp_overlay.png` and `temp_gray.png`. |
| `probe_386.py` | Frame layout discovery tool. Used during initial reverse engineering. |
| `thermal_live.py` | OpenCV window live display. Superseded by Flask server; kept for reference. |
| `Dockerfile` | Container build. `python:3.12-slim` + uv + opencv-python-headless. |
| `docker-compose.yml` | Compose config. Device passthrough, recordings volume, auto-restart. |
| `99-thermal-cam.rules` | Udev rule. Creates `/dev/thermal_cam` stable symlink by VID:PID. |
| `requirements.txt` | Python dependencies for Docker build. |

---

## Run (Bare Metal)

```bash
cd ThermalCamera
uv venv
uv pip install -r requirements.txt
uv run python3 thermal_server.py
```

Open `http://localhost:5800` in a browser.

---

## Run (Docker)

**Step 1: Install the udev rule (one-time, on the host)**

The camera enumerates as `/dev/videoN` where N changes on every replug and reboot.
The udev rule creates a stable `/dev/thermal_cam` symlink matched by USB VID:PID.

```bash
sudo cp 99-thermal-cam.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Verify (camera must be plugged in):

```bash
ls -la /dev/thermal_cam
```

**Step 2: Build and run**

```bash
docker compose up --build
```

Open `http://localhost:5800` in a browser.

Snapshots and recordings are saved to `./recordings/` on the host.

**Without the udev rule (manual device)**

If you skip the udev rule, find the device node and pass it explicitly:

```bash
lsusb | grep 3474        # find bus/device
ls /dev/video*           # identify node
THERMAL_DEVICE=/dev/video0 docker compose up --build
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
| `THERMAL_DEVICE` | (auto-detect) | Camera device path. Set to skip VID:PID sysfs scan. |
| `SAVE_DIR` | Script directory | Where snapshots and recordings are written. |

These apply to both bare-metal and container deployments.

---

## Auto-Reconnect

The capture thread validates every frame:

1. Packet size must equal `256 x 386 x 2` bytes.
2. Sync row (row 193) must begin with `[255, 0, 255, 0, ...]`.
3. Temperature values must fall within -50C to 200C.

After 8 consecutive bad frames the camera container is closed and reopened.
Physically replugging the camera is not required.

---

## Known Limitations

- **Absolute temperature accuracy**: The firmware performs NUC correction internally before streaming. Host-side recalibration is not possible without implementing the proprietary IRCMD USB protocol (native `.so` library, not decompilable). The `/64.0 - 273.15` formula gives accurate readings under normal conditions.
- **Single camera**: The server handles one camera. Multi-camera split view is not implemented.
- **256x386 mode required**: The 256x194 mode (thermal image only, no temp data) does not expose the raw temperature rows and is not supported by this server.
