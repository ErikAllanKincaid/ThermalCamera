#!/usr/bin/env bash
# start.sh -- resolve the udev symlink before handing the device to Docker.
#
# Docker's `devices:` directive does not follow udev symlinks. It sees
# /dev/thermal_cam as a symlink (not a char device) and either skips the
# cgroup allowance or creates a device node with the wrong major:minor inside
# the container. Resolving to the real /dev/videoN path first avoids this.
#
# Usage:
#   ./start.sh              -- same as docker compose up
#   ./start.sh --build      -- rebuild image then start
#   ./start.sh -d           -- detach (background)
#   ./start.sh down         -- stop and remove containers

SYMLINK="${THERMAL_SYMLINK:-/dev/thermal_cam}"

if [ -L "$SYMLINK" ]; then
    REAL_DEV=$(readlink -f "$SYMLINK")
    if [ -c "$REAL_DEV" ]; then
        export THERMAL_DEVICE_HOST="$REAL_DEV"
        echo "Thermal device: $SYMLINK -> $REAL_DEV"
    else
        echo "Warning: $SYMLINK -> $REAL_DEV is not a char device. Falling back to $SYMLINK."
        export THERMAL_DEVICE_HOST="$SYMLINK"
    fi
elif [ -c "$SYMLINK" ]; then
    export THERMAL_DEVICE_HOST="$SYMLINK"
    echo "Thermal device: $SYMLINK"
else
    echo "Error: thermal camera not found at $SYMLINK"
    echo "  Is the camera plugged in?"
    echo "  Is 99-thermal-cam.rules installed in /etc/udev/rules.d/?"
    exit 1
fi

exec docker compose "$@"
