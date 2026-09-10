#!/bin/sh
set -e

PUID=${PUID:-1000}
PGID=${PGID:-1000}

groupmod -o -g "$PGID" whisperx 2>/dev/null || groupadd -o -g "$PGID" whisperx
usermod -o -u "$PUID" -g "$PGID" whisperx 2>/dev/null || \
    useradd -o -u "$PUID" -g "$PGID" -M -d /config whisperx

mkdir -p /config/huggingface /config/torch /config/.cache
chown -R "$PUID:$PGID" /config

exec gosu whisperx "$@"
