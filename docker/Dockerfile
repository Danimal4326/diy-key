# syntax=docker/dockerfile:1
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# DIY-Key (UniFi Protect "AI Key" emulator) with its own PostgreSQL + pgvector, for
# Docker/Podman on Linux. Build from the repo root:
#
#   docker build -f docker/Dockerfile -t diykey .                      # without torch
#   docker build -f docker/Dockerfile --build-arg TORCH=1 -t diykey .  # + CPU torch/open_clip
#
# The console connects INTO this container's PostgreSQL (:5432) to deliver AI results, and
# rotates the role's password through the changeUserPassword RPC, which ucp4_client.py applies
# with psql over /var/run/postgresql as the postgres OS user: hence PostgreSQL in the same
# container. entrypoint.sh creates the cluster on first start under /data/pg.
#
# TORCH=1 adds CPU-only torch + open_clip for Find Anything (DIYKEY_CLIP/DIYKEY_NL_SEARCH) and
# OSNet re-ID; from PyTorch's CPU index, since requirements.txt's torch would pull the multi-GB
# CUDA build. Everything else in requirements.txt (onnxruntime, OpenCV, fast-alpr) is always in.
#
# Run it with its own LAN address (macvlan) and NET_ADMIN, like the systemd instances: the
# console must reach :8080 and :5432, and discovery is UDP broadcast.

FROM debian:trixie-slim

ARG TORCH=0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# hadolint ignore=DL3008
RUN case "$TORCH" in 0|1) ;; *) echo "TORCH must be 0 or 1" >&2; exit 1 ;; esac \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        python3 python3-venv ffmpeg openssl ca-certificates procps iproute2 \
        postgresql-17 postgresql-17-pgvector \
 && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY diykey/requirements.txt /tmp/requirements.txt
# hadolint ignore=DL3013
RUN python3 -m venv "$VIRTUAL_ENV" \
 && grep -vE '^(torch|open_clip_torch)([<>=! ]|$)' /tmp/requirements.txt > /tmp/requirements-base.txt \
 && pip install --no-cache-dir -r /tmp/requirements-base.txt \
 && if [ "$TORCH" = 1 ]; then \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
     && pip install --no-cache-dir "open_clip_torch>=2.24"; \
    fi \
 && rm -f /tmp/requirements*.txt

WORKDIR /app
COPY diykey/ diykey/
COPY docker/entrypoint.sh docker/healthcheck.sh /app/
RUN rm -rf diykey/tests diykey/instances \
 && find diykey -name __pycache__ -prune -exec rm -rf {} + \
 && chmod 755 /app/entrypoint.sh /app/healthcheck.sh

# Everything that must survive a rebuild lives on /data: pg/ (the results database), state/
# (adoption cert and token), firmware/, cache/ (open_clip / Hugging Face / fast-alpr
# downloads) and models/ (point DIYKEY_*_MODEL there).
ENV XDG_CACHE_HOME=/data/cache HF_HOME=/data/cache/huggingface TORCH_HOME=/data/cache/torch \
    DIYKEY_FW_DIR=/data/firmware DIYKEY_PG_BIN=/usr/lib/postgresql/17/bin
VOLUME /data
EXPOSE 10001/udp 8080/tcp 5432/tcp 50051/tcp 7788/tcp

STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD ["/app/healthcheck.sh"]
ENTRYPOINT ["/app/entrypoint.sh"]
