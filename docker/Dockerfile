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
# OSNet re-ID; from PyTorch's CPU index, since requirements-clip.txt's torch would pull the multi-GB
# CUDA build. requirements.txt (onnxruntime, OpenCV), requirements-lpr.txt (fast-alpr) and the rest
# of requirements-clip.txt (Pillow, tokenizers for the ONNX CLIP) are always in.
#
# ORT=openvino (amd64) swaps onnxruntime for onnxruntime-openvino and adds Intel's GPU compute
# runtime, so DIYKEY_ORT_DEVICE=GPU runs the onnxruntime models on an Intel iGPU/Arc (see
# stage3_wss/ort_device.py; run with --device /dev/dri). Debian trixie has no intel-opencl-icd,
# so these are Intel's release debs, sha256-pinned (the same ones diy-port's image uses). It also
# adds OpenVINO GenAI (OV_GENAI, ~200 MB) for AI Summary on the GPU (DIYKEY_VLM=openvino); it
# brings its own OpenVINO runtime next to the one inside onnxruntime-openvino, and both run in one
# process (checked 2026-11-11: detector session on the GPU before and after a VLM summary).
#
# Run it with its own LAN address (macvlan) and NET_ADMIN, like the systemd instances: the
# console must reach :8080 and :5432, and discovery is UDP broadcast.

FROM debian:trixie-slim

ARG TORCH=0
ARG ORT=cpu
ARG TARGETARCH
ARG NEO_TAG=26.35.39758.10
ARG IGC_TAG=v2.41.5
ARG IGC_VER=2.41.5+22716
ARG GMMLIB_VER=22.10.0
ARG OV_GENAI=2026.4.1.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# hadolint ignore=DL3008
RUN case "$TORCH" in 0|1) ;; *) echo "TORCH must be 0 or 1" >&2; exit 1 ;; esac \
 && case "$ORT" in cpu|openvino) ;; *) echo "ORT must be cpu or openvino" >&2; exit 1 ;; esac \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        python3 python3-venv ffmpeg openssl ca-certificates procps iproute2 curl \
        postgresql-17 postgresql-17-pgvector \
 && if [ "$ORT" = "openvino" ]; then \
        [ "$TARGETARCH" = "amd64" ] || { echo "ORT=openvino: amd64 only, got '$TARGETARCH'" >&2; exit 1; }; \
        igc="https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_TAG}"; \
        neo="https://github.com/intel/compute-runtime/releases/download/${NEO_TAG}"; \
        mkdir /tmp/neo && cd /tmp/neo; \
        for url in "$igc/intel-igc-core-2_${IGC_VER}_amd64.deb" \
                   "$igc/intel-igc-opencl-2_${IGC_VER}_amd64.deb" \
                   "$neo/intel-opencl-icd_${NEO_TAG}-0_amd64.deb" \
                   "$neo/libigdgmm12_${GMMLIB_VER}_amd64.deb"; do \
            curl -fsSLO "$url"; \
        done; \
        printf '%s\n' \
            "0a6e64a663ae65a0fa02d6912ae3b6b37cf85b90c21cc423fd9fef70aaf4f628  intel-igc-core-2_${IGC_VER}_amd64.deb" \
            "779e1b9e88098eb25711e9a8f67c2752665bad22f134aa40ed5649f6e1b87058  intel-igc-opencl-2_${IGC_VER}_amd64.deb" \
            "61712caaddeba3d38e4f79e2a0fb23fea25596ca2d72c3144c6eea2331ec4301  intel-opencl-icd_${NEO_TAG}-0_amd64.deb" \
            "6031a63d6e8a12ce61c14efc15f2c8e727061286e3820b8594e6d00615e04d54  libigdgmm12_${GMMLIB_VER}_amd64.deb" \
            | sha256sum -c -; \
        apt-get install -y --no-install-recommends /tmp/neo/*.deb; \
        cd / && rm -rf /tmp/neo; \
    fi \
 && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY diykey/requirements.txt diykey/requirements-lpr.txt diykey/requirements-clip.txt /tmp/
# ORT=openvino: onnxruntime-openvino ships the same "onnxruntime" module, so the plain wheel
# (also pulled in for fast-alpr) is removed first, and the build fails without the provider.
# hadolint ignore=DL3013
RUN python3 -m venv "$VIRTUAL_ENV" \
 && cat /tmp/requirements.txt /tmp/requirements-lpr.txt /tmp/requirements-clip.txt \
    | grep -vE '^(torch|open_clip_torch)([<>=! ]|$)' > /tmp/requirements-base.txt \
 && pip install --no-cache-dir -r /tmp/requirements-base.txt \
 && if [ "$TORCH" = 1 ]; then \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
     && pip install --no-cache-dir "open_clip_torch>=2.24"; \
    fi \
 && if [ "$ORT" = "openvino" ]; then \
        pip uninstall -y onnxruntime && pip install --no-cache-dir onnxruntime-openvino \
     && python3 -c 'import onnxruntime as o, sys; sys.exit("OpenVINOExecutionProvider" not in o.get_available_providers())' \
     && pip install --no-cache-dir "openvino-genai==$OV_GENAI" \
     && python3 -c 'import openvino_genai; openvino_genai.VLMPipeline'; \
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
