<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# DIY-Key on Linux (Orange Pi 5 Plus / RK3588)

How to run the AI work on the deployed runtime: Debian/Ubuntu on an Orange Pi 5 Plus, one
systemd unit per instance, a system PostgreSQL with pgvector. The macOS workflow is unchanged;
see [`macos.md`](macos.md).

## What runs where

| | macOS (`diykey_service.sh`) | Linux (this doc) |
|---|---|---|
| Process supervision | shell wrapper, `~/.diy-key/<name>/` | systemd, `core/instance_manager.py` |
| Address | `ifconfig <if> alias` | macvlan child interface (or a real NIC) |
| State | `~/.diy-key/<name>/state/` | `diykey/instances/<name>/` |
| PostgreSQL | per-instance cluster in `pg/` | system cluster on `:5432` (role/db `unifi-protect`) |
| Rotated PG password | `DIYKEY_PG_SOCKET=<name>/pg` | automatic: psql as `postgres` over `/var/run/postgresql` |
| Settings file | `~/.diy-key/<name>/diykey.env` | `<state-dir>/diykey.env` (unit `EnvironmentFile`) |

The AI code itself is platform-neutral: the COCO detector runs through `onnxruntime`
(CPUExecutionProvider), and CLIP uses whichever of MPS/CUDA/CPU torch finds. Nothing in
`ram_tagger`, `clip_embed`, `nl_search`, `second_verifier`, `clip_fetch` or `firmware` is
macOS-specific.

## Dependencies

```bash
sudo apt-get install -y python3-venv ffmpeg postgresql postgresql-server-dev-15
# pgvector is not packaged for Debian/aarch64; build it (see diykey/README.md "PostgreSQL setup")

python3 -m venv ~/diykey-venv
. ~/diykey-venv/bin/activate
pip install -r diykey/requirements.txt
```

`onnxruntime` (RAM tags, verification) is a normal aarch64 wheel. `torch` + `open_clip_torch`
are only needed when `DIYKEY_CLIP=1`; they install on aarch64 but are large, and CPU CLIP is
slow — leave Find Anything off unless you want it on the board.

## Configure an instance

Create the unit as before (identity is fixed at creation):

```bash
cd diykey
sudo python3 core/instance_manager.py create main --parent-iface <board-nic> \
  --mac FC:EC:DA:00:00:01 --device-id <device-id> --state-dir instances/main
```

Then put the AI settings in the per-instance env file the unit reads:

```bash
sudo tee /path/to/diy-key/diykey/instances/main/diykey.env >/dev/null <<'EOF'
# COCO ONNX detector (Ultralytics anchor-free export, e.g. pi-port's yolov9-s-320.onnx).
DIYKEY_RAM_MODEL=/root/models/yolov9-s-320.onnx
# Find Anything (CLIP ViT-L/14 + the nl-search WebSocket). Optional, heavy.
# DIYKEY_CLIP=1
# DIYKEY_NL_SEARCH=1
# DIYKEY_RAM_TAGS=/root/up/app/fixtures/ram_tags.csv
# DIYKEY_FW_DIR=/root/diy-key-firmware
# Features advertised to Protect; each is off unless set here.
# DIYKEY_FACE=1          # face recognition
# DIYKEY_FACE_MODELS=/path/to/models
# DIYKEY_LPR=1           # licence plate recognition
# DIYKEY_FACE_ENHANCE=1  # face enhancement
# DIYKEY_FACE_ENHANCE_MODEL=/path/to/gfpgan_1.4.onnx
# DIYKEY_VLM=mlx         # AI summary: mlx (macOS only) or transformers; off/0 or unset = off
# DIYKEY_VLM_MODEL=mlx-community/<SmolVLM2 or Qwen3-VL-2B repo>
# DIYKEY_TASK_WORKERS=1
# DIYKEY_TASK_QUEUE_MAX=200
EOF
sudo systemctl restart diy-key-instance-main
```

`EnvironmentFile` is `-`-prefixed, so a missing file does not stop the unit. Identity and
network stay in the unit (macvlan MAC, `--state-dir`); the env file is for the AI settings
only. `DIYKEY_FW_VERSION` / `DIYKEY_DISCOVERY_FW_VERSION` also work there if you want to pin a
factory version without touching `diykey.cfg`. The full list of keys is in
`scripts/diykey.env.example`.

## Verify

```bash
journalctl -u diy-key-instance-main -f
```

Expected once adopted: `WSS CONNECTED`, and with the detector configured, `RAM tagger: …` at
the first `recognizeKeyFrames` task followed by `event …: <description>` and
`POST …/recognize-anything -> 200`. With CLIP, `NL_PARSE … -> tags … embed 768` and
`nl-search connected to wss://<console>:7443/wss/nl-search/v1`.

## Known limits on this runtime

- **No NPU path for the new stages.** `stage5_faceid/` still uses `rknnlite`, but RAM tagging,
  CLIP and verification run on CPU via onnxruntime/torch. Feeding the detector through RKNN is
  not implemented.
- **One shared PostgreSQL.** The console's result database is the host cluster, so several
  instances on one board would share `:5432`; the macOS per-instance cluster is not used.
- **RAM verified, CLIP not.** RAM tagging ran live on the board (2026-10-30, CPU); CLIP and
  Find Anything have only been exercised on macOS against Protect 7.3.70.
- `scripts/diykey_*.sh` remain the macOS service; the Linux path is systemd only.
