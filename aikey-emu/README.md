# aikey-emu — DIY UniFi Protect "AI Key" on Orange Pi 5 Plus / RK3588

**STATUS: adoption works end-to-end, live, against a real UniFi Protect
console (confirmed 2026-09-04).** The Pi shows up in Protect as `AI Key`,
adopts cleanly, and holds a stable connection. See `../protocol.md` for
the full confirmed wire-level writeup and `../llm_decisions.md` for the
project's history. This README describes the resulting architecture.

## Real protocol (confirmed live, not guessed)

Turned out to be architecturally close to the already-reverse-engineered
UniFi Protect **Chime** protocol — not the SSH `mca-ctrl` mechanism found
in AI Key's own firmware, which appears to be for a different/unused
adoption path.

1. **UDP/10001 discovery** — the classic UBNT L2 discovery protocol,
   generic across device classes. The `platform` TLV field (`0x0C`) must
   be a SKU string the controller's catalog recognizes; `UP-AI-KEY` works.
2. **HTTPS control API on the device, port 8080** — `POST /api/info`
   (polled ~60s), `POST /api/adopt` (on clicking Adopt; body includes
   `hosts`, `token`, `protocol:"wss"`, `consoleId`, etc. — same shape as
   the chime spec).
3. **Outbound mTLS WebSocket to `hosts[0]` (`<console-ip>:7442`)** — same
   binary `ucp4` envelope as chime (8-byte record header + JSON payload,
   one head record + one body record per message). RPCs specific to
   `aiprocessors` (not in the chime spec) discovered live: `getInfo`,
   `networkStatus`, `changeUserPassword`, `sshService`, `firmwareState`,
   `diskInfo`, `updateTimezone`, `changeAiInferAgentSettings` (the real
   feature toggles: face recognition, LPR, RAM, STT, region),
   `setConsoleInfo`, `updateLcmSettings`, `getTaskQueueInfo` (a ~5s
   keepalive) and `systemMetrics` (backs the System Performance tab).
   Exact response *schemas* for `systemMetrics` and `getTaskQueueInfo`
   were confirmed by reading live Zod validation errors out of the real
   controller's own `errors.log` and cross-referencing its bundled
   `service.js` source — see `../protocol.md` §3 for the exact shapes.
4. **Direct PostgreSQL connection from the console to the device, port
   5432** — confirmed to be the actual AI-results delivery channel (not
   any RPC). See "PostgreSQL setup" below.

The local gRPC service (`stage2_grpc/`, byte-compatible with the real
`.proto` contract) does **not** appear to be dialed by the console during
this flow — it may serve a different purpose (e.g. detections streaming
once real inference exists) not yet exercised.

## Components

- **`stage1_discovery/beacon.py`** — UDP/10001 discovery responder.
- **`stage1_http/adopt_http.py`** — HTTPS `:8080` server (`/api/info`,
  `/api/adopt`); on a successful adopt, writes `adopt_state.json` for the
  WSS client.
- **`stage3_wss/ucp4_client.py`** — outbound `ucp4` WebSocket client that
  completes adoption and handles the ongoing RPC traffic. Falls back to a
  tokenless reconnect to a known console host if the device restarts
  after already being adopted server-side, and actively watches
  `adopt_state.json` for a fresh token to force a clean reconnect after
  an unadopt/readopt cycle (both failure modes were subtle — see
  `../todo.md`'s "adopt_state.json doesn't persist across reboots" item
  if touching this logic).
- **`stage3_wss/system_stats.py`** — real RK3588 hardware stats (CPU via
  `/proc/stat`, GPU via `/sys/class/devfreq/*.gpu/load`, NPU via
  `/sys/kernel/debug/rknpu/load`, RAM, storage) feeding `systemMetrics`.
- **`stage2_grpc/`** — stub `ProtectAPI` gRPC server on `:50051`, real
  `.proto` contract. Kept running; not yet confirmed to be used by the
  live flow.
- **`proto/`** — verbatim `.proto` files from the real `ui-uaibd` package.
- **`stage5_faceid/`** — real face detection + embedding on the RK3588 NPU, standalone-tested
  (not yet wired into `run_all.py` or the ucp4/model-server RPC handlers — see `../todo.md`):
  - `face_detect.py` — `FaceDetector`, wraps a RetinaFace_mobile320 RKNN model (converted from
    `airockchip/rknn_model_zoo`'s example) via `rknnlite`. ~9ms/frame.
  - `face_embed.py` — `FaceEmbedder` + `align_face()`, wraps a MobileFaceNet RKNN model
    (`w600k_mbf`, insightface's `buffalo_sc` pack, converted fp16/unquantized) plus the standard
    insightface 5-point alignment. Confirmed on a 6-face group photo: same-face similarity 1.000,
    cross-face pairs near 0.
  - `pipeline.py` — `FacePipeline`, combines the two into one `process(image) -> [{bbox, score,
    embedding}]` call. This is the entry point to use once real frames/dispatch are wired in.
  - `models/` — the converted `.rknn` files plus test images.

(`stage1_sshd/`, an SSH honeypot that turned out to be a dead end for the
real adoption flow, was archived to `../archive/aikey-emu_stage1_sshd/` —
see `../llm_decisions.md`.)

## Quick start on the Orange Pi

```bash
sudo apt-get install -y python3-pip python3-venv openssl
python3 -m venv ~/aikey-emu-venv
source ~/aikey-emu-venv/bin/activate
pip install -r requirements.txt

# Generate the gRPC/protobuf python bindings (only needs the files stage2_grpc
# actually imports -- console/v1's api.proto and device.proto pull in protos
# not copied here and aren't needed):
python -m grpc_tools.protoc -I proto \
  --python_out=stage2_grpc/generated --grpc_python_out=stage2_grpc/generated \
  proto/unifi/protect/v1/api.proto proto/unifi/protect/v1/types.proto \
  proto/unifi/core/console/v1/types.proto proto/google/type/latlng.proto
find stage2_grpc/generated -type d -not -path 'stage2_grpc/generated/google*' -exec touch {}/__init__.py \;
rm -f stage2_grpc/generated/google/__init__.py stage2_grpc/generated/google/type/__init__.py

# Run everything (needs root for ports 22/10001/50051/8080)
sudo ~/aikey-emu-venv/bin/python run_all.py --iface <your-iface> --hostname aikey-emu
```

Then trigger "Adopt" from the Protect console UI. Watch `run_all.log` (or
the terminal) — you'll see the discovery probes, the `/api/adopt` POST
with its real `hosts`/`token`, the WSS connect, and the RPC handshake.

## PostgreSQL setup (required for the console to fully "operate" the device)

Confirmed exact connection parameters (from the real controller's
`service.js`): host = this device's IP, port `5432`, user `unifi-protect`,
password `unifi-protect` (rotated later via the `changeUserPassword` RPC),
database `unifi-protect`, SSL required but not certificate-validated.
**The console migrates its own schema into this database** — you only
need to stand up an empty, correctly-configured instance:

```bash
sudo apt-get install -y postgresql postgresql-server-dev-15
# SSL is on by default via Debian's snakeoil cert -- sufficient, since the
# controller doesn't validate it. Just open it up to the LAN:
sudo sed -i "s/^#listen_addresses = 'localhost'/listen_addresses = '*'/" /etc/postgresql/15/main/postgresql.conf
echo "hostssl unifi-protect unifi-protect <your-lan-cidr> scram-sha-256" | sudo tee -a /etc/postgresql/15/main/pg_hba.conf
sudo -u postgres psql -c "CREATE ROLE \"unifi-protect\" WITH LOGIN PASSWORD 'unifi-protect' SUPERUSER;"
sudo -u postgres psql -c "CREATE DATABASE \"unifi-protect\" OWNER \"unifi-protect\";"

# pgvector isn't packaged for Debian/aarch64 -- build from source:
git clone --depth 1 https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install
sudo -u postgres psql -d unifi-protect -c "CREATE EXTENSION IF NOT EXISTS vector;"
# (SUPERUSER above is needed because vector.control lacks `trusted = true`,
# so even IF NOT EXISTS needs superuser to satisfy the console's own
# unconditional CREATE EXTENSION during migration.)

sudo systemctl restart postgresql
```

Then trigger a fresh WSS reconnect (kill and restart `ucp4_client.py`, or
just wait — the console retries the PG connection with backoff on every
reconnect) and the console's migrator creates the real tables:
`ramDetections` (`embedding vector(768)`, `reidEmbedding vector(1024)`),
`ramDetectionLabels`, `smartDetectSessionObjects`
(`reidEmbedding vector(512)`), `smartDetectSessionsSearch`
(`descEmbedding vector(384)`, free-text `description`). Full column list
in `../protocol.md` §6 — this is the real schema for delivering AI
results into Protect.

## Known open questions / next steps

See `../todo.md` for the current, actively-maintained list (facial
recognition is the active thread; NPU inference, video-frame delivery,
the gRPC dial question, and `adopt_state.json` persistence are parked/
open questions there).

## Safety / scope note

This only talks to hardware and a console **you own**, using vendor-default
credentials **you are entitled to use on your own gear**, purely to
interoperate with your own UniFi Protect install. Nothing here targets
third-party systems.
