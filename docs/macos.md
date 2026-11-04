# DIY-Key on macOS (Protect 7.3)

How to build, configure and run DIY-Key as an AI Key on a Mac, as developed on the `MacOS`
branch against UniFi Protect 7.3.70 on a UCG-Max. One Mac can run several instances; each is an
AI Key with its own IP address, identity, state and PostgreSQL.

## What it does

| Protect feature | DIY-Key | Notes |
|---|---|---|
| Adoption, settings sync, firmware state | ✅ | `getInfo` advertises the AI Key feature flags (person re-ID, face, LPR, RAM, …) |
| Basic Understanding (`recognizeKeyFrames`) | ✅ | Clip fetched as MP4, frames tagged by a COCO ONNX detector, per-object CLIP embeddings |
| Retroactive processing | ✅ | Image tasks (`multipleImages`); one-shot per adoption |
| Second Stage Verification (`RequestAI`) | ✅ (offline-tested) | Box (0–1000 `[x,y,w,h]`) matched against the detector |
| Find Anything (typed search) | ✅ | The AI Key's nl-search WebSocket; CLIP ViT-L/14 text/image embeddings |
| Firmware update | Simulated | Real file stored, version pinned, nothing installed |
| Face / LPR / face enhancement / speech-to-text / VLM descriptions | ❌ | See `BACKLOG.md` |

## Prerequisites

```bash
brew install postgresql@17 pgvector ffmpeg python@3.14
```

- A COCO object detector exported to ONNX in the Ultralytics anchor-free layout
  (`[1, 84, N]` output), e.g. Frigate's `yolov9-s-320.onnx`.
- Optional, for Find Anything: ~2 GB of disk for torch and the CLIP ViT-L/14 weights (downloaded
  from Hugging Face on first use).
- Optional, for query tags: Protect's `app/fixtures/ram_tags.csv` from your console's Protect
  install (`/usr/share/unifi-protect/app/fixtures/ram_tags.csv`).

## Build

From the repository root:

```bash
python3 -m venv venv
venv/bin/pip install -r diykey/requirements.txt     # includes onnxruntime, torch, open_clip
venv/bin/python -m unittest discover -s diykey/tests   # 3 test_run_all errors are macOS-only
```

## Configure an instance

Everything for instance `<name>` lives in `~/.diy-key/<name>/` (see `scripts/diykey_paths.sh`):

```
diykey.env        settings (DIYKEY_* only)
state/            adopt state, server.crt/key (pinned by Protect), fw_version.json, clips/
pg/               the instance's PostgreSQL cluster
logs/             diykey.log (rotated), postgres.log
restart           touch to restart DIY-Key
```

1. **Pick a free IP** on the console's network and add it to the interface (not persistent):
   ```bash
   sudo ifconfig en0 alias 192.0.2.18 255.255.255.255
   ```
2. **Write `~/.diy-key/<name>/diykey.env`** from `scripts/diykey.env.example`: a new MAC
   (`FCECDA` + 6 random hex digits), device ID and GUID (`uuidgen | tr A-Z a-z`), the bind IP and
   the console address. Keep the identity fixed once adopted.
3. **Create the instance's PostgreSQL** (listens on the bind IP, admits only the console over SSL):
   ```bash
   DIYKEY_INSTANCE=<name> scripts/diykey_pg_init.sh
   ```

### Settings (`diykey.env`)

| Key | Purpose |
|---|---|
| `DIYKEY_IFACE`, `DIYKEY_MAC`, `DIYKEY_DEVICE_ID`, `DIYKEY_GUID`, `DIYKEY_HOSTNAME` | Identity (required except hostname) |
| `DIYKEY_BIND_IP` | The instance's own address: listeners, outbound ucp4 and nl-search, PostgreSQL |
| `DIYKEY_CONSOLE_HOST` | Console address (tokenless reconnect, PostgreSQL access rule) |
| `DIYKEY_RAM_MODEL` | COCO ONNX detector for tags, verification and crops; unset = placeholder results |
| `DIYKEY_CLIP=1` | CLIP ViT-L/14 embeddings: per-object rows for search, query embeddings |
| `DIYKEY_RAM_TAGS` | Path to Protect's `ram_tags.csv` (query tags; a small built-in list otherwise) |
| `DIYKEY_NL_SEARCH` | `0` to not open the nl-search connection (default on) |
| `DIYKEY_CLIP_TAG_BLEND` | Weight of the tag text blended into stored embeddings (default 0.1, 0 = off) |
| `DIYKEY_FW_VERSION`, `DIYKEY_DISCOVERY_FW_VERSION` | Factory firmware; a simulated update pins newer in `state/fw_version.json` |
| `DIYKEY_FW_DIR`, `DIYKEY_FW_DOWNLOAD`, `DIYKEY_FAKE_UPDATE_S` | Where offered firmware is kept (default `~/.diy-key/firmware`), whether, and how long the fake install takes |
| `DIYKEY_PG`, `DIYKEY_PG_BIN` | `0` to run without the instance's PostgreSQL; Homebrew `postgresql@17` bin dir |
| `DIYKEY_DEBUG`, `DIYKEY_LOG_MAX_MB`, `DIYKEY_LOG_KEEP` | Logging |

## Run

```bash
DIYKEY_INSTANCE=<name> scripts/diykey_service.sh      # in a terminal; Ctrl-C stops it and its PostgreSQL
touch ~/.diy-key/<name>/restart                        # restart with new code/settings
tail -f ~/.diy-key/<name>/logs/diykey.log
```

The service refuses to start until the bind IP is on the interface, starts the instance's
PostgreSQL, restarts DIY-Key if it exits, and exports the PostgreSQL socket so Protect's rotated
database password is applied. Then click **Adopt** on the AI Key in Protect.

Not yet automatic: nothing (alias, PostgreSQL, service) survives a reboot; a launchd agent is on
the backlog.

## Check it works

- `logs/diykey.log`: `WSS CONNECTED`, `nl-search connected to wss://<console>:7443/wss/nl-search/v1`,
  and per event `clip for event …`, `event …: <description>`, `POST …/recognize-anything -> 200`.
- Search rows: `psql -h ~/.diy-key/<name>/pg -d unifi-protect -c 'select count(*) from "ramDetections"'`.
- Find Anything: `NL_PARSE '<query>' -> tags … types … embed 768` in the log. Protect caches each
  query's reply for 5 minutes (every repeat resets it), so use new wording after a change.
- On the console: `/srv/unifi-protect/logs/aiprocessors.log`, `aiprocessors.db.log`, `ds.log`.

## Move to another Mac

**New pairing** (a fresh AI Key on the console; e.g. moving to a replacement Mac):

1. In Protect, **Unadopt/Forget** the old AI Key while it is still online, then stop the old
   service (Ctrl-C in its terminal stops DIY-Key and its PostgreSQL). Nothing in
   `~/.diy-key/<name>/` needs to come along: a new pairing gets a new identity, cert and database.
2. On the new Mac: clone, check out `main`, install the prerequisites and build (above).
3. Copy the files that are not in the repo (vendor material or weights, never committed):
   - the detector ONNX (e.g. `yolov9-s-320.onnx`) → path in `DIYKEY_RAM_MODEL`;
   - Protect's `ram_tags.csv` → path in `DIYKEY_RAM_TAGS`;
   - optional: `~/.diy-key/firmware/` (only to skip re-downloading offered firmware);
   - the CLIP weights download again on first use (or copy `~/.cache/huggingface/hub/`).
4. Configure an instance (above) with a **new** MAC, device ID, GUID and a free IP; don't reuse
   the old identity, or the console may treat it as the old, pinned device.
5. Run, then adopt in Protect. Retroactive processing starts again for the new AI Key record.

**Same identity** (keep the adoption): stop the old instance, copy all of `~/.diy-key/<name>/`
(`state/` holds the cert Protect pinned, `pg/` the database Protect migrated), keep
`DIYKEY_BIND_IP` (Protect remembers the address) or update it in Protect, and fix the paths in
`diykey.env`.

## Notes

- Firmware Protect offers is kept in `~/.diy-key/firmware` with its md5; never commit vendor
  firmware. `<aikey-fw>/fetch.sh` (outside the repo) fetches any version from
  `fw-update.ui.com` the way Protect does.
- PostgreSQL statement logging may be on for debugging:
  `psql -h ~/.diy-key/<name>/pg -d postgres -c "ALTER SYSTEM RESET log_statement" -c "SELECT pg_reload_conf()"`.
