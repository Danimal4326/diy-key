<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# DIY-Key in Docker / Podman (Linux)

One container runs DIY-Key and the PostgreSQL + pgvector it needs. The console connects into
that database (`:5432`) to deliver AI results, and rotates the `unifi-protect` role's password
over ucp4 (`changeUserPassword`), which `ucp4_client.py` applies with `psql` over
`/var/run/postgresql` as the `postgres` OS user. A second container would have to share that
socket; one container keeps the Linux code path unchanged.

```bash
docker build -f docker/Dockerfile -t diykey .                      # ~1.2 GB
docker build -f docker/Dockerfile --build-arg TORCH=1 -t diykey .  # + CPU torch/open_clip
```

`TORCH=1` is only needed for Find Anything (`DIYKEY_CLIP`, `DIYKEY_NL_SEARCH`) and OSNet re-ID.
It installs torch from PyTorch's CPU index; `requirements.txt`'s plain `torch` would pull the
multi-GB CUDA build.

## Running it

Give it its own LAN address, as the systemd instances get a macvlan child: the console must
reach `:8080` (adoption) and `:5432`, and discovery is UDP broadcast on `:10001`. With Podman
that needs a rootful macvlan network. A compose service:

```yaml
services:
  diykey:
    build: {context: /path/to/diy-key, dockerfile: docker/Dockerfile}
    image: localhost/diykey:latest
    environment:
      - DIYKEY_CONSOLE_HOST=<console-ip>
      - DIYKEY_MAC=FCECDA......        # = mac_address below; keep fixed
      - DIYKEY_DEVICE_ID=<uuid4>       # keep fixed
      - DIYKEY_GUID=<uuid4>            # keep fixed
    volumes: ["/srv/diykey:/data"]
    cap_add: [NET_ADMIN, NET_RAW]
    networks:
      lan: {ipv4_address: <free LAN IP>, mac_address: "FC:EC:DA:..:..:.."}
    healthcheck: {test: ["CMD", "/app/healthcheck.sh"], interval: 30s, start_period: 60s}
    restart: always
networks:
  lan:
    driver: macvlan
    driver_opts: {parent: <host NIC or bridge>}
    ipam: {config: [{subnet: <lan>/24, gateway: <router>}]}
```

Podman's default (OCI) image format drops the Dockerfile `HEALTHCHECK`, so set it in compose.

## What `/data` holds

| Path | Contents |
|---|---|
| `pg/` | The results database. Made on first start: `initdb`, a self-signed server cert (the console requires SSL but doesn't validate it), role/db `unifi-protect` (password `unifi-protect`, SUPERUSER for the console's `CREATE EXTENSION vector`), the `vector` extension. Later starts reuse it, so a rotated password is kept. |
| `state/` | `adopt_state.json`, `server.crt`/`server.key`: the adopted identity |
| `firmware/` | Firmware Protect offers (`DIYKEY_FW_DIR`) |
| `cache/` | open_clip / Hugging Face / fast-alpr downloads (`XDG_CACHE_HOME`, `HF_HOME`, `TORCH_HOME`) |
| `models/` | Put model files here and point the `DIYKEY_*_MODEL` / `DIYKEY_FACE_MODELS` settings at them |

`DIYKEY_PG_ALLOW` (default `samenet`) is the `pg_hba.conf` address the console's `hostssl`
login is accepted from; it only applies when the cluster is first created.

## Checked (Podman 5.8, Fedora host, 2026-11-10)

On a test network with the console pointed at 127.0.0.1: the cluster and role are created and
`run_all.py` starts its five components; an SSL login as `unifi-protect` works (pgvector
0.8.0) and a non-SSL one is refused; `/api/info` reports `sysid 0xa5f0`; `_pg_password_plan`
rotates the password (old one rejected, new one accepted) and the rotated password survives a
restart; `podman stop` exits 0 with PostgreSQL shut down cleanly. Not yet adopted by a console
in this form.
