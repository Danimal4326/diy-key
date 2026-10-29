# Credits

The AI Key emulator's protocol code is original to this project. The shared
plumbing imported from the sibling **pi-port** project (also by hutchx86) is:

| File | Origin |
|---|---|
| `aikey-emu/core/config.py` | derived from `piport/config.py` |
| `aikey-emu/core/instance_manager.py` | derived from `piport/instance_manager.py` |
| `aikey-emu/stage1_discovery/beacon.py` | rewritten on `piport/discovery.py` |
| `aikey-emu/stage1_http/adopt_http.py` | state/TLS handling from `piport/http_api.py` |
| `aikey-emu/stage3_wss/ucp4_client.py` | framing/reconnect handling from `piport/ucp4_client.py` |
| `aikey-emu/run_all.py` | supervision policy from `piport/run_all.py` |
| `aikey-emu/tests/*` | adapted from `piport/tests/*` |

Those files carry `SPDX-License-Identifier: AGPL-3.0-or-later` and
`Copyright (C) 2026 hutchx86`, matching pi-port's licence. The rest of this
repository is a private proof-of-concept; no vendor code is redistributed.
