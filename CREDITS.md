# Credits

The AI Key emulator's protocol code is original to this project. The shared
plumbing imported from the sibling **pi-port** project (also by hutchx86) is:

| File | Origin |
|---|---|
| `diykey/core/config.py` | derived from `piport/config.py` |
| `diykey/core/instance_manager.py` | derived from `piport/instance_manager.py` |
| `diykey/stage1_discovery/beacon.py` | rewritten on `piport/discovery.py` |
| `diykey/stage1_http/adopt_http.py` | state/TLS handling from `piport/http_api.py` |
| `diykey/stage3_wss/ucp4_client.py` | framing/reconnect handling from `piport/ucp4_client.py` |
| `diykey/run_all.py` | supervision policy from `piport/run_all.py` |
| `diykey/tests/*` | adapted from `piport/tests/*` |

Those files carry `SPDX-License-Identifier: AGPL-3.0-or-later` and
`Copyright (C) 2026 hutchx86`. The repository as a whole is licensed
AGPL-3.0-or-later (see `LICENSE`); no vendor code or binaries are redistributed.
