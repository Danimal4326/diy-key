"""Real RK3588 hardware stats for the `systemMetrics` ucp4 RPC.

Field-name shape is NOT confirmed from a live capture (the real request/
response bodies were both empty `{}` -- the console asked once when the
System Performance tab was opened, we had no handler yet, and it got an
empty reply). This shape is our best guess, adapted from the commented-out
`ConsoleStats`/`CpuStats`/`RamStats`/`StorageStats` messages found in the
real uaibd source (grpc_server.py) -- camelCased to match every other
confirmed ucp4 RPC's JSON convention (e.g. `changeAiInferAgentSettings`'s
`deepModeSupported`, `updateLcmSettings`'s `nightMode`). If the Protect UI
still shows blank/zero after this deploys, the shape needs correcting --
ask the user to reopen the System Performance tab and report what (if
anything) shows.
"""
import glob
import os
import time

THERMAL_ZONE_TYPES = {}


def _load_thermal_zone_types():
    global THERMAL_ZONE_TYPES
    if THERMAL_ZONE_TYPES:
        return THERMAL_ZONE_TYPES
    zones = {}
    for type_path in glob.glob("/sys/class/thermal/thermal_zone*/type"):
        try:
            with open(type_path) as f:
                name = f.read().strip()
            zone_dir = os.path.dirname(type_path)
            zones[name] = os.path.join(zone_dir, "temp")
        except OSError:
            continue
    THERMAL_ZONE_TYPES = zones
    return zones


def _read_temp_c(zone_type: str):
    zones = _load_thermal_zone_types()
    path = zones.get(zone_type)
    if not path:
        return None
    try:
        with open(path) as f:
            millidegrees = int(f.read().strip())
        return round(millidegrees / 1000.0, 1)
    except (OSError, ValueError):
        return None


def _read_cpu_load_percent(sample_interval=0.15):
    """Overall CPU load %, sampled from /proc/stat over a short window."""
    def read_stat_line():
        with open("/proc/stat") as f:
            parts = f.readline().split()
        vals = [int(x) for x in parts[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals)
        return idle, total

    try:
        idle1, total1 = read_stat_line()
        time.sleep(sample_interval)
        idle2, total2 = read_stat_line()
        d_idle = idle2 - idle1
        d_total = total2 - total1
        if d_total <= 0:
            return 0.0
        return round(100.0 * (1.0 - d_idle / d_total), 1)
    except (OSError, ValueError, IndexError):
        return None


def _read_ram_kb():
    total_kb = avail_kb = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
    except (OSError, ValueError):
        pass
    return total_kb, avail_kb


def _read_storage_bytes(path="/"):
    try:
        st = os.statvfs(path)
        total = st.f_frsize * st.f_blocks
        free = st.f_frsize * st.f_bavail
        return total, total - free
    except OSError:
        return None, None


def _read_gpu_load_percent():
    for path in glob.glob("/sys/class/devfreq/*.gpu/load"):
        try:
            with open(path) as f:
                raw = f.read().strip()  # e.g. "12@300000000Hz"
            return float(raw.split("@")[0])
        except (OSError, ValueError, IndexError):
            continue
    return None


def _read_npu_load_percent():
    path = "/sys/kernel/debug/rknpu/load"
    try:
        with open(path) as f:
            raw = f.read().strip()  # "NPU load:  Core0:  0%, Core1:  0%, Core2:  0%,"
    except OSError:
        return None
    loads = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if "%" in chunk and ":" in chunk:
            try:
                loads.append(float(chunk.split(":")[-1].replace("%", "").strip()))
            except ValueError:
                continue
    if not loads:
        return None
    return round(sum(loads) / len(loads), 1)


def collect_system_metrics() -> dict:
    cpu_load = _read_cpu_load_percent()
    cpu_temp = _read_temp_c("soc-thermal")
    gpu_load = _read_gpu_load_percent()
    gpu_temp = _read_temp_c("gpu-thermal")
    npu_load = _read_npu_load_percent()
    npu_temp = _read_temp_c("npu-thermal")
    total_kb, avail_kb = _read_ram_kb()
    storage_total, storage_used = _read_storage_bytes()

    return {
        "cpu": {"load": cpu_load, "temperature": cpu_temp},
        "gpu": {"load": gpu_load, "temperature": gpu_temp},
        "npu": {"load": npu_load, "temperature": npu_temp},
        "ram": {"totalKilobytes": total_kb, "availableKilobytes": avail_kb},
        "storage": {"totalBytes": storage_total, "usedBytes": storage_used},
    }


if __name__ == "__main__":
    import json
    print(json.dumps(collect_system_metrics(), indent=2))
