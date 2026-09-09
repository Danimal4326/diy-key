import urllib.request, ssl, json, http.cookiejar, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pi_ssh

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj),
                                      urllib.request.HTTPSHandler(context=ctx))
body = json.dumps({"username": "Token", "password": "REDACTED"}).encode()
req = urllib.request.Request("https://192.0.2.5/api/auth/login", data=body, method="POST",
                              headers={"Content-Type": "application/json"})
resp = opener.open(req, timeout=10)
csrf = resp.headers.get("X-Updated-Csrf-Token")

AIPROC_ID = "6a99e293016cf703e4042e29"

last_state = None
last_log_line_count = 0

print("watching... (ctrl-c to stop)", flush=True)
start = time.time()
while time.time() - start < 90:
    try:
        req = urllib.request.Request(f"https://192.0.2.5/proxy/protect/api/aiprocessors/{AIPROC_ID}",
                                      headers={"X-Csrf-Token": csrf})
        r = opener.open(req, timeout=5)
        d = json.loads(r.read())
        snapshot = {k: d.get(k) for k in ["state","canAdopt","isAttemptingToConnect","isAdopting",
                                            "isAdopted","isProvisioned","connectionHost","lastSeen",
                                            "connectedSince","name"]}
        if snapshot != last_state:
            print(f"[{time.strftime('%H:%M:%S')}] NVR state change: {snapshot}", flush=True)
            last_state = snapshot
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] NVR poll error: {e}", flush=True)

    try:
        rc, out, err = pi_ssh.run("tail -n 200 /root/aikey-emu/run_all.log", timeout=10)
        lines = out.splitlines()
        if len(lines) > last_log_line_count:
            for line in lines[last_log_line_count:]:
                print(f"[PI LOG] {line}", flush=True)
            last_log_line_count = len(lines)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] PI poll error: {e}", flush=True)

    time.sleep(1)
print("done watching")
