#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Run a DIY-Key instance as a root launchd daemon on macOS: started at boot, restarted if it
# exits, independent of any terminal (scripts/diykey_service.sh does the work). Its files are in
# ~/.diy-key/$DIYKEY_INSTANCE/ ("main" when not set; see scripts/diykey_paths.sh) and its job is
# local.diykey.<instance>.
#
#   sudo DIYKEY_INSTANCE=main scripts/macos_launchd.sh install     install it and start it
#   sudo DIYKEY_INSTANCE=main scripts/macos_launchd.sh restart     restart everything
#   sudo DIYKEY_INSTANCE=main scripts/macos_launchd.sh stop        stop it until start or boot
#   sudo DIYKEY_INSTANCE=main scripts/macos_launchd.sh start       start it again
#   sudo DIYKEY_INSTANCE=main scripts/macos_launchd.sh uninstall   stop it and remove the plist
#   DIYKEY_INSTANCE=main scripts/macos_launchd.sh status           launchd's view, last log lines
#   DIYKEY_INSTANCE=main scripts/macos_launchd.sh logs             follow logs/diykey.log
#
# Root because the instance's address (DIYKEY_BIND_IP) is an interface alias, which isn't kept
# across reboots and only root can add, and because macOS's Local Network privacy blocks a user
# agent from LAN devices other than the gateway (a console needn't be the gateway). PostgreSQL
# refuses to run as root, so the service runs it as the owner of its pg/ folder. The instance's
# folder stays yours: edit diykey.env and touch restart (new code, new settings) without sudo.
set -u

die() { echo "macos_launchd: $*" >&2; exit 1; }
[ "$(uname)" = Darwin ] || die "this is for macOS"

CMD=${1:-}
if [ "$(id -u)" -eq 0 ]; then
    [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ] \
        || die "run it with sudo from your own account, so it knows whose home folder to use"
    USER_NAME=$SUDO_USER
else
    USER_NAME=$(id -un)
fi
USER_ID=$(id -u "$USER_NAME")
USER_HOME=$(eval echo "~$USER_NAME")
REPO=$(cd "$(dirname "$0")/.." && pwd)
. "$REPO/scripts/diykey_paths.sh"

LABEL=${DIYKEY_LAUNCHD_LABEL:-local.diykey.$INSTANCE}
PLIST=/Library/LaunchDaemons/$LABEL.plist
DOMAIN=system
LAUNCHD_LOG=$BASE/logs/launchd.log   # the wrapper's own errors only

case "$CMD" in
    install|uninstall|start|stop|restart)
        [ "$(id -u)" -eq 0 ] || die "$LABEL is a root daemon: run this with sudo" ;;
esac

loaded() { launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; }

check_not_running() {
    # This instance only (its state dir). A job just booted out takes a few seconds to stop.
    local dir pids
    dir=$(sed -n 's/^[[:space:]]*DIYKEY_STATE_DIR=//p' "$ENV_FILE" | tail -1 | sed 's/[[:space:]]*#.*//')
    for _ in $(seq 1 30); do
        pids=$(pgrep -f "run_all.py .*--state-dir ${dir:-$DEFAULT_STATE_DIR}( |$)" || true)
        [ -z "$pids" ] && return 0
        sleep 1
    done
    echo "$INSTANCE is already running outside launchd (pid $(echo $pids | tr ' ' ','))." >&2
    die "stop it first (Ctrl-C in its terminal)"
}

write_plist() {
    # The folders stay yours, or PostgreSQL (run as you) couldn't write its log.
    mkdir -p "$(dirname "$LAUNCHD_LOG")"
    chown "$USER_NAME:staff" "$USER_HOME/.diy-key" "$BASE" "$(dirname "$LAUNCHD_LOG")"
    # HOME: the CLIP weights (~/.cache/huggingface) and offered firmware (~/.diy-key/firmware)
    # stay where a terminal run keeps them, not under /var/root. No .pyc files: root's would
    # block a later run as you from replacing them.
    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$REPO/scripts/diykey_service.sh</string>
    </array>
    <key>WorkingDirectory</key><string>$REPO</string>
    <key>EnvironmentVariables</key>
    <dict>
        <!-- ffmpeg comes from Homebrew; launchd's own PATH doesn't have it. -->
        <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>HOME</key><string>$USER_HOME</string>
        <key>DIYKEY_INSTANCE</key><string>$INSTANCE</string>
        <key>SUDO_USER</key><string>$USER_NAME</string>
        <key>SUDO_UID</key><string>$USER_ID</string>
        <key>PYTHONDONTWRITEBYTECODE</key><string>1</string>
    </dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>ThrottleInterval</key><integer>10</integer>
    <!-- The wrapper gives DIY-Key 10 s to stop cleanly before TERM, then stops PostgreSQL. -->
    <key>ExitTimeOut</key><integer>30</integer>
    <key>StandardOutPath</key><string>$LAUNCHD_LOG</string>
    <key>StandardErrorPath</key><string>$LAUNCHD_LOG</string>
</dict>
</plist>
EOF
    plutil -lint "$PLIST" >/dev/null || die "wrote an invalid $PLIST"
    chown root:wheel "$PLIST" && chmod 644 "$PLIST"
}

start() {
    loaded || launchctl bootstrap "$DOMAIN" "$PLIST" || die "launchctl bootstrap failed"
    launchctl enable "$DOMAIN/$LABEL"
}

status() {
    echo "$LABEL (root daemon, files in $BASE):"
    if loaded; then
        launchctl print "$DOMAIN/$LABEL" | grep -E "^\s*(state|pid|last exit code|runs) =" \
            | sed 's/^[[:space:]]*/  /'
    else
        echo "  not loaded"
    fi
    echo "last log lines ($LOG):"
    tail -n 5 "$LOG" 2>/dev/null | cut -c1-200 | sed 's/^/  /'
}

case "$CMD" in
    install)
        [ -r "$ENV_FILE" ] || die "no $ENV_FILE (copy scripts/diykey.env.example)"
        [ -x "$REPO/venv/bin/python" ] || die "no venv at $REPO/venv"
        grep -qE '^[[:space:]]*DIYKEY_BIND_IP=[0-9]' "$ENV_FILE" \
            || die "a daemon needs its own address: set DIYKEY_BIND_IP in $ENV_FILE"
        loaded && launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
        check_not_running
        write_plist
        start
        echo "installed $PLIST; DIY-Key starts now and at every boot"
        sleep 3
        status
        ;;
    uninstall)
        loaded && launchctl bootout "$DOMAIN/$LABEL"
        rm -f "$PLIST"
        echo "removed $LABEL (its files stay in $BASE)"
        ;;
    start)
        [ -f "$PLIST" ] || die "not installed (run: $0 install)"
        start
        status
        ;;
    stop)
        loaded && launchctl bootout "$DOMAIN/$LABEL"
        echo "stopped $LABEL (starts again at boot, or with: $0 start)"
        ;;
    restart)
        loaded || die "not running (run: $0 start)"
        launchctl kickstart -k "$DOMAIN/$LABEL"
        status
        ;;
    status) status ;;
    logs) exec tail -n 50 -F "$LOG" ;;
    *)
        sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
