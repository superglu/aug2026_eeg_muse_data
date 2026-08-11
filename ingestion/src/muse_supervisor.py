"""Keep the Muse-to-LSL bridge alive for a whole collection session.

Parent/worker design so a hung Bluetooth connect cannot stall the session:

- The worker (--worker) does one cycle: scan, report what is visible,
  connect, stream until disconnect, exit.
- The parent respawns the worker forever and enforces a connect watchdog:
  if a worker has not reached "Streaming" within --connect-timeout seconds,
  it is killed and respawned. macOS BLE-stack hangs (observed after abrupt
  headset power loss) self-heal instead of requiring a manual restart.

The scan log distinguishes the failure modes:
  - "MUSE VISIBLE at <addr>"  -> advertising; connecting
  - "address changed"         -> macOS reassigned the BLE UUID (e.g. after a
                                 headset reboot); auto-switching
  - "NO MUSE advertising"     -> headset off, asleep, or connected elsewhere
                                 (phone app!) — fix at the headset

Usage:
    python ingestion/src/muse_supervisor.py                     # any Muse
    python ingestion/src/muse_supervisor.py --address <UUID>    # prefer one
"""

import argparse
import subprocess
import sys
import threading
import time
from queue import Empty, Queue

# Printed by muselsl.stream once the LSL outlet is live ("Streaming EEG...").
STREAMING_MARKER = "Streaming"
# Every line the worker itself logs carries this prefix. Those lines quote
# device-supplied text (BLE names/addresses), so the parent must never read a
# stream marker out of them — otherwise a device advertising itself as
# "Streaming" would disable the connect watchdog before any connection exists.
LOG_PREFIX = "[supervisor "


def _quote(text: str) -> str:
    """Escape device-supplied text so it cannot forge lines in the log stream."""
    return repr(str(text))


def log(msg: str) -> None:
    print(f"{LOG_PREFIX}{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_streaming_line(line: str) -> bool:
    """True only for muselsl's own stream announcement, not for worker log lines."""
    return not line.startswith(LOG_PREFIX) and line.lstrip().startswith(STREAMING_MARKER)


def worker(address: str | None) -> None:
    """One scan+connect+stream cycle. Runs in a child process."""
    from muselsl import list_muses, stream

    log("scanning...")
    muses = list_muses()
    if not muses:
        log("NO MUSE advertising — headset is off, asleep, or connected elsewhere (phone app?)")
        return

    names = ", ".join(f"{_quote(m['name'])} at {_quote(m['address'])}" for m in muses)
    log(f"MUSE VISIBLE: {names}")

    if address and not any(m["address"] == address for m in muses):
        log(f"address changed: configured {_quote(address)} not seen — "
            f"switching to {_quote(muses[0]['address'])}")
        address = None
    address = address or muses[0]["address"]

    log(f"connecting to {_quote(address)}")
    stream(address)  # blocks until the headset disconnects; prints "Streaming EEG..."
    log("headset disconnected")


def supervise(address: str | None, connect_timeout: float) -> None:
    while True:
        cmd = [sys.executable, __file__, "--worker"]
        if address:
            cmd += ["--address", address]
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        lines: Queue[str | None] = Queue()
        threading.Thread(target=lambda: ([lines.put(l) for l in child.stdout] and None) or lines.put(None),
                         daemon=True).start()

        started = time.monotonic()
        streaming = False
        while True:
            try:
                line = lines.get(timeout=5)
            except Empty:
                line = ""
            if line is None:  # child exited
                break
            if line:
                print(line, end="", flush=True)
                if is_streaming_line(line):
                    streaming = True
            if not streaming and time.monotonic() - started > connect_timeout:
                log(f"WATCHDOG: no stream after {connect_timeout:.0f} s — killing stuck worker and retrying")
                child.kill()
                break

        child.wait()
        time.sleep(3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-reconnecting Muse-to-LSL bridge with connect watchdog")
    parser.add_argument("--address", default=None, help="Preferred device address (auto-switches if it changes)")
    parser.add_argument("--connect-timeout", type=float, default=90.0,
                        help="Seconds a worker may spend scanning+connecting before it is killed (default 90)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker(args.address)
    else:
        try:
            supervise(args.address, args.connect_timeout)
        except KeyboardInterrupt:
            log("stopped")


if __name__ == "__main__":
    main()
