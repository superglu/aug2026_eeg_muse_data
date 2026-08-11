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
import contextlib
import subprocess
import sys
import threading
import time
from queue import Empty, Queue

# Every line the worker itself logs carries this prefix, and nothing else in the worker's
# output may: muselsl's own prints embed device-supplied text (BLE names are arbitrary
# bytes, embedded newlines included), so they are escaped and re-tagged before being
# forwarded. No text is trusted to mean "streaming" either — muselsl prints
# "Streaming EEG..." when the outlet is live, but a device advertising itself as
# "\nStreaming" would print the same thing before any connection exists, so the worker
# confirms the outlet through LSL instead (see _confirm_stream_when_live).
LOG_PREFIX = "[supervisor "
# Tag on muselsl's escaped output, so its origin is visible in the combined log.
UNTRUSTED_PREFIX = "[muselsl] "
# Logged by the worker once it has resolved the live LSL outlet for the connected
# headset. This is the only text that turns the parent's connect watchdog off.
STREAM_CONFIRMED = "STREAM CONFIRMED"
# How often the worker re-checks for its outlet while muselsl is connecting.
OUTLET_POLL_SEC = 2.0


def _quote(text: str) -> str:
    """Escape device-supplied text so it cannot forge lines in the log stream."""
    return repr(str(text))


def _log_line(msg: str) -> str:
    return f"{LOG_PREFIX}{time.strftime('%H:%M:%S')}] {msg}"


def log(msg: str) -> None:
    print(_log_line(msg), flush=True)


def is_streaming_line(line: str) -> bool:
    """True only for the worker's own confirmation that its LSL outlet went live."""
    return line.startswith(LOG_PREFIX) and line.rstrip().endswith(STREAM_CONFIRMED)


class _UntrustedOutput:
    """Line-buffered stdout/stderr proxy for everything muselsl prints.

    muselsl's output carries device-supplied text verbatim, so each line is escaped
    (which also flattens embedded newlines) and tagged before it reaches the parent.
    Nothing that passes through here can look like a supervisor log line, so no
    advertised BLE name can forge one.
    """

    def __init__(self, out):
        self._out = out
        self._buf = ""

    def write(self, chunk: str) -> int:
        self._buf += chunk
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)
        return len(chunk)

    def _emit(self, line: str) -> None:
        self._out.write(f"{UNTRUSTED_PREFIX}{_quote(line)}\n")
        self._out.flush()

    def flush(self) -> None:
        if self._buf:
            self._emit(self._buf)
            self._buf = ""
        self._out.flush()


def _confirm_stream_when_live(address: str, out) -> None:
    """Log STREAM_CONFIRMED once this headset's LSL outlet actually resolves.

    Polling the outlet is the only check no advertising device can spoof: the
    watchdog is disarmed by an outlet that exists on the machine and carries the
    address we chose to connect to, not by a string somebody printed. Writes to the
    real stdout, which the worker keeps out of the muselsl redirect for this reason.
    """
    from pylsl import resolve_byprop

    while True:
        streams = resolve_byprop("type", "EEG", timeout=OUTLET_POLL_SEC)
        if any(address in s.source_id() for s in streams):
            out.write(_log_line(STREAM_CONFIRMED) + "\n")
            out.flush()
            return
        time.sleep(OUTLET_POLL_SEC)


def worker(address: str | None) -> None:
    """One scan+connect+stream cycle. Runs in a child process."""
    from muselsl import list_muses, stream

    untrusted = _UntrustedOutput(sys.stdout)

    log("scanning...")
    with contextlib.redirect_stdout(untrusted), contextlib.redirect_stderr(untrusted):
        muses = list_muses()
    untrusted.flush()
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
    threading.Thread(target=_confirm_stream_when_live, args=(address, sys.stdout), daemon=True).start()
    with contextlib.redirect_stdout(untrusted), contextlib.redirect_stderr(untrusted):
        stream(address)  # blocks until the headset disconnects
    untrusted.flush()
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
