"""Continuously report electrode contact quality from the Muse LSL stream.

Prints one line per second with the rolling 2 s standard deviation per
channel and a verdict (GOOD < 60 uV, ok < 150 uV, else NOISY). Survives
bridge restarts: when the stream vanishes it waits and re-resolves.

Run alongside muse_supervisor.py for the whole session; read the latest
line to know current fit quality instantly.
"""

import time

import numpy as np
from pylsl import StreamInlet, resolve_byprop

CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]
WINDOW = 512  # 2 s at 256 Hz


def verdict(std: float) -> str:
    return "GOOD" if std < 60 else ("ok" if std < 150 else "NOISY")


def main() -> None:
    while True:
        print("[contact] waiting for EEG stream...", flush=True)
        streams = resolve_byprop("type", "EEG", timeout=10)
        if not streams:
            continue
        inlet = StreamInlet(streams[0], max_chunklen=12, recover=False)
        print("[contact] stream connected", flush=True)
        buffer: list[list[float]] = []
        last_line = 0.0
        try:
            while True:
                chunk, timestamps = inlet.pull_chunk(timeout=2.0)
                if not chunk:
                    raise RuntimeError("stream went quiet")
                buffer.extend(chunk)
                buffer = buffer[-WINDOW:]
                now = time.monotonic()
                if len(buffer) >= WINDOW and now - last_line >= 1.0:
                    last_line = now
                    arr = np.array(buffer)[:, : len(CHANNEL_NAMES)]
                    arr = arr - arr.mean(axis=0)
                    stds = arr.std(axis=0)
                    fields = "  ".join(
                        f"{name}:{std:7.1f} {verdict(std):5s}"
                        for name, std in zip(CHANNEL_NAMES, stds)
                    )
                    summary = "ALL-GOOD" if all(verdict(s) == "GOOD" for s in stds) else (
                        "USABLE" if sum(verdict(s) != "NOISY" for s in stds) >= 3 else "POOR")
                    print(f"{time.strftime('%H:%M:%S')}  {fields}  => {summary}", flush=True)
        except Exception as exc:  # noqa: BLE001 - stream dropped; go back to resolving
            print(f"[contact] lost stream ({exc}), re-resolving...", flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main()
