"""Read the Muse EEG LSL stream and print samples to the console.

Run src/stream_eeg.py in another terminal first. Prints the latest raw
electrode values plus per-band power averages once per second.

Usage:
    python src/print_eeg.py
"""

import numpy as np
from pylsl import StreamInlet, resolve_byprop

CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}


def band_powers(window: np.ndarray, sampling_rate: float) -> dict:
    """Average power per frequency band across channels, from a (samples, channels) window."""
    detrended = window - window.mean(axis=0)
    freqs = np.fft.rfftfreq(len(detrended), d=1.0 / sampling_rate)
    psd = np.abs(np.fft.rfft(detrended, axis=0)) ** 2
    return {
        name: psd[(freqs >= lo) & (freqs < hi)].mean()
        for name, (lo, hi) in BANDS.items()
    }


def main() -> None:
    print("Looking for an EEG stream (run src/stream_eeg.py first)...")
    streams = resolve_byprop("type", "EEG", timeout=30)
    if not streams:
        raise SystemExit("No EEG stream found after 30 s.")

    inlet = StreamInlet(streams[0], max_chunklen=12)
    sampling_rate = inlet.info().nominal_srate()
    print(f"Connected. Streaming at {sampling_rate:.0f} Hz. Ctrl+C to stop.\n")

    window_len = int(2 * sampling_rate)
    buffer: list[list[float]] = []
    last_print = None

    try:
        while True:
            chunk, timestamps = inlet.pull_chunk(timeout=1.0)
            if not chunk:
                continue
            buffer.extend(chunk)
            buffer = buffer[-window_len:]

            if last_print is not None and timestamps[-1] - last_print < 1.0:
                continue
            last_print = timestamps[-1]

            latest = chunk[-1]
            raw = "  ".join(
                f"{name}: {value:8.2f} uV" for name, value in zip(CHANNEL_NAMES, latest)
            )
            print(raw)

            if len(buffer) >= window_len:
                powers = band_powers(np.array(buffer)[:, : len(CHANNEL_NAMES)], sampling_rate)
                line = "  ".join(f"{name}: {value:9.2f}" for name, value in powers.items())
                print(f"  band powers -> {line}\n")
    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == "__main__":
    main()
