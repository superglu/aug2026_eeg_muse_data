"""Record the Muse EEG LSL stream to a CSV file.

Run src/stream_eeg.py in another terminal first. Writes one row per
sample with the LSL timestamp, into data/ by default. Stop with Ctrl+C
or use --duration.

Usage:
    python src/record_eeg.py                     # record until Ctrl+C
    python src/record_eeg.py --duration 60       # record 60 seconds
    python src/record_eeg.py --out data/rest.csv
"""

import argparse
import csv
import time
from pathlib import Path

from pylsl import StreamInlet, local_clock, resolve_byprop

CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10", "AUX"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Muse 2 EEG from LSL to CSV")
    parser.add_argument("--duration", type=float, default=None, help="Seconds to record (default: until Ctrl+C)")
    parser.add_argument("--out", default=None, help="Output file (default: data/eeg_<timestamp>.csv)")
    args = parser.parse_args()

    print("Looking for an EEG stream (run src/stream_eeg.py first)...")
    streams = resolve_byprop("type", "EEG", timeout=30)
    if not streams:
        raise SystemExit("No EEG stream found after 30 s.")

    inlet = StreamInlet(streams[0], max_chunklen=12)
    sampling_rate = inlet.info().nominal_srate()

    if args.out:
        out_path = Path(args.out)
    else:
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        out_path = Path("data") / f"eeg_{stamp}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Connected ({sampling_rate:.0f} Hz). Recording to {out_path} — Ctrl+C to stop.")

    n_samples = 0
    started = local_clock()
    next_report = 5.0
    try:
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lsl_timestamp"] + CHANNEL_NAMES)
            while True:
                chunk, timestamps = inlet.pull_chunk(timeout=1.0)
                for ts, sample in zip(timestamps, chunk):
                    writer.writerow([f"{ts:.6f}"] + [f"{v:.3f}" for v in sample])
                n_samples += len(chunk)

                elapsed = local_clock() - started
                if elapsed >= next_report:
                    print(f"  {elapsed:6.1f} s  {n_samples} samples")
                    next_report += 5.0
                if args.duration is not None and elapsed >= args.duration:
                    break
    except KeyboardInterrupt:
        pass

    elapsed = local_clock() - started
    print(f"\nSaved {n_samples} samples ({elapsed:.1f} s) to {out_path}")


if __name__ == "__main__":
    main()
