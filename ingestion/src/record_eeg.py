"""Record the Muse EEG LSL stream to a CSV file.

Run src/stream_eeg.py in another terminal first. Records 60 seconds by
default (use --duration, or --duration 0 to run until Ctrl+C), writing
one row per sample with the LSL timestamp into data/<user>_<timestamp>.csv.

Usage:
    python src/record_eeg.py --user gary          # 60 s for subject "gary"
    python src/record_eeg.py --user ada --duration 120
    python src/record_eeg.py --out data/rest.csv
"""

import argparse
import csv
import time
from pathlib import Path

from pylsl import StreamInlet, local_clock, resolve_byprop

CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10", "AUX"]

# pull_chunk returning nothing is normal jitter for a moment and a dead headset after
# this long, so a dropped Bluetooth link stops the recording instead of padding the
# wall clock with silence (which --duration 0 would otherwise do forever).
QUIET_TIMEOUT = 5.0

# A block that kept fewer than this share of sampling_rate * duration samples is not a
# usable block: the missing time is a dropout, and the CSV looks complete without it.
MIN_SAMPLE_FRACTION = 0.9


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Muse 2 EEG from LSL to CSV")
    parser.add_argument("--user", default="anon", help="Subject name, used in the filename (default: anon)")
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds to record (default: 60; 0 = until Ctrl+C)")
    parser.add_argument("--out", default=None, help="Output file (default: data/<user>_<timestamp>.csv)")
    args = parser.parse_args()
    if args.duration == 0:
        args.duration = None

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
        out_path = Path("data") / f"{args.user}_{stamp}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target = f"{args.duration:.0f} s" if args.duration else "until Ctrl+C"
    print(f"Connected ({sampling_rate:.0f} Hz). Recording {target} to {out_path}.")

    n_samples = 0
    started = local_clock()
    last_sample_at = started
    next_report = 5.0
    went_quiet = False
    try:
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lsl_timestamp"] + CHANNEL_NAMES)
            while True:
                chunk, timestamps = inlet.pull_chunk(timeout=1.0)
                for ts, sample in zip(timestamps, chunk):
                    writer.writerow([f"{ts:.6f}"] + [f"{v:.3f}" for v in sample])
                n_samples += len(chunk)

                now = local_clock()
                if chunk:
                    last_sample_at = now
                elapsed = now - started
                if elapsed >= next_report:
                    print(f"  {elapsed:6.1f} s  {n_samples} samples")
                    next_report += 5.0
                if now - last_sample_at >= QUIET_TIMEOUT:
                    went_quiet = True
                    break
                if args.duration is not None and elapsed >= args.duration:
                    break
    except KeyboardInterrupt:
        pass

    elapsed = local_clock() - started
    print(f"\nSaved {n_samples} samples ({elapsed:.1f} s) to {out_path}")

    # An incomplete block has to fail loudly: the CSV is a plausible-looking short
    # recording, and averaging it in with the full ones is exactly the contamination
    # the collection checklist says must be flagged rather than silently absorbed.
    problems = []
    if went_quiet:
        problems.append(f"the stream went quiet for {QUIET_TIMEOUT:.0f} s (headset or bridge dropped)")
    if args.duration is not None and sampling_rate > 0:
        expected = sampling_rate * args.duration
        if n_samples < MIN_SAMPLE_FRACTION * expected:
            problems.append(f"kept {n_samples} of the ~{expected:.0f} samples "
                            f"{args.duration:.0f} s at {sampling_rate:.0f} Hz should yield "
                            f"({n_samples / expected:.0%})")
    if problems:
        raise SystemExit(f"\nINCOMPLETE RECORDING: {'; '.join(problems)}.\n"
                         f"{out_path} is short — re-record this block before analysing it.")


if __name__ == "__main__":
    main()
