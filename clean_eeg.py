"""Clean/reconstruct EEG recordings with ZUNA1.1 (Zyphra's EEG foundation model).

Reads every .fif in the input directory, denoises and reconstructs bad or
missing channels, and writes cleaned files to <output_dir>/full_reconstruction/
(model output everywhere) and <output_dir>/hybrid/ (original signal, model
output only on inferred cells), plus diagnostic overlays in the figures dir.

The model reconstructs the UNION of: channels/spans already marked bad in the
file (MNE info['bads'] and BAD_ annotations), --repair-channels,
--target-channels (montage upsampling), and --bad-segments.

Research use only — Zyphra explicitly disclaims medical/clinical validity.
Input segments must be 0.5-30 s, and electrode positions (montage) must be set
on the inputs: the model predicts from scalp coordinates.

Usage:
    python clean_eeg.py                                  # clean everything in fif_in/
    python clean_eeg.py --repair-channels Cz T3          # also fully reconstruct Cz, T3
    python clean_eeg.py --target-channels Fz Pz          # upsample montage with new channels
    python clean_eeg.py --target-channels 64             # upsample to 64 channels
    python clean_eeg.py --bad-segments 5:6 10:11:C3      # mark spans (start:end[:channel]) bad
"""

import argparse
import os

from zuna import reconstruct_fif


def parse_bad_segment(spec: str):
    """'5:6' -> (5.0, 6.0); '10:11:C3' -> (10.0, 11.0, 'C3')"""
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(f"bad segment {spec!r}: use start:end or start:end:channel")
    start, end = float(parts[0]), float(parts[1])
    return (start, end, parts[2]) if len(parts) == 3 else (start, end)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Denoise/reconstruct EEG .fif files with ZUNA1.1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default="fif_in", help="Directory of input .fif files")
    parser.add_argument("--output-dir", default="fif_out", help="Directory for cleaned .fif files")
    parser.add_argument("--figures-dir", default="figures", help="Directory for diagnostic figures")
    parser.add_argument("--gpu-device", default="", help='CUDA device id; "" = CPU (the supported path on this Mac)')
    parser.add_argument("--repair-channels", nargs="+", metavar="CH",
                        help="Channel names to fully reconstruct (in addition to those marked bad in the file)")
    parser.add_argument("--target-channels", nargs="+", metavar="CH_OR_N",
                        help="Upsample montage: channel names to add, or a single integer channel count")
    parser.add_argument("--bad-segments", nargs="+", type=parse_bad_segment, metavar="START:END[:CH]",
                        help="Time spans (seconds) to reconstruct, optionally on a single channel")
    args = parser.parse_args()

    for d in (args.input_dir, args.output_dir, args.figures_dir):
        os.makedirs(d, exist_ok=True)

    kwargs = {}
    if args.repair_channels:
        kwargs["repair_channels"] = args.repair_channels
    if args.target_channels:
        if len(args.target_channels) == 1 and args.target_channels[0].isdigit():
            kwargs["target_channel_count"] = int(args.target_channels[0])
        else:
            kwargs["target_channel_count"] = args.target_channels
    if args.bad_segments:
        kwargs["bad_segments"] = args.bad_segments

    reconstruct_fif(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        gpu_device=args.gpu_device,
        **kwargs,
    )


if __name__ == "__main__":
    main()
