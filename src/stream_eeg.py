"""Stream real-time EEG data from a Muse 2 to the console.

Connects to the headset over native Bluetooth (no dongle needed), then
prints the latest raw sample for each electrode plus average band powers
once per second.

Usage:
    python src/stream_eeg.py                # connect to first Muse found
    python src/stream_eeg.py --name Muse-ABCD   # connect to a specific headset
"""

import argparse
import time

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter

BOARD_ID = BoardIds.MUSE_2_BOARD
CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Muse 2 EEG to the console")
    parser.add_argument("--name", default="", help="Device name, e.g. Muse-ABCD (default: first Muse found)")
    parser.add_argument("--mac", default="", help="MAC address of the headset (optional)")
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.serial_number = args.name
    params.mac_address = args.mac

    board = BoardShim(BOARD_ID, params)
    sampling_rate = BoardShim.get_sampling_rate(BOARD_ID)
    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)

    print("Searching for Muse 2 (make sure it is on and not paired to another app)...")
    board.prepare_session()
    board.start_stream()
    print(f"Connected. Streaming at {sampling_rate} Hz. Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
            data = board.get_current_board_data(4 * sampling_rate)
            if data.shape[1] < sampling_rate:
                continue

            latest = [data[ch][-1] for ch in eeg_channels[: len(CHANNEL_NAMES)]]
            raw = "  ".join(
                f"{name}: {value:8.2f} uV" for name, value in zip(CHANNEL_NAMES, latest)
            )

            bands, _ = DataFilter.get_avg_band_powers(data, eeg_channels, sampling_rate, True)
            powers = "  ".join(f"{name}: {value:.3f}" for name, value in zip(BAND_NAMES, bands))

            print(raw)
            print(f"  band powers -> {powers}\n")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
