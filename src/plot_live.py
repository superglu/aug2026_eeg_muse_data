"""Live scrolling plot of Muse 2 EEG channels.

Opens a matplotlib window showing the last few seconds of all four
electrodes, updating in real time.

Usage:
    python src/plot_live.py
    python src/plot_live.py --name Muse-ABCD --window 10
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes
from matplotlib.animation import FuncAnimation

BOARD_ID = BoardIds.MUSE_2_BOARD
CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Live plot of Muse 2 EEG")
    parser.add_argument("--name", default="", help="Device name, e.g. Muse-ABCD")
    parser.add_argument("--window", type=float, default=5.0, help="Seconds of data shown (default 5)")
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.serial_number = args.name

    board = BoardShim(BOARD_ID, params)
    sampling_rate = BoardShim.get_sampling_rate(BOARD_ID)
    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)[: len(CHANNEL_NAMES)]
    num_points = int(args.window * sampling_rate)

    print("Searching for Muse 2 (make sure it is on and not paired to another app)...")
    board.prepare_session()
    board.start_stream()
    print("Connected. Close the plot window to stop.")

    fig, axes = plt.subplots(len(eeg_channels), 1, sharex=True, figsize=(10, 8))
    fig.suptitle("Muse 2 EEG (bandpassed 1-50 Hz)")
    t = np.arange(-num_points, 0) / sampling_rate
    lines = []
    for ax, name in zip(axes, CHANNEL_NAMES):
        (line,) = ax.plot(t, np.zeros(num_points))
        ax.set_ylabel(f"{name}\n(uV)")
        ax.set_ylim(-100, 100)
        lines.append(line)
    axes[-1].set_xlabel("seconds")

    def update(_frame):
        data = board.get_current_board_data(num_points)
        for line, ch in zip(lines, eeg_channels):
            signal = data[ch].copy()
            if signal.size < 2:
                continue
            DataFilter.detrend(signal, DetrendOperations.CONSTANT)
            DataFilter.perform_bandpass(
                signal, sampling_rate, 1.0, 50.0, 4, FilterTypes.BUTTERWORTH, 0
            )
            padded = np.zeros(num_points)
            padded[-signal.size:] = signal[-num_points:]
            line.set_ydata(padded)
        return lines

    _anim = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
