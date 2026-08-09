"""Live scrolling plot of the Muse EEG LSL stream.

Run src/stream_eeg.py in another terminal first. Opens a matplotlib
window showing the last few seconds of all four electrodes.

Usage:
    python src/plot_live.py
    python src/plot_live.py --window 10
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from pylsl import StreamInlet, resolve_byprop

CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Live plot of Muse 2 EEG from LSL")
    parser.add_argument("--window", type=float, default=5.0, help="Seconds of data shown (default 5)")
    args = parser.parse_args()

    print("Looking for an EEG stream (run src/stream_eeg.py first)...")
    streams = resolve_byprop("type", "EEG", timeout=30)
    if not streams:
        raise SystemExit("No EEG stream found after 30 s.")

    inlet = StreamInlet(streams[0], max_chunklen=12)
    sampling_rate = int(inlet.info().nominal_srate())
    num_points = int(args.window * sampling_rate)
    print(f"Connected. Streaming at {sampling_rate} Hz. Close the plot window to stop.")

    data = np.zeros((num_points, len(CHANNEL_NAMES)))

    fig, axes = plt.subplots(len(CHANNEL_NAMES), 1, sharex=True, figsize=(10, 8))
    fig.suptitle("Muse 2 EEG")
    t = np.arange(-num_points, 0) / sampling_rate
    lines = []
    for ax, name, column in zip(axes, CHANNEL_NAMES, data.T):
        (line,) = ax.plot(t, column)
        ax.set_ylabel(f"{name}\n(uV)")
        ax.set_ylim(-200, 200)
        lines.append(line)
    axes[-1].set_xlabel("seconds")

    def update(_frame):
        nonlocal data
        chunk, _ = inlet.pull_chunk(timeout=0.0)
        if chunk:
            samples = np.array(chunk)[:, : len(CHANNEL_NAMES)]
            data = np.vstack([data, samples])[-num_points:]
            centered = data - data.mean(axis=0)
            for line, column in zip(lines, centered.T):
                line.set_ydata(column)
        return lines

    _anim = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
