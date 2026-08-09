# Muse 2 real-time EEG

Stream real-time EEG data from a [Muse 2](https://choosemuse.com) headset in Python, using [BrainFlow](https://brainflow.org) over native Bluetooth (no dongle required).

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Turn on the Muse 2 and make sure it is **not** connected to the Muse app or any other program (it only accepts one Bluetooth connection at a time).

Console stream — prints raw electrode values and average band powers once per second:

```sh
python src/stream_eeg.py
```

Live scrolling plot of all four channels:

```sh
python src/plot_live.py
```

Both scripts accept `--name Muse-XXXX` to target a specific headset (the name is printed on the Muse app's device screen); without it they connect to the first Muse found.

## Notes

- Channels: TP9, AF7, AF8, TP10 (plus a right-ear AUX input BrainFlow exposes if you attach an electrode).
- Sampling rate: 256 Hz.
- macOS will prompt for Bluetooth permission for your terminal the first time you run a script; grant it in System Settings → Privacy & Security → Bluetooth if the connection fails.
- To access raw data as a NumPy array in your own code, see `board.get_current_board_data()` / `board.get_board_data()` in either script.
