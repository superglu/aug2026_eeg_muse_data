# Muse 2 real-time EEG

Stream real-time EEG data from a [Muse 2](https://choosemuse.com) headset in Python, using [muselsl](https://github.com/alexandrebarachant/muse-lsl) and the [lab streaming layer](https://labstreaminglayer.org) (LSL).

muselsl runs a bridge process that connects to the headset over Bluetooth and publishes its data as an LSL stream on your machine; any number of other processes can then subscribe to that stream.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Turn on the Muse 2 and make sure it is **not** connected to the Muse app or any other program (it only accepts one Bluetooth connection at a time).

**Terminal 1** — start the Bluetooth-to-LSL bridge and leave it running:

```sh
python src/stream_eeg.py
```

**Terminal 2** — subscribe to the stream:

```sh
python src/print_eeg.py    # raw electrode values + band powers, once per second
python src/plot_live.py    # live scrolling plot of all four channels
```

`stream_eeg.py` accepts `--address <MAC or UUID>` to target a specific headset; without it, it connects to the first Muse found. The muselsl CLI is also available directly: `muselsl list`, `muselsl stream`, `muselsl view`.

## Notes

- Channels: TP9, AF7, AF8, TP10 (plus a right-ear AUX input if you attach an electrode).
- Sampling rate: 256 Hz.
- macOS will prompt for Bluetooth permission for your terminal the first time you run the bridge; grant it in System Settings → Privacy & Security → Bluetooth if the connection fails.
- To consume the data in your own code, resolve the stream with `pylsl` — see the top of `src/print_eeg.py`.
- muselsl can also publish PPG, accelerometer, and gyro streams: `muselsl stream --ppg --acc --gyro`.
