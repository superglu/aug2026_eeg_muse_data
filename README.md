# Muse 2 real-time EEG

Stream, record, and analyze real-time EEG from a [Muse 2](https://choosemuse.com) headset in Python, using [muselsl](https://github.com/alexandrebarachant/muse-lsl), the [lab streaming layer](https://labstreaminglayer.org) (LSL), and [MNE-Python](https://mne.tools).

## Architecture

muselsl runs a **bridge process** that connects to the headset over Bluetooth and publishes its data as an LSL stream on your machine; any number of consumer processes then subscribe to that stream simultaneously. The stream itself is ephemeral (small in-memory buffers, nothing persisted) — recording is just one more consumer that writes to disk.

This two-process design is deliberate: the exclusive, fragile Bluetooth link lives in its own process so consumers can crash and restart freely, and being on LSL keeps the setup compatible with the standard neuroscience tooling ecosystem (LabRecorder/XDF, MNE, `mne-lsl`, marker streams for experiments).

```
Muse 2 --BLE--> stream_eeg.py --LSL--> print_eeg.py / plot_live.py / record_eeg.py / your code
                                                                         |
                                                                         v
                                                            data/*.csv --> analyze_session.py (MNE)
```

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Power on the headset (hold the button until the LEDs pulse) and make sure it is **not** connected to the Muse app or any other program — it accepts only one Bluetooth connection at a time.

**Terminal 1** — start the Bluetooth-to-LSL bridge and leave it running:

```sh
python src/stream_eeg.py
```

**Terminal 2** — subscribe to the stream with any (or several) of:

```sh
python src/print_eeg.py                  # raw electrode values + band powers, once per second
python src/plot_live.py                  # live scrolling plot of all four channels
python src/record_eeg.py --user gary     # record 60 s to data/gary_<timestamp>.csv (--duration to change, 0 = until Ctrl+C)
```

Analyze a recording offline:

```sh
python src/analyze_session.py data/eeg_<timestamp>.csv
```

This loads the session into MNE with proper electrode positions, bandpass filters 1–50 Hz, prints relative band powers (delta/theta/alpha/beta/gamma), and opens MNE's power-spectrum and raw-trace viewers (`--no-plot` to skip the windows).

`stream_eeg.py` accepts `--address <MAC or UUID>` to target a specific headset; without it, it connects to the first Muse found. The muselsl CLI is also available directly: `muselsl list`, `muselsl stream`, `muselsl view`.

## Checking signal quality

Watch `plot_live.py` while wearing the headset:

- **Good contact**: a trace settles from full-height noise into a tight band (roughly ±50–100 µV) within ~10 s. A channel stuck railing at ±1000 µV has no skin contact — reposition it; slightly wetting the skin helps a lot with the ear electrodes.
- **Blink test**: hard blinks produce sharp spikes on AF7/AF8 (the forehead channels).
- **Jaw clench test**: clenching produces broadband noise bursts, strongest on TP9/TP10 (the ear channels).

If the plot reacts instantly to blinks and clenches, the whole pipeline is live.

## Collecting data from multiple users

For each subject:

1. Fit the headset and check contact quality with `plot_live.py` (see below) — don't record until all four channels have settled.
2. Record one minute: `python src/record_eeg.py --user <name>`.
3. Recordings land in `data/<name>_<timestamp>.csv`, one file per session.

The `data/` directory is **gitignored** — recordings are shared via Google Drive, not the repo. Upload after a collection session with [rclone](https://rclone.org) (one-time setup: `brew install rclone && rclone config create gdrive drive scope=drive`):

```sh
rclone copy data/ "gdrive:muse-eeg-data 2026-08-09"
```

Collaborators drop the CSVs into their own `data/` directory to analyze with `analyze_session.py`.

## Headset behavior worth knowing

- The Muse only advertises over Bluetooth when nothing is connected to it (LEDs pulsing = ready to pair).
- If it loses its connection while not being worn, it goes to sleep within minutes and disappears from scans — press the power button to wake it, then restart the bridge.
- macOS prompts for Bluetooth permission for your terminal on first use; grant it in System Settings → Privacy & Security → Bluetooth if scanning finds nothing.

## Notes

- Channels: TP9 (left ear), AF7 (left forehead), AF8 (right forehead), TP10 (right ear), plus an AUX input if you attach an electrode. Sampling rate: 256 Hz.
- Recordings are one CSV per session in `data/` (gitignored): an `lsl_timestamp` column plus one column per channel, in µV.
- To consume the stream in your own code, resolve it with `pylsl` — see the top of `src/print_eeg.py`.
- muselsl can also publish PPG, accelerometer, and gyro streams: `muselsl stream --ppg --acc --gyro`.
- A good first experiment: record 60 s with eyes open for the first half and closed for the second — alpha power (8–13 Hz) should visibly increase with eyes closed.

## Cleaning recordings with ZUNA1.1

[ZUNA1.1](https://huggingface.co/Zyphra/ZUNA1.1) (Zyphra's open-weight EEG foundation model, Apache 2.0) denoises recordings, reconstructs bad or missing channels, and can upsample to larger montages. It runs locally as a Python library — weights (~1.5 GB) auto-download from Hugging Face on first use.

```sh
python clean_eeg.py                              # clean every .fif in fif_in/
python clean_eeg.py --repair-channels Cz T3      # also fully reconstruct named channels
python clean_eeg.py --target-channels 64         # upsample the montage to 64 channels
python clean_eeg.py --bad-segments 5:6 10:11:C3  # reconstruct time spans (start:end[:channel])
```

- Inputs are `.fif` files in `fif_in/`; cleaned files land in `fif_out/full_reconstruction/` (model output everywhere) and `fif_out/hybrid/` (original signal, model output only where inferred), with diagnostic overlays in `figures/`. All three directories are gitignored data.
- Channels/spans already marked bad in the file (MNE `info['bads']`, `BAD_` annotations) are reconstructed automatically, in union with the flags above.
- Convert numpy arrays to `.fif` with `eeg_io.numpy_to_fif()` — electrode positions are required (the model predicts from scalp coordinates), and segments must be 0.5–30 s.
- On this Mac inference runs on CPU (`--gpu-device ""`, the default here); `--gpu-device 0` etc. is for CUDA machines.
- **Research use only** — Zyphra explicitly disclaims medical/clinical validity.

## Roadmap

- [x] Bluetooth → LSL bridge, live console + plot consumers
- [x] Hardware smoke test (bridge connects, samples flow at 256 Hz)
- [x] CSV session recorder + MNE analysis (validated on synthetic 10 Hz alpha data)
- [ ] On-head signal-quality pass (settle / blink / jaw-clench checks)
- [ ] First real recording: eyes-open vs eyes-closed alpha comparison
- [ ] Marker stream publisher for stimulus/event timestamps, recorded alongside EEG
- [ ] Optional: LabRecorder + XDF for multi-stream recordings once markers exist
