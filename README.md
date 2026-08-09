# Can ZUNA clean up consumer EEG?

Testing [ZUNA1.1](https://huggingface.co/Zyphra/ZUNA1.1) — Zyphra's 380M-parameter EEG foundation model — on two jobs: **denoising what a 4-electrode Muse 2 headband records, and synthesizing the electrodes it doesn't have.** Built on a live Python pipeline: [muselsl](https://github.com/alexandrebarachant/muse-lsl) + [LSL](https://labstreaminglayer.org) for streaming, [MNE-Python](https://mne.tools) for analysis.

## 📊 Final results: `zuna_evaluation_v4.pdf`

**The evaluation deck is [`zuna_evaluation_v4.pdf`](zuna_evaluation_v4.pdf)** (18 slides: study design, reconstruction case studies, spectra, decoding, verdicts). The scoreboard:

| branch | EC−EO alpha (dB) | decode accuracy | verdict (draft) |
|---|---|---|---|
| `raw` | +1.9 to +6.1 | 44.3% ± 27.4 | baseline: effect in, decoding ~chance |
| `denoised` | +3.9 — uniform across channels | **58.0% ± 39.8** | effect survives, compressed · best decoder, unstable |
| `upsampled` | real: +1.9 to +6.0 (≈ raw) · synth: +0.7 to +3.2 | 45.3% ± 27.9 | splice clean · synth carries real physiology |

**What was answered this round** (4 subjects × 60 s eyes-closed + 60 s eyes-open, Muse 2, 2026-08-09):

- **The physiology survives ZUNA.** EC−EO alpha stays positive in every branch and channel — the Berger effect, our built-in ground truth, is preserved by processing.
- **Denoising is aggressive on artifact, conservative on clean data.** Case studies in the deck: ZUNA refuses to reproduce a muscle-noise artifact on AF7 (it has never seen such activity in clean EEG), yet on a clean recording the reconstruction hugs the input nearly sample-for-sample. The trade: per-channel variety compresses to a uniform +3.9 dB, and denoised windows collapse onto their own manifold in the t-SNE.
- **Upsampling is honest twice over.** The 4 real channels pass through matching raw almost exactly, and the synthesized occipitals (O1/O2/Oz/Pz — positions the headband doesn't have) show genuine, if muted, alpha behavior — including visible eyes-closed alpha bursts in the reconstructions.
- **Decoding is the open question, and it's about power, not direction.** Denoised decodes best (58% vs 44% raw) but with ±40-point fold swings; with 8 recordings under GroupKFold, that can't be told apart from luck. The t-SNE shows why: **subject identity dominates alpha features** — decoding a held-out recording is brutal at n=4.
- **Next**: more subjects; per-subject normalization (let eye state, not identity, drive the classifier); confirm the upsampled `n_epochs` logging discrepancy (24 vs 240); compare hybrid vs full on upsampled — the one condition where the output modes truly differ.

Supporting artifacts: t-SNE and band-power figures in `data/figures/`, branch reconstructions in `data/branches/` (gitignored; shared via Drive). The earlier working deck (`zuna_evaluation.pptx`) is kept for the raw figure exports.

## Repo layout

| Path | What it is |
|---|---|
| `zuna_evaluation_v4.pdf` | **Final results deck** |
| `zuna_evaluation.pptx` | Earlier working deck (raw figure exports) |
| `ingestion/src/` | Live streaming: bridge (`stream_eeg.py`), session supervisor, contact watcher, live plot, recorder, first-pass MNE analysis |
| `ingestion/` | `muse_to_fif.py` (CSV→.fif with EO/EC routing), `clean_eeg.py` (standalone ZUNA CLI), `eeg_io.py` helpers |
| `eo_ec_pipeline/` | The evaluation pipeline: branches → features → classification, band-power check, t-SNE |
| `zuna/` | Vendored ZUNA1.1 source (shadows the pip package when running from the repo root) |
| `data/`, `fif_in/`, `fif_out/`, `figures/` | Recordings and model outputs — gitignored; shared via Google Drive |

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Collecting data

Power on the headset — LEDs **pulsing** means advertising and ready; **solid** means something else grabbed it (force-quit the Muse phone app). It accepts one Bluetooth connection at a time, and it sleeps if left disconnected off-head — press the power button to wake it.

Session-length helpers (leave running for a whole multi-participant session; they self-heal across swaps, BLE drops, and connect hangs):

```sh
python ingestion/src/muse_supervisor.py    # auto-reconnecting Bluetooth→LSL bridge with scan diagnostics + connect watchdog
python ingestion/src/contact_watch.py      # prints per-channel fit quality once per second (GOOD/ok/NOISY)
python ingestion/src/plot_live.py          # live scrolling plot of all four channels
```

Per participant — wait until `contact_watch` holds GOOD/USABLE for ~15 consecutive seconds (recordings started on a still-bouncing fit came out unusable):

```sh
python ingestion/src/record_eeg.py --user <name>_open     # 60 s eyes open
python ingestion/src/record_eeg.py --user <name>_closed   # 60 s eyes closed
```

The `_open`/`_closed` suffixes matter: `muse_to_fif.py` routes files to `data/input_eo/` or `data/input_ec/` by filename. Other consumers: `print_eeg.py` (console band powers), `analyze_session.py <csv>` (MNE first-pass: band powers, PSD, filtered traces).

### Signal quality

Good contact settles under ~60 µV std within a minute of wearing. Hard blinks spike AF7/AF8; jaw clenches flood TP9/TP10 — instant proof the pipeline is live. Wetting the skin behind the ears helps the TP electrodes. Historical note from collection day: **this unit's AF7 read noisy on most fits even after cleaning the strip** — treat AF7 with suspicion in analysis.

## Data flow

```
Muse 2 --BLE--> muse_supervisor --LSL--> record_eeg --> data/<name>_<open|closed>_<ts>.csv
   ingestion/muse_to_fif.py (routes by filename) --> data/input_eo/ | data/input_ec/
   eo_ec_pipeline/run_eo_ec_test.py --> data/branches/{EC,EO}/{raw,denoised,upsampled}
                                    --> data/branches/features.csv + classification table
   eo_ec_pipeline/band_power_check.py + latent_tsne.py --> data/figures/*.png
```

Recordings are never committed — share via [rclone](https://rclone.org): `rclone copy data/ "gdrive:muse-eeg-data <date>"` (one-time setup: `brew install rclone && rclone config create gdrive drive scope=drive`).

## Running the evaluation

```sh
python eo_ec_pipeline/run_eo_ec_test.py                       # branches -> features -> EO/EC classification
python eo_ec_pipeline/band_power_check.py --max-per-condition 10
python eo_ec_pipeline/latent_tsne.py
```

Operational notes:

- The four ZUNA branch reconstructions run **sequentially**: ~8 min per denoised stage, ~25–30 min per upsampled stage (~70 min total on an M-series Mac; torch routes to MPS). The real compute runs in a spawned child process — don't judge progress by the parent PID's CPU.
- There is **no resume** — killing the pipeline mid-run redoes everything.
- Set `HF_HUB_OFFLINE=1` after the first run: unauthenticated Hugging Face requests can rate-limit and stall.
- `ingestion/clean_eeg.py` remains the standalone ZUNA CLI for ad-hoc cleaning (`--repair-channels`, `--target-channels`, `--bad-segments`) into `fif_out/`.

## Caveats

- **Research use only** — Zyphra explicitly disclaims medical/clinical validity for ZUNA1.1.
- ZUNA inputs: 0.5–30 s segments; electrode positions required (`eeg_io.numpy_to_fif` attaches standard-1020).
- Muse 2: 256 Hz; TP9 (left ear), AF7/AF8 (forehead), TP10 (right ear), plus AUX. macOS asks for Bluetooth permission for your terminal on first use.
- muselsl can also publish PPG/accelerometer/gyro streams: `muselsl stream --ppg --acc --gyro`.

## Roadmap

- [x] Bluetooth → LSL bridge, live console + plot consumers
- [x] Hardware smoke test (256 Hz, 4 channels, verified live)
- [x] Session recorder + MNE analysis (validated on synthetic 10 Hz alpha)
- [x] Multi-participant collection tooling (supervisor, contact watcher, stability gate)
- [x] Eyes-open vs eyes-closed experiment, 4 subjects (alpha rose 1.3–5.7× with eyes closed, 4/4 subjects)
- [x] ZUNA1.1 three-branch evaluation — **results in `zuna_evaluation_v4.pdf`**
- [ ] Per-subject normalization for cross-subject EO/EC classification; more subjects
- [ ] Marker stream publisher for stimulus/event timestamps
- [ ] Optional: LabRecorder + XDF for multi-stream recordings once markers exist
