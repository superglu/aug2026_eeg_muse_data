Does the EEG foundation model preserve eyes-open/eyes-closed brain state?
Research question

ZUNA1.1 (380M param reconstruction-style EEG foundation model, Zyphra) claims to denoise and/or upsample consumer EEG. Does routing real Muse 2 recordings through it preserve the eyes-open (EO) vs eyes-closed (EC) alpha-blocking effect — the most robust, well-characterized signal in resting-state EEG — or does it get smoothed away by a model optimized for reconstruction error, not task-relevant information?

Muse 2 channels: TP9, AF7, AF8, TP10 (referenced to Fpz). TP9/TP10 sit behind the ears, close enough to posterior cortex to pick up alpha blocking via volume conduction. AF7/AF8 are prefrontal — weaker alpha signal, more blink/EOG contamination.

Study design
n = 4 subjects — originally scoped as n=5; one fewer than planned, which weakens the between-subject argument below slightly further. Framed explicitly as a pilot study, not a generalization claim (see rationale below).
One session per subject.
One 60-second block per condition per subject (60s EO + 60s EC, 2 min/subject total). No repeated blocks, no repeated sessions.

This is a much smaller design than a standard resting-state EEG protocol (which typically uses several alternating blocks per condition to average out moment-to-moment noise), so it changes what the harness can and can't tell you — see the caveats below before trusting any specific number.

Why n=4 is defensible here, and where it isn't

Alpha blocking is a large within-subject effect (Cohen's d often >1.5 with good data), so a handful of subjects can detect whether the effect replicates across independent people, and that remains true even with one block each — the between-subject question doesn't depend on within-subject repetition. n=4 is thinner than the originally planned n=5, though, and the actual results below (see "Results so far") show why that matters in practice: with this few subjects, per-subject alpha baseline differences dominate the feature space more than the EO/EC effect does.

It is not enough for: subgroup analysis, a trustworthy classification generalization estimate, or ruling out that one "low-alpha" subject (~10-15% of people show minimal alpha blocking) is driving a group result. Report per-subject results individually, not just pooled.

What a single 60s block per condition costs you specifically

With no repeated blocks or sessions, three things change from a richer design:

Epochs within a block are not independent samples. Slicing 60s into 2s epochs (the epoch length actually used — see implementation notes) gives 30 epochs per recording, but they're consecutive slices of one continuous physiological state — alpha power drifts slowly, so adjacent epochs are highly autocorrelated. Any effect size or classifier accuracy computed by treating those epochs as independent trials will look cleaner than the true trial-to-trial effect. Report this as a known inflation, not a corrected number — there isn't enough data here to correct it properly.
Test-retest reliability cannot be measured. That metric needs ≥2 sessions per subject to correlate the effect across repeats. It is dropped from this design entirely, not weakened — there's nothing to compare.
Standard (shuffled) K-fold cross-validation leaks. If epochs from the same 60s block land in both the train and test split, the classifier can partly learn that specific block's noise signature rather than genuine EO/EC structure, inflating accuracy. Leave-One-Subject-Out (LOSO) becomes the primary, and effectively the only trustworthy, classification metric in this design, since it's the one split that doesn't leak within a subject's single block. **This was not yet implemented — see "Deviations from this spec" below; the classification numbers reported so far use a weaker split.**

Pipeline (conceptual stages)
Muse recording (60s EO + 60s EC per subject, 4 subjects)
        │
        ▼
Ingest — muse_to_fif.py routes each recording to data/input_eo/ or data/input_ec/
by filename ("open"/"closed"), one file per subject per condition
        │
        ├──────────────────┬──────────────────────┐
        ▼                  ▼                       ▼
   [raw branch]        [denoise branch]       [upsample branch]
   no processing       ZUNA1.1 reconstructs   ZUNA1.1 adds 4
                        the same 4 channels    hallucinated posterior
                                                channels (O1, O2, Oz, Pz)
        │                  │                       │
        ▼                  ▼                       ▼
        └────────── Feature extraction (per branch) ──────────┘
                     alpha-band (8-13Hz) power per 2s epoch/channel
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                                ▼
        EO/EC classification                 Spectral sanity check
        (GroupKFold by recording --          (EC-EO alpha power, dB,
         see caveats, not LOSO)               per channel per branch)

Implemented as eo_ec_pipeline/: branches.py (build the 3 branches via ZUNA), features.py
(epoch + alpha power), classify.py (spectral check + classification), run_eo_ec_test.py
(end-to-end driver), plus band_power_check.py (all 5 bands, relative power), latent_tsne.py
(t-SNE of the alpha-power feature space), and plot_raw_figures.py (raw-signal figures to
compare against ZUNA's own denoised/upsampled overlay figures).

Deviations from this spec — what was actually implemented

- **n=4, not n=5.** One fewer subject than planned.
- **GroupKFold by recording, not LOSO.** classify.py groups folds by individual recording
  (8 groups: 4 subjects x 2 conditions), not by subject (which would be 4 groups). This is
  weaker than LOSO: a fold can train on a subject's EC recording and test on that same
  subject's EO recording, so subject-specific alpha baseline can still leak into a "pass."
  Per this doc's own reasoning, **the classification numbers below should be read as an
  optimistic upper bound relative to true LOSO, not a validated generalization estimate.**
  Implementing real LOSO (group by subject only) is the most important open item.
- **2s epochs, not the illustrative 4s** used in this doc's original example — an
  implementation choice, not a deviation in kind (30 epochs/recording instead of ~15).
  The "not independent samples" caveat above applies just as much, just with more epochs.
- **Upsample branch adds 4 channels (O1, O2, Oz, Pz), not 7.** The originally-listed
  POz/P3/P4 were not implemented.
- **Both branch flavors ZUNA writes (full_reconstruction/hybrid) were tried, then hybrid
  was dropped entirely.** For denoised, the two are numerically identical (100% of the
  channel is masked, so there's nothing for "hybrid" to preserve). For upsampled, they
  differ only on the real 4 channels (hybrid keeps them exactly as recorded,
  full_reconstruction regenerates them too) — measured at ~1-3% of each channel's own
  signal std, i.e. small but real. full_reconstruction was kept for both branches, for a
  consistent "model regenerates everything" comparison against raw.
- **Metrics 1, 2, 4, 5, 7, 9, 12 from the table below are not yet implemented**
  (NMSE/RMSE, PSD similarity, Cohen's d, IAF shift, confusion matrix, frozen-probe/
  finetuned latent gap, blink/EOG contamination rate). What's actually running today is
  metric 3 (as an EC-EO dB difference, not a ratio), metric 6 (accuracy only, no F1/AUC),
  and an informal version of metric 10 (per-channel dB comparison between hallucinated and
  real channels — not yet a formal Cohen's d).

Evaluation metrics — what each one tells you
#	Metric	Pipeline stage	What it catches	Status
1	NMSE / RMSE, raw vs. reconstructed	Signal fidelity check, denoise/upsample branches	Waveform-level distortion. Necessary but not sufficient — low error ≠ preserved task info.	Not implemented
2	PSD similarity (full spectrum)	Signal fidelity check	Distortion outside the alpha band you're targeting.	Partially covered by band_power_check.py (relative power, all 5 bands) but not as a formal similarity metric
3	Alpha suppression index (EC/EO alpha power ratio)	Effect-size analysis	The core EO/EC signal. Should stay ≥ raw value after processing, not shrink toward 1.	Implemented as EC-EO dB difference (classify.py spectral_check), not a ratio
4	Cohen's d, alpha power EC vs EO	Effect-size analysis	Effect size, more informative than p-values at n=4. Computed per subject and pooled.	Not implemented
5	IAF (individual alpha frequency) shift	Effect-size analysis	Subtler failure mode: peak alpha frequency drifting even if power looks OK.	Not implemented
6	Classification accuracy / F1 / AUC-ROC	EO/EC classification	Downstream task utility — the metric that actually matters for "does the FM preserve brain state."	Accuracy only, implemented (classify.py), via GroupKFold-by-recording (see deviations)
7	Confusion matrix	EO/EC classification	Asymmetric errors (e.g. EC→EO misclassification specifically) point to alpha attenuation rather than random noise.	Not implemented
8	Leave-One-Subject-Out (LOSO) accuracy — primary classification metric in this design	EO/EC classification	The one classification split that doesn't leak, given only one block per condition per subject. Standard shuffled K-fold is not used here — with a single 60s block per condition, its epochs aren't independent, so a shuffled split would let train/test share the same block and inflate accuracy. If reported at all, K-fold must be labeled as an optimistic upper bound, not a real estimate.	Not implemented — GroupKFold-by-recording used instead (weaker than LOSO; see deviations)
9	Frozen-probe vs. finetuned accuracy gap	EO/EC classification (extension)	Per the ZUNA1.1 paper's own caveat: frozen linear probes underperformed finetuned models. Large gap = info is present but not linearly accessible. Needs ZUNA1.1's latent-extraction API confirmed first.	Not implemented
10	Hallucinated-channel vs. real-channel effect size (upsample branch only)	Effect-size analysis, by channel type	The single most ZUNA-specific test, and unaffected by the single-block design since it compares channel types within the same data rather than across repeats. If hallucinated posterior channels don't show larger alpha Cohen's d than real TP9/TP10, ZUNA isn't adding real cross-channel information — just smoothing.	Informal version implemented (per-channel dB, not Cohen's d) — see results below
11	Test-retest reliability	—	Not measurable with this design. Needs ≥2 sessions per subject; dropped, not weakened.	Correctly not measured
12	Blink/EOG contamination rate	Feature extraction (extension)	Rules out that AF7/AF8 "effects" are blink-rate differences between EO/EC, not alpha.	Not implemented — flagged as a live concern below (AF7/AF8 show the largest raw-branch dB shift of any channel, which is exactly the pattern blink contamination would produce)

Note on effect sizes (rows 3–4): with one 60s block per condition, epoch-level Cohen's d and the alpha suppression index are computed over autocorrelated slices of a single continuous state, not independent trials. Report them, but flag them as inflated relative to a true trial-to-trial effect size — this design can't produce a corrected number, only a caveated one.

Results so far (n=4: dillon, gary, kotora, vishwani; 8 recordings, 240 epochs/branch)

Spectral sanity check (EC-EO log-alpha-power, dB) — the effect direction holds up everywhere:
- raw: real channels range 1.9-6.1 dB (AF7 highest at 6.1, TP9 lowest at 1.9)
- denoised: compressed but uniform, ~3.9 dB across all 4 real channels
- upsampled: real channels track raw almost exactly (1.9-6.0 dB); the 4 hallucinated
  channels (O1, O2, Oz, Pz) show a smaller but nonzero effect, 0.7-3.2 dB

Classification accuracy (GroupKFold by recording, NOT LOSO — read as an optimistic
upper bound per the caveats above):
- denoised: mean_acc 0.580, std_acc 0.398
- raw: mean_acc 0.443, std_acc 0.274 (below chance)
- upsampled: mean_acc 0.453, std_acc 0.279 (below chance)

raw and upsampled landing below 50% chance, combined with std_acc almost as large as
the means themselves, means these numbers should not be read as "ZUNA hurts
classification" — with only 5 GroupKFold folds over 8 recording-groups, each fold's
accuracy is a very noisy point estimate. The t-SNE visualization (latent_tsne.py) gives
the likely explanation: coloring the alpha-power feature space by subject shows tight,
mostly-separate clusters per person, while coloring by EC/EO condition shows the two
thoroughly intermixed within every cluster. Subject identity dominates this 4-feature
space far more than the EO/EC effect does — consistent with the "n=4 is too few for a
trustworthy classification generalization estimate" caveat above, and a concrete
argument for implementing real LOSO next rather than trusting these numbers as-is.

The hallucinated-vs-real channel comparison (metric 10) is at least encouraging: ZUNA's
invented posterior channels aren't flat/uninformative (0.7-3.2 dB effect), but they're
smaller than the real channels' effect, not larger — so by this design's own success
criterion ("hallucinated channels show a larger effect size than real channels"), the
upsample branch has not yet demonstrated it's adding real cross-channel information
rather than a plausible-looking smoothing.

What "success" looks like for ZUNA1.1

Rank branches (raw < denoise < upsample) by whichever of these hold:

Alpha suppression index and Cohen's d preserved or increased vs. raw.
LOSO classification accuracy preserved or increased vs. raw. (Not yet tested — current
numbers use GroupKFold-by-recording, a weaker and more optimistic split.)
(Upsample only) hallucinated channels show a larger effect size than real channels.
(Not yet met — see results above.)

If denoise/upsample match or fall below raw on these, that's evidence for the paper's own flagged dissociation between reconstruction accuracy and downstream utility — a genuine, useful negative result for a Muse-specific stress test.

Data collection checklist

For each of the 4 subjects, in one session:

Fit the Muse 2 headset, confirm signal quality on all 4 channels (TP9, AF7, AF8, TP10) before starting.
Record one 60-second eyes-closed block and one 60-second eyes-open block. Name the export file with "open" or "closed" in it — muse_to_fif.py routes to data/input_eo/ or data/input_ec/ by filename and errors out if it can't tell which.
Export the raw EEG recording (e.g. via Mind Monitor CSV export) and convert with ingestion/muse_to_fif.py.
Note anything unusual per subject (headset slipped, excessive blinking, external noise) — with n=4 and one block each, a single contaminated block can visibly shift a pooled result, so this needs to be flagged and checked, not silently averaged away.

Reporting checklist for the writeup
Report per-subject alpha suppression index and Cohen's d — not just pooled — since n=4 can't distinguish "consistent moderate effect" from "huge effect in some, absent in others."
Lead with LOSO accuracy, not GroupKFold-by-recording, for the classification result — LOSO is not yet implemented, so there is currently no result that meets this bar (see deviations).
Label epoch-level effect sizes as inflated relative to a true trial-to-trial estimate, given the single-block design.
Do not report a test-retest number — it isn't measurable here.
Highlight the hallucinated-vs-real channel comparison as the most ZUNA-specific finding, since it's the one metric this design doesn't weaken — and note that, so far, it does not favor ZUNA (hallucinated effect < real effect).
