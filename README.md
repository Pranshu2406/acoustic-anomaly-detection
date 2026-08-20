# Acoustic Anomaly Detection

Detecting anomalous machine sounds (fan, pump, valve, slider) from audio recordings.

## Project layout

- `data/raw/` — raw audio recordings, organized by machine type (`fan/`, `pump/`, `valve/`, `slider/`). Not tracked in git.
- `src/preprocessing.py` — audio-to-Mel-spectrogram feature extraction.
- `src/models.py` — PyTorch model definitions.
- `notebooks/` — exploration, preprocessing, classical baselines, and PyTorch modeling notebooks.
- `tests/` — unit tests.

## Data leakage rule

Segments from the same recording session must never be split across train/test sets.
Splitting must happen at the session (or machine unit) level, before any windowing or
augmentation, so that no segment derived from a given session appears in both the
training set and the test set.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Source: [MIMII dataset](https://zenodo.org/records/3384388) (Malfunctioning Industrial
Machine Investigation and Inspection), Zenodo record 3384388.

Currently downloaded: **fan machine type, 6dB SNR** (`6_dB_fan.zip`, the cleanest noise
condition) — 4 machine IDs (`id_00`, `id_02`, `id_04`, `id_06`), 5,550 total 10-second,
16kHz, 8-channel `.wav` clips (4,075 normal / 1,475 abnormal), extracted into
`data/raw/fan/<machine_id>/{normal,abnormal}/`.

This is a pilot slice used to validate the preprocessing and modeling pipeline before
committing to the full ~80-100GB dataset. If the pipeline works well on this slice,
the other machine types (`pump`, `valve`, `slider`) and SNR levels (`-6dB`, `0dB`) can
be added the same way, from the same Zenodo record.

## Preprocessing

`src/preprocessing.py` converts each raw `.wav` file into a fixed-size log-Mel
spectrogram (`wav_to_logmel`, shape `(64, 313)`) using `n_fft=1024`, `hop_length=512`,
`n_mels=64` (the MIMII baseline convention), at the dataset's native 16000 Hz — the
sample rate is asserted rather than silently resampled, so a mismatched file fails
loudly instead of corrupting features. Every spectrogram is padded or truncated to
exactly 313 time frames (10s * 16000 / 512 ≈ 313) so every CNN input has identical shape.

**Channel selection:** MIMII recordings are 8-channel (TAMAGO-03 microphone array), but
only channel 0 is used; channels 1-7 are discarded. This matches the MIMII baseline
convention and most published work on this dataset, and keeps the initial model simple
and interpretable. Multi-channel/array-based input is a documented possible follow-up
experiment, not the default — the same "don't add complexity without proving it's
needed" reasoning behind the bank-failure literature's XGBoost-vs-logistic-regression
result, where the more complex model wasn't adopted until it actually earned its keep.

**Per-clip loudness normalization:** `wav_to_logmel` computes the log-Mel spectrogram
with `librosa.power_to_db(mel, ref=np.max)`, i.e. each clip's dB scale is normalized to
that clip's own peak. This preserves relative spectral shape within a clip but discards
absolute loudness differences between clips — two recordings of the same event at
different volumes end up looking identical. This is a candidate ablation for later
(e.g. comparing against a fixed global reference instead of per-clip `ref=np.max`), not
a settled decision.

`build_manifest()` walks `data/raw/` and returns a dataframe of every `.wav` file
(`filepath`, `machine_type`, `machine_id`, `label`, `split`). `split` is left as a
placeholder — train/test assignment must group by `machine_id` (per the leakage rule
above) rather than splitting files at random.

## Phase 3: classical ML baselines

`src/features.py` and `src/baselines.py` establish the bar a later CNN needs to beat:
logistic regression, XGBoost, and random forest, trained on 10 hand-engineered scalar
features (spectral centroid/bandwidth/rolloff, zero-crossing rate, RMS energy — mean
and std of each), evaluated with 4-fold leave-one-machine-out cross-validation.

XGBoost has the best mean PR-AUC (0.594) but is also the least stable across machines
(±0.241 std) and actually falls below its own chance baseline on one held-out machine
(`id_02`: 0.244 vs. 0.261 chance). Random forest is more consistent (0.589 ± 0.036 mean
PR-AUC) and stays comfortably above chance on `id_02` (0.601), which shows XGBoost's
`id_02` failure is model-specific overfitting rather than that machine being
fundamentally hard for tree-based methods. Logistic regression trails both on mean
PR-AUC (0.474) but is the most stable of the three. Net takeaway: simple scalar features
already beat chance by a wide margin on most machines, but no single classical model is
reliably good on *every* held-out machine — the CNN's bar isn't just "beat the average
PR-AUC," it's "don't catastrophically fail on any one machine the way XGBoost did."

## Phase 4: CNN

`src/cnn_model.py` and `src/train_cnn.py` train a small CNN on precomputed log-Mel
spectrograms (`scripts/precompute_spectrograms.py`, cached to
`data/processed/spectrograms.npy`), with the same 4-fold leave-one-machine-out
structure as Phase 3, for a fair comparison against the classical baselines.

**BatchNorm distribution-shift finding and fix:** the first CNN (3 conv+BatchNorm+
ReLU+maxpool blocks) collapsed to below-chance PR-AUC on 2 of 4 held-out machines
(`id_02`, `id_06`) — predicting "abnormal" for almost every clip regardless of true
label. Diagnosis showed the held-out machines are genuinely louder and have a more
compressed dynamic range than the training machines (e.g. `id_02`: +6.8 dB mean shift,
`id_06`: +2.4 dB), which BatchNorm's running statistics — fit only on the training
machines — don't generalize to at eval time. Swapping BatchNorm for GroupNorm (same
architecture, same hyperparameters, everything else identical) resolved it: mean PR-AUC
across folds rose from 0.410 to 0.674, and every fold moved above its own chance
baseline. **`CNNGroupNorm` is now the default model**; the BatchNorm version (`SmallCNN`)
is kept only as a comparison/ablation (`scripts/compare_cnn_norm.py`).

**Final CNN vs. classical comparison** (PR-AUC per machine; classical = best of logistic
regression / XGBoost / random forest for that machine):

| machine held out | CNN (GroupNorm) | best classical | chance |
|---|---|---|---|
| id_00 | 0.662 | 0.627 (xgboost) | 0.287 |
| id_02 | 0.570 | 0.601 (random forest) | 0.261 |
| id_04 | 0.780 | 0.731 (xgboost) | 0.252 |
| id_06 | 0.686 | 0.773 (xgboost) | 0.262 |
| **mean** | **0.674** | **0.683** | — |

**Headline finding: roughly matched performance, not a clear win for either.** The CNN
wins on 2 machines, classical wins on 2, and the aggregate means differ by less than
0.01 PR-AUC. With only 4 machine IDs, each "per-machine" result is a single held-out
test, and margins this size (±0.03–0.09 PR-AUC) are well within what you'd expect from
noise alone at n=4 — this comparison shows the two approaches are in the same
performance tier, not that either reliably wins.

## Phase 5: cross-model error analysis

`scripts/build_predictions_table.py` persists every model's held-out-fold predicted
probability for every clip to `data/processed/all_predictions.csv`
(logistic regression, XGBoost, random forest, CNN), and `scripts/error_analysis.py`
analyzes it.

**Unique-error-rate finding (corrected):** raw counts of "clips only this model gets
wrong" are misleading on their own, because CNN has a higher total error count (2,856)
than the other three models (1,710–2,363) — a model that's simply wrong more often will
also rack up more unique-wrong clips without necessarily failing in a more distinctive
way. Normalizing by each model's own total error count
(`unique_error_rate = unique_wrong / total_wrong`) tells a different story: CNN's
unique_error_rate (0.252) falls *within* the classical models' range (0.039–0.295),
not above it. Logistic regression — the simplest model — actually has the highest
unique_error_rate (0.295) despite having the fewest total errors, while random forest
has the lowest (0.039), meaning its errors mostly reproduce failures other models also
make rather than contributing distinct ones. **CNN is not making disproportionately more
idiosyncratic errors than the classical models — its higher unique-error count is
roughly proportional to its higher overall error count.**

**What does hold up: error concentration by machine.** `id_00` shows heavy false-positive
collapse (predicting "abnormal" on the majority of normal clips) across *multiple*
models, not just the CNN — random forest (92.7%) and CNN (100%) both collapse there,
XGBoost partially (69.6%). CNN additionally collapses on `id_02` specifically (99.4% FP
rate on normal), which the classical models don't share. `id_04` is easy for every
model (near-zero false-positive rate on normal across the board). This machine-level
concentration is the more reliable and more interesting result from this phase — not
any claim that a given model is uniquely idiosyncratic in *what* it gets wrong.

## Phase 6: latency and efficiency

`scripts/latency_benchmark.py` measures inference speed and model size for all 4
models, forced to **CPU** regardless of MPS availability (`torch device='cpu'`) — the
realistic resource-constrained edge-deployment scenario, even though CNN *training*
uses MPS elsewhere in this project. Each model is timed with 10 discarded warm-up
calls, then 100 timed calls on the same 100 randomly sampled clips (seed 42) for every
model. Two latency numbers are reported: **model-only** (input already extracted/
normalized, isolates the model itself) and **full pipeline** (raw `.wav` filepath to
prediction, including feature extraction).

| model | size | model-only mean/p95 (ms) | full-pipeline mean/p95 (ms) | throughput (clips/sec) |
|---|---|---|---|---|
| logistic_regression | 0.9 KB | 0.050 / 0.054 | 14.3 / 17.3 | 69.8 |
| xgboost | 143.1 KB | 0.077 / 0.085 | 12.0 / 12.6 | 83.0 |
| random_forest | 218.8 KB | 2.201 / 2.287 | 14.6 / 15.1 | 68.4 |
| cnn | 25,633 params / 106.3 KB | 0.984 / 1.208 | **5.6 / 6.0** | **179.5** |

Full results: `data/processed/latency_results.csv`.

**Correctness fix applied to these numbers:** `extract_scalar_features` originally called
`spectral_centroid`/`spectral_bandwidth`/`spectral_rolloff` with `y=`, and each recomputed
its own STFT internally — 3 redundant STFTs per clip. Verified numerically that all three
accept a shared magnitude spectrogram via `S=` (matching default `n_fft=2048,
hop_length=512`) and return bit-identical values (max abs diff 0.0 across all 5,550 clips'
10 feature columns before vs. after); `rms` was deliberately left on `y=` since
`rms(S=...)` is a genuinely different algorithm (spectral-magnitude vs. time-domain
framing) that would have changed its values (~0.003 max diff). Sharing one STFT across
the three fixed functions cut classical full-pipeline latency by ~26-30% (e.g. XGBoost:
17.3ms → 12.0ms mean) — a real fairness fix to the benchmark, not a change to any
feature's actual value.

**Interpretation.** Under a realistic near-real-time threshold (e.g. under 100ms per
clip), all four models pass comfortably — even the slowest full-pipeline p95 (random
forest, 15.1ms) is well under budget, so raw latency doesn't rule anything out here.
**The CNN still has the lowest full-pipeline latency and highest throughput of all
four** (179.5 clips/sec vs. 68-83 clips/sec), despite XGBoost and logistic regression
having the fastest model-only latency — full-pipeline cost is dominated by feature
extraction, not the model, and the CNN's single log-Mel spectrogram is still cheaper
than even the STFT-shared classical pipeline. That said, **the "~3x throughput"
finding shrinks once the redundant-STFT bug is fixed: it's now ~2.4x** (179.5 vs. an
average of ~73.8 clips/sec across the three classical models, down from ~3.4x /
~53.3 clips/sec before the fix) — a real and still meaningful CNN advantage, but a
smaller one than first reported; a third of the original gap was a benchmark
artifact, not a genuine architectural difference. Combined with the Phase 4/5 finding
that CNN and the best classical model are roughly accuracy-matched (mean PR-AUC 0.674
vs. 0.683), there still isn't a clear accuracy-vs-latency tradeoff: the CNN remains
competitive on accuracy and meaningfully ahead on end-to-end latency/throughput, just
by a smaller margin than the pre-fix numbers suggested. The one other caveat is model
size at the extreme low end — logistic regression's 0.9 KB footprint is in a different
class from the others (218x smaller than random forest, ~120x smaller than the CNN),
which would only matter for genuinely memory-constrained embedded targets, not a
typical edge SBC.

## Phase 8: compression

`src/compression.py` and `scripts/run_compression_benchmark.py` apply software-only
quantization and pruning to `CNNGroupNorm`, CPU-only, reusing the Phase 6
latency-benchmarking methodology (10 discarded warm-up calls + 100 timed model-only
calls, mean/p95). This is a stretch goal, not full CV: everything is evaluated on a
single held-out fold (`id_00`, same split/hyperparameters/seed as Phase 4/5) rather
than re-running 4-fold CV for every compression variant. No checkpoint from the
earlier CV runs existed on disk for this specific fold (Phase 6's saved checkpoint was
trained on *all* 4 machines, which would leak `id_00` into training), so this fold was
retrained from scratch and saved to
`data/processed/compression_models/cnn_id00_holdout.pt`.

**Quantization.** Post-training static int8 quantization via
`torch.ao.quantization.quantize_fx` (FX graph mode, `qnnpack` backend — the only
quantized CPU backend available on arm64/Apple Silicon; `fbgemm` is x86-only),
calibrated on 200 spectrograms drawn only from the 3 training machines (never `id_00`,
which would be calibration leakage). **Quantization is partial, not full**: GroupNorm
has no registered quantized CPU kernel in `torch.ao.quantization`, so FX mode leaves
every GroupNorm layer in fp32 and inserts a dequantize/quantize boundary around each
one, while Conv2d and Linear layers are genuinely replaced with `qnnpack` int8 kernels.
This is reported programmatically, not just in prose: `quantize_static()` returns the
list of fallback module types alongside the converted model, and
`tests/test_compression.py` asserts `"GroupNorm"` appears in that list so a future
change to PyTorch's quantization support (or the architecture) wouldn't silently pass.

**Pruning.** Unstructured global L1-magnitude pruning (`torch.nn.utils.prune`) across
every Conv2d/Linear weight tensor at 30%/50%/70% target sparsity, masks made permanent
so the reported sparsity is literally-zero weights, not a mask. **Pruning here
demonstrates accuracy headroom, not a deployment speedup** — unstructured pruning
zeroes weights but does not change dense tensor shapes, so it will not meaningfully
reduce latency or serialized size under standard PyTorch dense CPU inference; a real
speedup would require structured pruning or a sparsity-aware runtime, out of scope
here.

| variant | size (KB) | model-only latency mean/p95 (ms) | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| fp32 baseline | 105.2 | 2.162 / 3.071 | 0.446 | 0.779 | 0.662 |
| int8 quantized (partial — GroupNorm stays fp32) | 36.4 | 2.333 / 3.031 | 0.446 | 0.756 | 0.633 |
| pruned 30% | 105.2 | 1.725 / 2.173 | 0.446 | 0.801 | 0.654 |
| pruned 50% | 105.2 | 1.700 / 2.067 | 0.446 | 0.823 | 0.713 |
| pruned 70% | 105.2 | 1.627 / 1.987 | 0.448 | 0.783 | 0.645 |

Full results: `data/processed/compression_results.csv`.

**Single-fold caveat.** Every number above comes from one held-out machine (`id_00`)
with no repeated runs — unlike the 4-fold leave-one-machine-out CV used elsewhere in
this project (Phases 3-5), this is a single train/eval pass per variant, as scoped at
the top of this section. Treat differences under ~0.05 PR-AUC between variants as
noise-level, not a reliable effect — e.g. pruning's apparent accuracy edge at 50%
sparsity (0.713 vs. the fp32 baseline's 0.662) is exactly this kind of difference and
should not be read as pruning genuinely improving accuracy, only as being roughly
accuracy-neutral. Two findings, though, are well-supported despite the single-fold
setup because they're corroborated by an independent mechanism, not just the metric
delta: **(1)** quantization's size reduction (65%) is a real, mechanism-consistent
result — it follows directly from which layers were actually replaced with int8
kernels, not from a noisy metric — and it comes with a small but *consistent*
accuracy cost (ROC-AUC and PR-AUC both drop, not just one) and no latency benefit on
a network this small with partial GroupNorm fallback; **(2)** the network tolerates
pruning up to at least 50% sparsity with no meaningful accuracy loss, indicating real
parameter redundancy — but the exact optimal sparsity level (e.g. whether 50% is
truly better than 30%) isn't something a single fold can reliably determine.

**Quantization: a real size win, no latency win, a small accuracy cost.** Serialized
size drops 65% (105.2 KB → 36.4 KB) even with GroupNorm excluded from quantization,
since Conv2d/Linear parameters dominate the model's footprint. But model-only latency
did **not** improve — it's marginally *slower* on mean (2.333ms vs 2.162ms), with p95
roughly flat. This contradicts the usual expectation that int8 quantization speeds up
CPU inference, and the likely explanation is architecture-specific: this network is
tiny (3 conv blocks) and has 3 separate quantize/dequantize boundary crossings (one
per GroupNorm), and at this size the quantize/dequantize overhead can outweigh the
`qnnpack` int8 kernels' compute savings — a bigger or more fully-quantizable model
would likely show a different result. Accuracy also drops measurably (ROC-AUC 0.779 →
0.756, PR-AUC 0.662 → 0.633), a real cost from int8 rounding, not noise-level. Net: on
this model, quantization is a genuine size optimization but not a latency one, and it
isn't accuracy-free.

**Pruning: substantial accuracy headroom, latency/size flat as expected.** Size is
byte-identical to the fp32 baseline at every sparsity level (105.2 KB, since
unstructured pruning zeroes values in place rather than shrinking tensors), confirming
the predicted non-result. Accuracy is essentially unaffected up to 50% sparsity, and
50% sparsity actually scores *higher* PR-AUC (0.713) and ROC-AUC (0.823) than the
unpruned baseline — magnitude pruning acting as a mild regularizer on this small,
possibly slightly overparameterized network. Only at 70% sparsity does PR-AUC fall
back toward baseline (0.645). F1 is flat at 0.446-0.448 across every variant because
it's computed at a fixed 0.5 probability threshold that none of these perturbations
move clips across for this fold. Measured latency shows a mild *downward* drift with
higher sparsity (2.162ms → 1.627ms mean) despite unstructured pruning not changing
tensor shapes or FLOP count — likely measurement noise or a CPU-level fast path for
multiply-by-zero rather than a genuine structural speedup, and not something to rely
on: **pruning here demonstrates accuracy headroom, not a deployment speedup** — that
would require structured pruning or a sparsity-aware runtime, out of scope here.

`tests/test_compression.py` confirms the quantized model produces correctly-shaped
output (single-sample and batched) and explicitly reports the GroupNorm fallback, and
confirms each pruned model's actual measured sparsity matches its target (30/50/70%,
±2%).

## Real-time inference

`src/streaming.py` runs the trained CNN as a live pipeline: microphone input (via
`sounddevice`, the only backend that installed cleanly on this machine without a
separate PortAudio dev install) → a 10-second rolling audio buffer (`RollingAudioBuffer`,
matching the training clip length and 16kHz sample rate) → `waveform_to_logmel` (a
core function extracted from `src/preprocessing.py`'s `wav_to_logmel` — training and
streaming inference now share the exact same log-Mel code path, so they can't
silently drift apart) → `CNNGroupNorm` → a raw probability → a **calibrated**
probability (see below) → a three-way uncertainty band → a console alert. Inference
runs on a timer (default every 2s), not on every audio chunk, so the CPU isn't pegged
running the model dozens of times a second.

The buffer starts as 10 seconds of silence and fills in as audio arrives, so
probabilities during the first ~10 seconds of a run are on a partially-silent window
and can be unreliable/noisy — this is expected and settles down once the buffer is
fully populated with real audio.

**Model checkpoint: trained on all 4 machines, not a held-out fold.**
`src/calibration.py`'s `load_deployment_model` (used by `src/streaming.py`) loads
`data/processed/deployment_model/cnn_deployment.pt` — a checkpoint trained on the
**full dataset (all 4 machine IDs: `id_00`, `id_02`, `id_04`, `id_06`)**, not a
leave-one-machine-out fold. This is a deliberate, different choice from Phases 3-5's
accuracy comparisons, not an oversight: Phases 3-5 hold out one machine at a time
specifically to *measure* generalization to an unseen machine, which requires
withholding data during training. A real deployment has no reason to withhold data —
there's no benefit to deliberately not training on a known machine's recordings — so
the streaming pipeline uses the model trained on everything available, the same way
you'd ship a production model rather than one of the CV folds. The 85/15 split used
for early stopping is **chronological, not random** — see "Session-leakage
investigation" below for why — with the held-out 15% taken from the end of each
machine/label's recording sequence. This same held-out 15% is what calibration is fit
and evaluated on below, and its indices are saved inside the checkpoint so calibration
can prove it never touched the training rows.

**Live microphone mode:**

```bash
python -m src.streaming --mode mic --interval 2
```

Runs until Ctrl+C (or pass `--duration 60` to stop automatically after 60s). Requires
an actual input device — `sounddevice` will raise if none is available.

**Test/simulated mode** (no microphone or live anomalous sound needed — streams a
real `.wav` file from the dataset through the identical buffer + inference pipeline,
in small chunks, as if it were arriving live):

```bash
python -m src.streaming --mode test --file data/raw/fan/id_00/normal/00000000.wav --interval 2
python -m src.streaming --mode test --file data/raw/fan/id_00/abnormal/00000000.wav --interval 2
```

```
[2026-08-18 12:12:42] ANOMALY DETECTED           calibrated=1.000  raw=1.000  latency=797.1ms
[2026-08-18 12:12:42] ANOMALY DETECTED           calibrated=0.981  raw=0.981  latency=4.8ms
[2026-08-18 12:12:42] UNCERTAIN - FLAG FOR REVIEW calibrated=0.254  raw=0.254  latency=4.8ms
[2026-08-18 12:12:42] NORMAL                     calibrated=0.000  raw=0.000  latency=4.3ms
```

```
[2026-08-18 12:12:46] ANOMALY DETECTED           calibrated=1.000  raw=1.000  latency=793.9ms
[2026-08-18 12:12:46] ANOMALY DETECTED           calibrated=1.000  raw=1.000  latency=4.7ms
[2026-08-18 12:12:46] ANOMALY DETECTED           calibrated=0.998  raw=0.998  latency=4.3ms
```

Verified against one known normal and one known abnormal `id_00` clip: once the
buffer is fully loaded, the normal clip settles to `NORMAL` and the abnormal clip to
`ANOMALY DETECTED` — both correctly banded. The `UNCERTAIN` readings above are from
the first couple of cycles, while the buffer is still mostly silence-padded (see
above) — this is expected transient behavior, not a calibration problem. **This is a
pipeline sanity check, not a generalization result**: because the loaded checkpoint
was trained on all 4 machines (including `id_00` itself, see above), both test clips
were part of that model's own training data, so a correct prediction here only
confirms that buffering, preprocessing, calibration, and banding are wired together
correctly end-to-end and produce sane output — it says nothing about accuracy on
unseen machines. The real generalization numbers are the Phase 4/5
leave-one-machine-out table earlier in this README. `tests/test_streaming.py` covers
the same two checks (with an explicit comment to the same effect), plus the rolling
buffer's sliding-window behavior on synthetic samples (fixed size, silence until
filled, correctly drops the oldest samples as new ones arrive, and handles a single
write larger than the whole window).

Each inference cycle is timed end-to-end (buffer read → preprocessing → model →
calibration → banding), same `time.perf_counter` approach as Phase 6. After the first
cycle in a process (~0.8-1.1s, dominated by one-time `librosa`/`torch` warm-up, not
the model), steady-state latency settles to ~4.5-5.5ms per cycle — consistent with
Phase 6's CNN full-pipeline latency (5.6/6.0ms mean/p95), comfortably inside a
near-real-time budget even at a 1-2s inference interval.

### Session-leakage investigation (calibration split)

Before trusting the calibration split, its construction was checked for a specific
risk: MIMII's `.wav` filenames are sequential zero-padded integers per
machine/label folder (`00000000.wav`, `00000001.wav`, ...). If those numbers encode
real recording order, a *random* 85/15 split could scatter near-identical,
temporally-adjacent clips across both train and held-out — inflating held-out
accuracy without the model actually generalizing to anything new.

**1. Do the filenames correspond to recording/session order?** Checked empirically
rather than assumed: for each machine/label folder, RMS energy was computed per clip
in filename order, then compared adjacent-index differences (`|rms[i] - rms[i+1]|`)
against random-pair differences. If filenames were arbitrary/shuffled, these two
should be about equal (ratio ≈ 1). They weren't:

| folder type | adjacent/random RMS-diff ratio (range across 4 machines) |
|---|---|
| `normal` | 0.70 – 0.94 (mild autocorrelation) |
| `abnormal` | 0.05 – 0.21 (strong autocorrelation) |

Plotting RMS energy vs. filename index confirms it visually: `abnormal` folders show
sharp plateau/block transitions (e.g. `id_00/abnormal` sits at RMS≈0.010 for indices
0-130, drops to ≈0.0045 for 130-265, then jumps to ≈0.0055 for 265-407) — almost
certainly distinct fault-severity recording sessions concatenated in file order.
`normal` folders show a milder settling-in drift early in the sequence. **Conclusion:
filenames do encode real session/chronological structure**, confirmed by data, not
assumed from the dataset's naming convention alone.

**2. Did the original random split leak across sessions?** Quantified directly: for
every held-out clip in the original random 85/15 split (stratified by
`machine_id`+`label`), the file-index distance to the nearest *training* clip from
the same machine+label was computed.

| metric | value |
|---|---|
| median distance to nearest same-machine-same-label training clip | **1** (i.e. the adjacent file) |
| held-out clips with a training clip within distance ≤ 1 | **97.8%** |
| held-out clips with a training clip within distance ≤ 5 | **100%** |

At 85% training density, this level of adjacency is close to the combinatorial
maximum for a random draw — essentially every held-out clip had its immediate
filename-neighbor sitting in the training set. Combined with finding 1 (adjacent
clips are near-duplicates, especially in `abnormal` folders), this is a real leakage
risk, not a theoretical one.

**3. Corrected split.** `chronological_last_fraction_split` (`src/calibration.py`)
replaced the random stratified draw: for each `(machine_id, label)` group, clips are
ordered by filename index and the chronologically **last 15%** is held out, with a
single boundary per group rather than clips interleaved throughout — mirroring the
leave-one-machine-out discipline used for Phases 3-5, applied here at the
session/sequence level instead of the whole-machine level.

**4. Did it matter?** The deployment model was retrained on the corrected split and
recalibrated. Held-out validation PR-AUC during training remained ~1.000 either way,
but Brier score and ECE both got measurably worse under the corrected split — in the
direction leakage would predict (a leaky split makes held-out performance look
*better* than it should):

| split | val PR-AUC (training) | raw Brier | raw ECE |
|---|---|---|---|
| random (leaky) | ~1.000 | 0.001564 | 0.003782 |
| chronological (corrected) | ~1.000 | 0.002677 | 0.004220 |

**Both things are true at once, and the numbers below now use the corrected split
throughout:** the leakage was real and measurable (Brier score worsened ~71% once
corrected, ECE ~12%), so the corrected split is the right one to report and the old
numbers are superseded — but the *qualitative* finding didn't change. Even with
adjacent-session clips no longer able to leak across the boundary, held-out PR-AUC on
known machines still converges to ~1.000. That's a genuine finding, not a leftover
leakage artifact: monitoring a **known** machine (has this specific unit's sound
changed from what it's shown before) is a fundamentally easier task than the Phase 4/5
leave-one-machine-out question (does this generalize to a **machine never heard
before**, mean PR-AUC 0.674). The corrected split fixed a real methodological flaw
without overturning the headline conclusion.

A ~1.000 PR-AUC surviving the leakage fix rules out *this specific* leakage
mechanism (adjacent-session clips crossing the split boundary), but doesn't by
itself prove the ~1.000 reflects genuine anomaly signal rather than some other
session-level confound the model could still be keying off (e.g. a persistent
background tone shared by both classes on a given machine). The
[Grad-CAM interpretability](#grad-cam-interpretability) section below investigates
that directly, on this same deployment model and split: 3 of 4 machines show
localized attention that lines up with real spectrogram energy and explicitly
ignores an available same-both-classes confound band, while `id_06` remains an
open caveat where attention does overlap such a band.

### Probability calibration

Raw sigmoid outputs from a classifier aren't automatically trustworthy as
probabilities — e.g. a model can be systematically overconfident even when its
*ranking* of clips (normal vs. abnormal) is accurate. `src/calibration.py` fits and
evaluates two calibrators on the held-out split described above, and — per the task
here — deliberately does **not** skip straight to uncertainty banding on raw
probabilities without checking this first.

**This calibration answers a different question than Phases 3-5.** It's fit and
evaluated on a held-out split of **all 4 machines**, because the deployment model
itself trains on all 4 machines. That answers "are this model's probabilities
trustworthy on *new clips from known machines*" — a deployment-monitoring question.
It is **not** the leave-one-machine-out generalization question from Phases 3-5
("does this model work on an *entirely unseen machine*"), which uses a completely
different held-out split (one whole machine, never seen at all) to answer a
completely different question (raw predictive accuracy on a new machine, not
probability trustworthiness on known ones). Both are legitimate evaluations that this
project runs; they measure different things and should not be conflated with each
other's numbers.

**Method:** Platt scaling (a 1D logistic regression on the model's raw logits) and
isotonic regression (fit on raw probabilities), evaluated against the raw
(uncalibrated) probabilities via Brier score and Expected Calibration Error (ECE,
10 equal-width bins), run once via `scripts/run_calibration.py` on the corrected
chronological split described above:

| variant | Brier score | ECE | n (held-out) |
|---|---|---|---|
| raw (uncalibrated) | 0.002677 | 0.004220 | 832 |
| Platt scaling | 0.002221 | 0.002804 | 832 |
| isotonic regression | 0.001202 | 0.000000 | 832 |

(For reference, the pre-correction numbers on the leaky random split were: raw
0.001564 / 0.003782, Platt 0.001392 / 0.003119, isotonic 0.000800 / 0.000000 —
superseded by the table above; see the leakage investigation for why.)

**Winner: none.** Neither calibrator improved both Brier score and ECE by a
meaningful margin (>0.005 absolute) over the raw probabilities, so raw
(uncalibrated) probabilities are used for banding — reported plainly rather than
picking a calibrator anyway, per the explicit instruction not to. The reason the bar
is essentially unreachable here: the deployment model already achieves ~1.000
validation PR-AUC on its own held-out split even after the leakage fix (see the
investigation above — that's a genuine property of same-machine monitoring, not a
split artifact), meaning it separates the two classes almost perfectly on known
machines, so raw probabilities are already concentrated near 0 and 1 and already
well-calibrated — there's very little miscalibration left to correct. Isotonic
regression's numbers look best on paper (ECE exactly 0.0), but that's a yellow flag,
not a green one: isotonic regression is evaluated here on the exact same 832-sample
split it was fit on (no separate calibration-test split), and it's flexible enough to
fit that specific sample's empirical frequencies almost exactly — a classic isotonic
overfitting risk on a calibration set this size, especially with most of the mass
sitting at the extremes. An ECE of exactly 0.0 on your own fitting data is a sign to
distrust the number, not to declare victory, so "none" is the more honest call than
crowning isotonic winner on numbers unlikely to hold up on a genuinely fresh sample.

The reliability diagram (`notebooks/calibration_reliability_diagram.png`) shows why:
predictions are heavily concentrated at the two large markers near (0,0) and (1,1)
(most samples), with only a handful of scattered, small (low-sample-count) points in
between — the middle of the curve is sparse and noisy by construction, not because
any calibrator is failing.

### Three-way uncertainty banding

Built on top of the (here: raw, since no calibrator won) calibrated probability:

| calibrated probability | band | streaming display |
|---|---|---|
| < 0.2 | normal | `NORMAL` |
| 0.2 – 0.8 | uncertain | `UNCERTAIN - FLAG FOR REVIEW` |
| > 0.8 | anomaly | `ANOMALY DETECTED` |

`CalibratedPredictor.band()` implements the thresholds; `src/streaming.py`'s
`run_inference_cycle` calls it every inference cycle and prints the resulting band
alongside both the calibrated and raw probabilities (see the example output above).
`tests/test_calibration.py` covers the exact threshold boundaries (0.19/0.2/0.8/0.81),
Brier/ECE correctness against synthetic examples with known calibration error
(including a case with a known 0.4 ECE gap), a synthetic-but-clearly-miscalibrated
scenario where both calibrators are confirmed to actually reduce ECE/Brier, and —
via monkeypatching `LogisticRegression.fit`/`IsotonicRegression.fit` to record
exactly what data reaches sklearn — that calibrator fitting only ever touches the
held-out arrays it was given, never anything resembling the (deliberately shifted,
easy-to-detect-if-leaked) training-fold data constructed alongside them in the test.

## Grad-CAM interpretability

**Model used: the all-4-machine deployment model, not a Phase 4/5 fold checkpoint.**
`scripts/run_gradcam.py` loads the model via `src.calibration.train_or_load_deployment_model`
— the same all-4-machine `CNNGroupNorm` checkpoint (`data/processed/deployment_model/cnn_deployment.pt`)
used for calibration and streaming — and draws every example from its
chronological held-out split (the same split used for the calibration Brier/ECE
numbers above), not from any of the Phase 4/5 leave-one-machine-out fold models.
This distinction matters: leave-one-machine-out folds are trained *without* the
held-out machine, so they can't answer a same-machine-fingerprint question at all
(the machine being explained was never in their training data). Only a model
trained on all 4 machines can be checked for whether it learned a
machine-identity/session shortcut on a machine it *did* train on — which is
exactly this section's diagnostic goal, and why the deployment model is the
correct one to use here.

`src/gradcam.py` implements Grad-CAM for `CNNGroupNorm`, targeting the ReLU output
of the last conv block (`features[2][2]`) — the finest-resolution post-nonlinearity
feature map that still feeds the model's global average pooling. Grad-CAM's
weighting (global-average-pooled gradients per channel) has no BatchNorm-specific
assumptions, so it was expected to work unmodified with GroupNorm — but that was
**verified empirically, not assumed**: `tests/test_gradcam.py` checks CAMs are
finite, non-constant, correctly shaped, input-dependent, and differ between the
"abnormal" and "normal" target directions, on both random-init and trained models.
One real implementation snag surfaced along the way: `register_full_backward_hook`
is incompatible with an in-place `ReLU` (`RuntimeError: ... is a view and is being
modified inplace`) — `GradCAM.__init__` switches the target layer to out-of-place
(`inplace=False`), which doesn't change ReLU's computed values, only how memory is
reused.

**Diagnostic goal.** Beyond general interpretability, this was built to answer a
specific question raised by the [session-leakage investigation](#session-leakage-investigation-calibration-split)
above: does the model attend to localized, anomaly-consistent time-frequency
regions, or to diffuse/broadband patterns — which would suggest it's exploiting a
session-level recording confound (e.g. a persistent background tone specific to one
machine or recording session, present regardless of anomaly status) rather than a
genuine fault signature? Concretely, this bears on how to read that section's
corrected ~1.000 held-out calibration PR-AUC: fixing the adjacent-session-clip
leakage didn't lower it, which is consistent with genuine signal, but doesn't rule
out a *different*, still-present session-level confound (like a background tone
shared by both classes) inflating it instead — this section checks that directly by
looking at what the model actually attends to, rather than inferring it indirectly
from a metric.

**A concrete confound candidate was already sitting in an existing plot.**
`notebooks/sanity_check_spectrograms.png` (from an earlier phase) shows a sharp,
persistent horizontal band at **~1024 Hz in BOTH the normal and abnormal `id_00`
spectrograms**, at similar intensity — this was never called out in writing before,
but it's a textbook candidate for exactly the kind of confound this investigation
is checking for: if it's equally present in both classes, a model keying off it
would be learning a machine fingerprint, not an anomaly signature.
`notebooks/sanity_check_id06_vs_id00.png` shows `id_06` has its own different but
analogously persistent bands (~500 Hz and ~950 Hz, in both classes), plus a
genuinely abnormal-only broadband high-frequency burst after ~7.5s — a good
candidate for real anomaly signal to contrast against.

**Examples generated** (`scripts/run_gradcam.py`, saved to `notebooks/gradcam_*.png`
+ `notebooks/gradcam_report.csv`), all correctly classified by the deployment model:

- 4 abnormal clips, one per machine (`id_00`, `id_02`, `id_04`, `id_06`), drawn from
  the held-out (chronological) calibration split
- 3 normal clips, one per machine (`id_00`, `id_02`, `id_04`), same source
- the exact `id_00` abnormal clip used in the original sanity-check spectrogram
  (`data/raw/fan/id_00/abnormal/00000000.wav`), for direct visual comparison

For a single-logit binary model there's no separate "normal" class score to
target, so normal-clip CAMs backpropagate the *negated* logit (what pushes the
score toward "normal") rather than the raw logit used for abnormal clips.

| example | machine | proba | normalized entropy | "hot" (≥0.7) region |
|---|---|---|---|---|
| abnormal (held-out) | id_00 | 0.980 | 0.80 | 0–144 Hz, periodic bursts at 1.6–6.3s |
| abnormal (held-out) | id_02 | 1.000 | 0.93 | 0–670 Hz, rhythmic pulses across nearly the whole clip |
| abnormal (held-out) | id_04 | 1.000 | 0.83 | 0–192 Hz, periodic bursts across nearly the whole clip |
| abnormal (held-out) | id_06 | 0.652 | 0.88 | 0–527 Hz, includes the persistent ~500/1024 Hz bands |
| normal (held-out) | id_00 | 0.000 | 0.99 | 0–192 Hz, milder periodic bursts |
| normal (held-out) | id_02 | 0.000 | 0.99 | 0–1287 Hz, more broadband |
| normal (held-out) | id_04 | 0.000 | 0.99 | 0–479 Hz, periodic low-frequency bursts |
| abnormal (same clip as sanity-check) | id_00 | 0.998 | 0.85 | 0–144 Hz, periodic bursts — visually matches the held-out `id_00` row above |

**Per-example read (is it localized/anomaly-consistent, or diffuse?):**

- **`id_00` (both the held-out example and the original sanity-check clip):**
  localized and time-periodic — hotspots concentrated in the 0–200 Hz range,
  pulsing at specific moments rather than spread evenly, and these pulses visibly
  line up with brighter low-frequency energy bursts in the underlying spectrogram.
  Critically, the CAM does **not** light up on the ~1024 Hz band even though it's
  right there in the input and highly salient by eye — direct evidence *against*
  the model using that particular confound, at least for this machine.
- **`id_04`:** the cleanest case — sharp, strongly time-periodic low-frequency
  hotspots that line up almost pixel-for-pixel with visible bright bursts in the
  spectrogram. Looks like genuine (likely rotational-fault-related) signal
  tracking, not diffuse attention.
- **`id_02`:** still concentrated in the low-to-mid frequency range with a
  rhythmic time pattern (consistent with tracking a mechanical rotation period),
  but broader and less sharply localized than `id_00`/`id_04` — a real but fuzzier
  signal.
- **`id_06` (the one exception):** this is both the *lowest-confidence* correct
  prediction (proba 0.652, near the decision boundary) and the one case where the
  CAM's hot region overlaps the machine's own persistent ~500 Hz/~1024 Hz bands —
  bands that are present in `id_06`'s normal clips too (see
  `sanity_check_id06_vs_id00.png`). This is the one piece of evidence *for* the
  confound-exploitation concern, and it's not a coincidence that it lands on
  `id_06` specifically: this is the same machine whose original BatchNorm model
  collapsed to below-chance performance (Phase 4), the same machine flagged for
  distribution shift (`scripts/diagnose_id06.py`). It's a plausible harder case
  where the model leans more on machine-identity cues precisely when the
  anomaly-specific signal is weaker.
- **Normal clips:** consistently more diffuse than abnormal clips (normalized
  entropy ~0.99 vs. ~0.80–0.93) and spread across a wider frequency range. This is
  expected rather than concerning — "evidence toward normal" is closer to an
  absence-of-anomaly signal than a specific pattern, so a less localized CAM here
  doesn't carry the same diagnostic weight as it would for an abnormal prediction.

**Diagnostic conclusion — an honest mixed read, not a predetermined one.** The
majority of examples (`id_00` ×2, `id_02`, `id_04`) show localized, time-periodic
attention that visibly aligns with real energy patterns in the spectrogram, and the
`id_00` comparison specifically shows the model ignoring an available, salient,
both-classes-present confound (~1024 Hz) that it easily could have keyed off. That
supports **the model has learned real, at least partially localized anomaly
signatures**, not just diffuse broadband pattern-matching, on 3 of the 4 machines
tested. But `id_06` is a genuine counterexample, not noise dressed up as one — it's
lower-confidence and its attention measurably overlaps a background band shared by
both classes on that specific machine, which is precisely the confound-exploitation
signature this investigation was checking for. **Verdict: the evidence favors "real
signal, with a machine-dependent exception," not a clean answer either way** —
`id_06` in particular should be treated as a standing caveat on this model's
trustworthiness, consistent with (and now reinforcing, via an independent method)
its already-documented history as this project's hardest machine.

## Edge deployment readiness

No Raspberry Pi or Jetson is owned yet, so everything below was validated with
software-only tooling on this development machine (an Apple Silicon Mac) instead:
model export, a standalone edge inference script, and a `linux/arm64` Docker
container run via [Colima](https://github.com/abiosoft/colima) + `docker buildx`.

### 1. Exported artifacts (`models/export/`)

`src/export.py` / `scripts/export_model.py` export the all-4-machine deployment
model (`src.calibration.load_deployment_model`) to:

| file | format | size |
|---|---|---|
| `cnn_deployment.torchscript.pt` | TorchScript (`torch.jit.script`) | 132.6 KB |
| `cnn_deployment.onnx` | ONNX (opset 17) | 108.7 KB |
| `norm_stats.json` | plain JSON (`mean`, `std`) | <1 KB |

Both `torch.jit.script` and `torch.jit.trace` work cleanly on `CNNGroupNorm` — no
GroupNorm-specific scripting issues turned up — `script` was used since it doesn't
bake in a fixed batch size the way a traced graph does. `torch.onnx.export` (the
current default/legacy exporter; the newer `dynamo=True` exporter needs an extra
`onnxscript` dependency this project doesn't otherwise need, so it was skipped)
also worked without modification. `tests/test_export.py` confirms both exported
formats match the original PyTorch model to within `atol=1e-4` on 5 real clips
(individually and batched) — actual measured max diff was ~7e-6, tolerance is set
conservatively above that. Normalization stats are exported as a standalone JSON
file rather than requiring the edge script to load the full training checkpoint
(and everything that pulls in) just to read two floats.

### 2. `scripts/edge_inference.py` — the actual on-device script

Dependency-light by design: only `numpy`, `librosa`+`soundfile` (log-mel extraction,
reusing `src/preprocessing.py` so it's guaranteed identical to training — not
reimplemented), and `onnxruntime`. No `torch`, no `pandas`, no `scikit-learn`
imported directly, no training code. **ONNX Runtime was chosen over full PyTorch
specifically for its smaller memory footprint** — the model-runtime piece of that
is confirmed below (~9MB resident for the ONNX session). **That said, this choice
alone turned out not to be sufficient for the Zero 2 W** — §3's memory
investigation found `librosa`'s own dependency chain (`numba`/`llvmlite`/`scipy`/
`scikit-learn`, all still required for correct preprocessing) costs far more
memory than the model runtime does either way; see that section and the runbook
below for the corrected picture. One small supporting refactor:
`src/preprocessing.py`'s `import pandas` was moved from module level to inside
`build_manifest()` (the only function that needs it), so importing `wav_to_logmel`
here doesn't transitively pull in pandas at all.

```bash
python scripts/edge_inference.py path/to/clip.wav
# ANOMALY DETECTED  proba=0.9981  logit=6.2686  latency=12.8ms  file=path/to/clip.wav
```

### 3. ARM64 validation via Docker buildx + Colima

No container runtime was installed on this machine at all going in; [Colima](https://github.com/abiosoft/colima)
(open-source, Homebrew-installable, no license/GUI requirement) plus the `docker`
and `docker-buildx` CLIs were installed to actually run this validation rather than
just writing the Dockerfile and stopping there.

```bash
brew install colima docker docker-buildx
colima start --cpu 4 --memory 4 --disk 30
docker run --privileged --rm tonistiigi/binfmt --install all   # registers QEMU handlers
docker buildx create --name edgebuilder --use --bootstrap

docker buildx build --platform linux/arm64 -f Dockerfile.edge -t acoustic-anomaly-edge:arm64 --load .

docker run --rm --platform linux/arm64 \
  -v "$(pwd)/data/raw/fan/id_00/abnormal/00000000.wav:/app/test.wav:ro" \
  acoustic-anomaly-edge:arm64 /app/test.wav
```

**Result: the build succeeded, and the container correctly classified real test
clips** (`ANOMALY DETECTED proba=0.9981` / `NORMAL proba=0.0000`, matching the host
PyTorch/ONNX outputs exactly).

**Important honesty check on "emulation":** this Mac is itself Apple Silicon
(arm64), so Colima's Linux VM runs `linux/arm64` **natively** — no QEMU CPU
instruction translation was actually involved in this build/run, despite the task
framing of "ARM emulation." QEMU binfmt handlers were installed and a genuine
`docker buildx` multi-platform workflow was used (so the exact same commands would
transparently trigger real QEMU emulation on an x86 dev machine building for
`linux/arm64`), but on this specific host, "emulated" would be the wrong word for
what happened — it was a real Linux/arm64 environment, just virtualized (via
Colima's VM) rather than bare-metal. That still validates something bare macOS
couldn't: real Linux/aarch64 wheel availability, `apt`/`glibc` behavior, and
container runtime behavior distinct from this host's Darwin/arm64 userspace.

**ARM-specific findings (the actual point of doing this before hardware arrives):**

- **No wheel-availability failures.** Every dependency — including `numba`/`llvmlite`,
  historically two of the more failure-prone packages on ARM due to slow/absent
  prebuilt wheels — resolved to prebuilt `manylinux_2_27/2_28_aarch64` wheels
  immediately for Python 3.11. `llvmlite`'s wheel alone is 58.3 MB and dominated
  total install time (~105s), but nothing needed a source build.
- **A virtualization-specific ONNX Runtime warning appears:** `onnxruntime cpuid_info
  warning: Unknown CPU vendor. cpuinfo_vendor value: 0`, on every run inside the
  container. It didn't affect correctness (outputs were bit-for-bit consistent with
  the host), but it's a known quirk of running onnxruntime inside a virtualized/
  emulated ARM environment (its `cpuinfo` dependency can't always read real CPU
  vendor info through a hypervisor) — expect to see it, and don't treat it as a
  failure on its own.
- **A large one-time JIT-compilation cost on the first inference call — clearly
  separate from steady-state per-inference latency.** These are two different
  numbers and get reported separately, not blended: the ~11s figure below is a
  **one-time startup cost paid once per process**, not what a deployed device
  pays per clip. Calling `predict()` repeatedly in the same process (not a fresh
  `docker run` each time) isolates the two cleanly:

  | | container (Colima, linux/arm64) | host (native macOS, arm64) |
  |---|---|---|
  | **one-time JIT warmup cost** (1st call) | ~11,300 ms | ~1,840 ms |
  | **steady-state per-inference latency** (post-warmup — 10 discarded + 100 timed, same methodology as Phase 6; full numbers in §4 below) | mean 20.05 ms / p95 30.78 ms | mean 12.84 ms / p95 20.90 ms |

  The one-time cost is `numba` JIT-compiling `librosa`'s internal functions (e.g.
  `zero_crossings`, mel-filterbank code) on first use — not the ONNX model itself —
  and it is **not** representative of per-clip inference cost once the process has
  warmed up. The container's cold-start is ~6x slower than the host's, plausibly
  VM/filesystem overhead rather than CPU emulation (there isn't any here, see
  above); steady-state is only ~1.5-2x slower in the container, a much smaller gap.
  A real device that stays running as a long-lived process pays the ~11s cost once
  at startup, then settles to the steady-state numbers above for every subsequent
  clip — but a device that re-invokes the script fresh per clip would eat that ~11s
  tax every single time, which matters a lot on constrained hardware (see runbook
  step 7 below).
- **Persistent memory footprint is ~370MB RSS after warmup — driven by `librosa`'s
  dependency chain, not by ONNX Runtime.** Measuring true current RSS (`/proc/self/status`
  `VmRSS`, not the `ru_maxrss` high-water mark) at each stage:

  | stage | RSS |
  |---|---|
  | after Python startup | 7.2 MB |
  | after imports (`onnxruntime` + `librosa`) | 46.5 MB |
  | after ONNX session load | 55.0 MB |
  | after 1st `predict()` (cold, triggers JIT) | 362.0 MB |
  | after 2nd/3rd `predict()` (warm) | 372.5-376.5 MB |

  The ONNX Runtime session itself only accounts for ~9 MB (55.0 - 46.5). The jump
  to ~362 MB happens entirely during `librosa`'s first real call and **never comes
  back down** — this is `numba`/`llvmlite`'s retained compiled-code and codegen
  state, not a transient spike. **This is the single most important finding here**:
  choosing ONNX Runtime over PyTorch specifically to save memory doesn't help if the
  *preprocessing* dependency chain (`librosa` → `numba`/`llvmlite`/`scipy`/
  `scikit-learn`) is what actually costs the memory. On a 512MB Zero 2 W, ~370MB
  resident leaves well under 150MB for the OS and everything else — tight, and a
  real risk this validation surfaced that wouldn't have been obvious without it.
- **Attempted mitigation, found not viable as-is:** `NUMBA_DISABLE_JIT=1` was tried
  to avoid paying the JIT memory/time cost at all. It breaks `librosa` outright —
  `AttributeError: 'function' object has no attribute 'get_call_template'` inside
  `numba`'s `@guvectorize`-decorated `zero_crossings` — because disabling JIT isn't
  cleanly supported for that decorator in this `numba`/`librosa` version
  combination. Not a fix; noted here so it isn't re-attempted blind later. A real
  fix would mean avoiding `librosa`'s numba-accelerated code paths entirely (e.g. a
  hand-rolled STFT/mel-filterbank in plain NumPy/SciPy), which is a real
  engineering task, not a flag flip — out of scope here, flagged as follow-up work.

### 4. Latency: emulated/virtualized numbers, explicitly not a hardware benchmark

Full pipeline (`edge_inference.py`: file read → log-mel → normalize → ONNX Runtime
→ prediction), 10 discarded warm-up calls + 100 timed, same methodology as Phase 6/8:

| environment | mean | p95 |
|---|---|---|
| host (native macOS, Apple Silicon arm64) | 12.84 ms | 20.90 ms |
| container (Colima `linux/arm64` VM) | 20.05 ms | 30.78 ms |

**Do not read these as real embedded-hardware numbers.** Two reasons, not one:
(1) as established above, this ran on Apple Silicon — a Cortex-A53 on a Raspberry
Pi Zero 2 W (or even the beefier cores on a Pi 4/5 or Jetson) is dramatically
slower than an M-series core, emulated or not; (2) the container number reflects
Colima VM overhead, not QEMU translation overhead, which would look different again
on an x86 dev machine. **These numbers are a smoke test that the code runs
correctly end-to-end on ARM/Linux, nothing more.** For reference, the original
native-PyTorch numbers from earlier phases (same machine, different code path —
Phase 6/8's CNN benchmark calls the model directly with a pre-loaded batch of
spectrograms rather than through this script's per-file subprocess-style
measurement):

| variant | model-only mean/p95 | full-pipeline mean/p95 |
|---|---|---|
| PyTorch fp32 (Phase 6/8) | 0.984 / 1.208 ms | 5.6 / 6.0 ms |
| ONNX Runtime, `edge_inference.py` (this section, host) | — | 12.84 / 20.90 ms |

### What's validated vs. not

**Validated:**
- TorchScript and ONNX exports are numerically equivalent to the original PyTorch
  model (`tests/test_export.py`, `atol=1e-4`, real clips).
- `scripts/edge_inference.py` runs correctly standalone, with only
  `numpy`/`librosa`/`soundfile`/`onnxruntime` as dependencies.
- The ONNX model + edge script build and run successfully in a real `linux/arm64`
  Docker environment via Colima + buildx, producing outputs matching the host
  exactly.
- No ARM wheel-availability blockers for any dependency in this stack, on Python 3.11.

**NOT validated (explicitly out of scope without real hardware):**
- Real embedded-ARM-silicon performance (Cortex-A53/A72/Jetson cores) — the numbers
  above are Apple Silicon, virtualized, not representative.
- Real microphone I/O on-device: `src/streaming.py`'s `sounddevice`/PortAudio path
  hasn't been tested against real embedded Linux ALSA configuration, USB vs.
  onboard mic behavior, or actual capture latency on a Pi.
- Actual RAM headroom on a real Raspberry Pi OS Lite install on a Zero 2 W — this
  validation used a generic `python:3.11-slim` container, not the real target OS
  image, and the ~370MB figure above should be treated as a lower bound to verify,
  not a guarantee.
- Thermal/power behavior, SD card I/O performance, boot time.

### Runbook: when a Raspberry Pi arrives

1. **Flash the OS.** Raspberry Pi Imager → Raspberry Pi OS Lite (64-bit, arm64) —
   the 64-bit build specifically; the 32-bit default on some Pi models won't run
   `manylinux_aarch64` wheels. For a **Zero 2 W**, Lite (no desktop) is not optional
   given the RAM finding above.
2. **First boot + update:** SSH in, `sudo apt update && sudo apt full-upgrade`,
   `sudo apt install python3-pip python3-venv libsndfile1` (`libsndfile1` is the
   native dependency `soundfile` needs; it's usually not preinstalled on Lite).
3. **Set up the environment:**
   ```bash
   python3 -m venv edge-env && source edge-env/bin/activate
   pip install numpy librosa soundfile onnxruntime
   ```
   **On a Zero 2 W specifically: use this ONNX Runtime path instead of full
   PyTorch — but be clear about what that does and doesn't fix.** ONNX Runtime
   itself is genuinely small (~9MB resident, measured above); full PyTorch plus
   this project's training-stack dependencies would be worse and there's no
   reason to ship them to a device that only ever runs inference. **But ONNX
   Runtime alone does not solve the Zero 2 W's memory constraint** — the ~370MB
   RSS finding above is caused by `librosa`'s `numba`/`llvmlite` preprocessing
   dependency chain, not the model runtime, and that cost is identical whether
   the model itself runs on ONNX Runtime or PyTorch. On a 512MB board, ~370MB
   resident for preprocessing alone is a real, currently-unresolved risk, not a
   solved problem. **Making the Zero 2 W realistic would require a genuinely
   lighter preprocessing path — e.g. a hand-rolled NumPy/SciPy FFT + mel
   filterbank that avoids `librosa`/`numba` entirely** (matching `wav_to_logmel`'s
   output exactly, the same non-negotiable constraint as everywhere else in this
   project). **This has not been attempted and is an open item, not something
   the current pipeline already handles** — treat the Zero 2 W as unvalidated for
   memory until that work is done and re-measured on real hardware; the current
   `models/export/` + `edge_inference.py` path is validated for correctness and
   is a reasonable target for a Pi 4/5 or similar with more RAM headroom, just
   not confirmed to fit the Zero 2 W's 512MB as-is. Budget real time for the
   `pip install` regardless: `numba`/`llvmlite`/`scipy` are large downloads even
   with prebuilt wheels available (as confirmed above), and will be slower on a
   Zero 2 W's weaker CPU/network than on this dev machine.
4. **Copy the artifacts** (from this repo, via `scp` or an SD card):
   `models/export/cnn_deployment.onnx`, `models/export/norm_stats.json`,
   `scripts/edge_inference.py`, `src/__init__.py`, `src/preprocessing.py`.
5. **Smoke test:**
   ```bash
   python3 edge_inference.py path/to/test_clip.wav
   ```
   Expect the first call to be slow (JIT compilation, seconds not milliseconds —
   see the container numbers above as a rough proxy) and subsequent calls to be
   much faster. **Verify actual RAM headroom on the real device before trusting
   this fits** (`free -h` before/during a run) — this is the one finding from this
   section most likely to differ on real hardware vs. this validation.
6. **For live microphone monitoring** (not just file-based smoke testing): the
   `RollingAudioBuffer` pattern in `src/streaming.py` is the right structure to
   reuse, but that module currently calls the full PyTorch model + calibration
   pipeline, not this ONNX path — porting it to call `scripts/edge_inference.py`'s
   `predict()` (ONNX Runtime, not PyTorch) is required before running it on a
   Zero 2 W, and is follow-up work, not done here.
7. **If cold-start latency matters for the deployment** (e.g. clips arrive
   sporadically rather than continuously): run as a long-lived process/service
   (e.g. a `systemd` unit) rather than re-invoking the script per clip, so the
   `numba` JIT cost is paid once at startup, not on every inference.
