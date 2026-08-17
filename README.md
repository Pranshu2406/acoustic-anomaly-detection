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
