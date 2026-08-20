"""Generate Grad-CAM overlays for CNNGroupNorm and report the localization
diagnostic described in README's "Grad-CAM interpretability" section: does
the model attend to localized, anomaly-consistent time-frequency regions, or
to diffuse/broadband patterns that could indicate it's exploiting a
session-level recording confound (see the calibration-leakage investigation,
src/calibration.py)?

Examples generated:
  (a) 3-4 correctly-classified ABNORMAL clips, one per machine_id, drawn from
      the deployment model's held-out (chronological) validation split.
  (b) 2-3 correctly-classified NORMAL clips, same source.
  (c) the exact id_00 abnormal clip used in scripts/sanity_check_spectrograms.py
      (data/raw/fan/id_00/abnormal/00000000.wav), for direct visual comparison
      against that earlier finding (the ~1024 Hz band visible in both normal
      and abnormal id_00 spectrograms there -- present in both classes, so a
      candidate session/background confound rather than an anomaly signature).
"""

import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.calibration import train_or_load_deployment_model
from src.cnn_model import CNNGroupNorm
from src.gradcam import GradCAM, summarize_cam
from src.preprocessing import HOP_LENGTH, N_MELS, SAMPLE_RATE, wav_to_logmel
from src.train_cnn import MANIFEST_PATH, SPECTROGRAMS_PATH

OUT_DIR = "notebooks"
SAME_MACHINE_ABNORMAL_FILE = "data/raw/fan/id_00/abnormal/00000000.wav"
MEL_FREQS = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=SAMPLE_RATE / 2)


def _select_examples(manifest, specs, model, mean, std, val_idx, label, n, one_per_machine):
    y_all = (manifest["label"] == "abnormal").astype(int).values
    target = 1 if label == "abnormal" else 0
    target_sign = 1.0 if label == "abnormal" else -1.0

    picked = []
    seen_machines = set()
    for idx in val_idx:
        if y_all[idx] != target:
            continue
        machine_id = manifest.loc[idx, "machine_id"]
        if one_per_machine and machine_id in seen_machines:
            continue
        spec = (specs[idx] - mean) / std
        x = torch.from_numpy(spec.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            proba = torch.sigmoid(model(x)).item()
        correctly_classified = (proba >= 0.5) if target == 1 else (proba < 0.5)
        if not correctly_classified:
            continue
        picked.append({"idx": idx, "filepath": manifest.loc[idx, "filepath"], "machine_id": machine_id, "proba": proba, "target_sign": target_sign})
        seen_machines.add(machine_id)
        if len(picked) >= n:
            break
    return picked


def _plot_and_report(filepath, machine_id, label, proba, cam, tag, fig_idx):
    logmel = wav_to_logmel(filepath)
    summary = summarize_cam(cam, MEL_FREQS, HOP_LENGTH, SAMPLE_RATE, threshold=0.7)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    img0 = librosa.display.specshow(logmel, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel", ax=axes[0])
    axes[0].set_title(f"{machine_id}/{label} -- log-mel spectrogram")
    fig.colorbar(img0, ax=axes[0], format="%+2.0f dB")

    librosa.display.specshow(logmel, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel", ax=axes[1], cmap="gray")
    img1 = librosa.display.specshow(cam, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel", ax=axes[1], cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title(f"{machine_id}/{label} (proba={proba:.3f}) -- Grad-CAM overlay")
    fig.colorbar(img1, ax=axes[1], label="Grad-CAM (normalized)")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"gradcam_{fig_idx:02d}_{tag}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(
        f"[{tag}] {machine_id}/{label}  file={os.path.basename(filepath)}  proba={proba:.3f}  "
        f"normalized_entropy={summary['normalized_entropy']:.3f}  frac_hot(>=0.7)={summary['frac_hot']:.3f}  "
        f"hz_range={summary['hz_range'][0]:.0f}-{summary['hz_range'][1]:.0f}Hz  "
        f"time_range={summary['time_range_s'][0]:.1f}-{summary['time_range_s'][1]:.1f}s  "
        f"-> {out_path}"
    )
    return {"tag": tag, "machine_id": machine_id, "label": label, "file": os.path.basename(filepath), "proba": proba, **summary, "out_path": out_path}


def main():
    ckpt = train_or_load_deployment_model()
    model = CNNGroupNorm()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean, std = ckpt["mean"], ckpt["std"]
    val_idx = ckpt["val_idx"]

    manifest = pd.read_csv(MANIFEST_PATH)
    specs = np.load(SPECTROGRAMS_PATH)

    abnormal_examples = _select_examples(manifest, specs, model, mean, std, val_idx, "abnormal", n=4, one_per_machine=True)
    normal_examples = _select_examples(manifest, specs, model, mean, std, val_idx, "normal", n=3, one_per_machine=True)

    print(f"Selected {len(abnormal_examples)} abnormal / {len(normal_examples)} normal held-out, correctly-classified examples.\n")

    rows = []
    fig_idx = 0
    with GradCAM(model) as cam_fn:
        for ex in abnormal_examples:
            spec = (specs[ex["idx"]] - mean) / std
            x = torch.from_numpy(spec.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            result = cam_fn(x, target_sign=ex["target_sign"])
            assert np.isfinite(result["cam"]).all(), "Grad-CAM produced non-finite values"
            assert result["cam"].std() > 1e-6, "Grad-CAM output is degenerate (constant)"
            fig_idx += 1
            rows.append(_plot_and_report(ex["filepath"], ex["machine_id"], "abnormal", ex["proba"], result["cam"], "abnormal_heldout", fig_idx))

        for ex in normal_examples:
            spec = (specs[ex["idx"]] - mean) / std
            x = torch.from_numpy(spec.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            result = cam_fn(x, target_sign=ex["target_sign"])
            assert np.isfinite(result["cam"]).all(), "Grad-CAM produced non-finite values"
            assert result["cam"].std() > 1e-6, "Grad-CAM output is degenerate (constant)"
            fig_idx += 1
            rows.append(_plot_and_report(ex["filepath"], ex["machine_id"], "normal", ex["proba"], result["cam"], "normal_heldout", fig_idx))

        # (c) same clip as the original sanity-check spectrogram, for direct comparison.
        logmel = wav_to_logmel(SAME_MACHINE_ABNORMAL_FILE)
        spec = (logmel - mean) / std
        x = torch.from_numpy(spec.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        result = cam_fn(x, target_sign=1.0)
        assert np.isfinite(result["cam"]).all(), "Grad-CAM produced non-finite values"
        assert result["cam"].std() > 1e-6, "Grad-CAM output is degenerate (constant)"
        fig_idx += 1
        rows.append(_plot_and_report(SAME_MACHINE_ABNORMAL_FILE, "id_00", "abnormal", result["proba"], result["cam"], "same_machine_as_sanity_check", fig_idx))

    report_df = pd.DataFrame(rows)
    report_df.to_csv(os.path.join(OUT_DIR, "gradcam_report.csv"), index=False)
    print("\n" + "=" * 100)
    print(report_df.drop(columns=["out_path"]).to_string(index=False))
    print(f"\nSaved {len(rows)} overlays + report to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
