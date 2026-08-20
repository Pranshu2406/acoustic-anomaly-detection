"""Visual comparison: id_00 (a machine the CNN handles fine) vs. id_06 (the
below-chance failure) -- normal and abnormal spectrograms side by side, on a
shared dB color scale so a genuine energy-level/noise-floor difference would
actually be visible rather than auto-scaled away per subplot.
"""

import os

import librosa.display
import matplotlib.pyplot as plt

from src.preprocessing import HOP_LENGTH, SAMPLE_RATE, build_manifest, wav_to_logmel

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "notebooks", "sanity_check_id06_vs_id00.png")
VMIN, VMAX = -80, 0  # shared dB scale (matches power_to_db's default top_db=80)


def main():
    df = build_manifest("data/raw")

    rows_by_machine = {}
    for machine_id in ("id_00", "id_06"):
        machine_df = df[df["machine_id"] == machine_id]
        rows_by_machine[machine_id] = {
            "normal": machine_df[machine_df["label"] == "normal"].iloc[0],
            "abnormal": machine_df[machine_df["label"] == "abnormal"].iloc[0],
        }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for row_idx, machine_id in enumerate(("id_00", "id_06")):
        for col_idx, label in enumerate(("normal", "abnormal")):
            ax = axes[row_idx, col_idx]
            filepath = rows_by_machine[machine_id][label]["filepath"]
            logmel = wav_to_logmel(filepath)
            img = librosa.display.specshow(
                logmel,
                sr=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                x_axis="time",
                y_axis="mel",
                ax=ax,
                vmin=VMIN,
                vmax=VMAX,
            )
            ax.set_title(f"{label} ({machine_id})")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved to {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
