"""Plot a normal vs. abnormal log-Mel spectrogram side by side as a sanity check."""

import os

import librosa.display
import matplotlib.pyplot as plt

from src.preprocessing import SAMPLE_RATE, HOP_LENGTH, build_manifest, wav_to_logmel

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "notebooks", "sanity_check_spectrograms.png")


def main():
    df = build_manifest("data/raw")
    id_00 = df[df["machine_id"] == "id_00"]

    normal_row = id_00[id_00["label"] == "normal"].iloc[0]
    abnormal_row = id_00[id_00["label"] == "abnormal"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, row in zip(axes, (normal_row, abnormal_row)):
        logmel = wav_to_logmel(row["filepath"])
        img = librosa.display.specshow(
            logmel,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            ax=ax,
        )
        ax.set_title(f"{row['label']} ({row['machine_id']})")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved to {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
