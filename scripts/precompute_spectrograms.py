"""Precompute log-Mel spectrograms for every file in the manifest and cache them
as a single array, aligned by row index to a saved copy of the manifest.
"""

import os
import time

import numpy as np
from tqdm import tqdm

from src.preprocessing import N_FRAMES, N_MELS, build_manifest, wav_to_logmel

MANIFEST_OUT = os.path.join("data", "processed", "manifest.csv")
SPEC_OUT = os.path.join("data", "processed", "spectrograms.npy")


def main():
    manifest = build_manifest("data/raw")
    os.makedirs(os.path.dirname(MANIFEST_OUT), exist_ok=True)
    manifest.to_csv(MANIFEST_OUT, index=False)
    print(f"Manifest: {len(manifest)} rows saved to {MANIFEST_OUT}")

    specs = np.zeros((len(manifest), N_MELS, N_FRAMES), dtype=np.float32)

    start = time.time()
    for i, row in enumerate(tqdm(manifest.itertuples(index=False), total=len(manifest), desc="computing spectrograms")):
        specs[i] = wav_to_logmel(row.filepath)
    elapsed = time.time() - start

    np.save(SPEC_OUT, specs)

    print(f"\nComputed {len(manifest)} spectrograms in {elapsed:.1f}s ({elapsed / len(manifest):.3f}s/file)")
    print(f"Shape: {specs.shape}")
    print(f"Row count matches manifest: {specs.shape[0] == len(manifest)}")
    print(f"Saved to {os.path.abspath(SPEC_OUT)}")


if __name__ == "__main__":
    main()
