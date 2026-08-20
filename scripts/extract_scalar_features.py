"""Extract scalar audio features for every file in the manifest and save to CSV."""

import os
import time

import pandas as pd
from tqdm import tqdm

from src.features import extract_scalar_features
from src.preprocessing import build_manifest

OUT_PATH = os.path.join("data", "processed", "scalar_features.csv")


def main():
    manifest = build_manifest("data/raw")
    print(f"Manifest rows: {len(manifest)}")

    start = time.time()
    rows = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="extracting features"):
        features = extract_scalar_features(row.filepath)
        rows.append({"filepath": row.filepath, "machine_id": row.machine_id, "label": row.label, **features})
    elapsed = time.time() - start

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nExtracted features for {len(df)} files in {elapsed:.1f}s ({elapsed / len(df):.3f}s/file)")
    print(f"Row count matches manifest: {len(df) == len(manifest)}")
    print(f"Saved to {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
