"""Pull a single machine-ID subset out of a MIMII zip via HTTP range requests,
without downloading the whole archive.
"""

import os
import sys
import time

import requests
from remotezip import RemoteZip, RangeNotSupported, RemoteIOError, RemoteZipError
from tqdm import tqdm

URL = "https://zenodo.org/records/3384388/files/6_dB_fan.zip?download=1"
MACHINE_ID = "id_00"
DEST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "fan")
CHUNK_SIZE = 1024 * 256


def main():
    dest_dir = os.path.abspath(DEST_DIR)
    os.makedirs(dest_dir, exist_ok=True)

    try:
        zf = RemoteZip(URL)
    except (RangeNotSupported, RemoteIOError, RemoteZipError, requests.exceptions.RequestException) as e:
        print(f"ERROR: could not open remote zip via range requests: {type(e).__name__}: {e}")
        print("Falling back to a full resumable download is recommended (aria2c -x8 -s8 -c ...).")
        sys.exit(1)

    with zf:
        members = zf.infolist()

        print(f"Archive contains {len(members)} entries. Top-level structure:")
        top_level = sorted({m.filename.split("/")[0] + "/" for m in members if "/" in m.filename})
        for entry in top_level:
            print(f"  {entry}")
        machine_dirs = sorted(
            {"/".join(m.filename.split("/")[:2]) + "/" for m in members if m.filename.count("/") >= 2}
        )
        print(f"\nMachine ID subfolders found ({len(machine_dirs)}):")
        for d in machine_dirs:
            print(f"  {d}")

        subset = [
            m for m in members
            if f"/{MACHINE_ID}/" in m.filename and not m.is_dir()
        ]

        if not subset:
            print(f"\nERROR: no files found for machine ID '{MACHINE_ID}'. Aborting.")
            sys.exit(1)

        total_bytes = sum(m.file_size for m in subset)
        total_files = len(subset)
        print(
            f"\nDownloading {total_files} files for machine '{MACHINE_ID}' "
            f"({total_bytes / (1024 ** 2):.1f} MB total) into {dest_dir}"
        )

        start = time.time()
        downloaded_bytes = 0

        with tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="starting",
        ) as pbar:
            for member in subset:
                rel_path = member.filename.split(f"{MACHINE_ID}/", 1)[-1]
                sub_label = member.filename.split("/")[-2]  # normal or abnormal
                out_path = os.path.join(dest_dir, MACHINE_ID, sub_label, os.path.basename(member.filename))
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                pbar.set_description(member.filename)
                with zf.open(member) as src, open(out_path, "wb") as dst:
                    while True:
                        chunk = src.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        dst.write(chunk)
                        pbar.update(len(chunk))
                        downloaded_bytes += len(chunk)

        elapsed = time.time() - start
        avg_speed_mb_s = (downloaded_bytes / (1024 ** 2)) / elapsed if elapsed > 0 else 0.0

        print("\n--- Summary ---")
        print(f"Files downloaded : {total_files}")
        print(f"Total size       : {downloaded_bytes / (1024 ** 2):.1f} MB")
        print(f"Elapsed time     : {elapsed:.1f} s")
        print(f"Average speed    : {avg_speed_mb_s:.2f} MB/s")


if __name__ == "__main__":
    main()
