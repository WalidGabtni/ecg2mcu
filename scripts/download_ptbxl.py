"""Download and unpack the PTB-XL dataset (v1.0.3, ~1.8 GB) from PhysioNet.

Usage:  python scripts/download_ptbxl.py [--dest data]

The dataset is released under CC BY 4.0: https://physionet.org/content/ptb-xl/1.0.3/
"""

import argparse
import os
import sys
import urllib.request
import zipfile

NAME = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
URL = f"https://physionet.org/static/published-projects/ptb-xl/{NAME}.zip"


def progress(blocks, block_size, total):
    done = blocks * block_size
    if total > 0:
        sys.stdout.write(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB")
        sys.stdout.flush()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_dest = os.path.join(os.path.dirname(here), "data")  # <repo>/data
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default=default_dest)
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    target = os.path.join(args.dest, NAME)
    if os.path.isfile(os.path.join(target, "ptbxl_database.csv")):
        print(f"Dataset already present at {target}")
        return

    zip_path = os.path.join(args.dest, NAME + ".zip")
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, zip_path, progress)
    print("\nExtracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(args.dest)
    os.remove(zip_path)
    print(f"Done: {target}")


if __name__ == "__main__":
    main()
