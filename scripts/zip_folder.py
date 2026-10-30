#!/usr/bin/env python
"""Zip a folder as one ZIP64 archive: already compressed render/*.npz files are stored, the rest
deflated; .git and __pycache__ are left out.

    python scripts/zip_folder.py /root/projects/drive_v3 /dev/shm/drive_v3_data.zip
"""
import os
import sys
import time
import zipfile
from pathlib import Path


def pack(src, dst):
    src, t0, n = Path(src), time.time(), 0
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, allowZip64=True, compresslevel=6) as z:
        for root, dirs, files in os.walk(src):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")   # .git, .superpowers
            for f in sorted(files):
                p = Path(root) / f
                store = p.parent.name == "render" and p.suffix == ".npz"
                z.write(p, Path(src.name) / p.relative_to(src), compress_type=zipfile.ZIP_STORED if store else zipfile.ZIP_DEFLATED)
                n += 1
                if n % 5000 == 0:
                    print(f"  {dst}: {n} files, {time.time() - t0:.0f}s", flush=True)
    print(f"{dst}: {n} files, {os.path.getsize(dst) / 1e9:.2f} GB, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    pack(sys.argv[1], sys.argv[2])
