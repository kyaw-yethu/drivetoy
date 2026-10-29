"""On-disk DriveToy dataset — reads a directory built by
``f_toy.datasets.drivetoy.builder.build_to_disk``. The video envelope is the
generic one from ``f_toy.data.disk``; every array in ``annotations.npz``
(docs/DRIVETOY.md §7) is passed through as the label dict."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ...data.disk import load_memmap_videos, load_meta


class DiskDataset(Dataset):
    def __init__(self, root):
        root = Path(root)
        self.root = root
        self.meta = load_meta(root)
        self.T = self.meta["frames"]
        self.videos = load_memmap_videos(root)
        ann = np.load(root / "annotations.npz")
        self.ann = {k: torch.from_numpy(ann[k]) for k in ann.files}

    def __len__(self):
        return self.meta["num_samples"]

    def __getitem__(self, i):
        return self.videos[i].float() / 255.0, {k: v[i] for k, v in self.ann.items()}
