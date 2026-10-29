"""Generic on-disk dataset envelope, dataset-agnostic: a ``meta.json`` describing
how the dataset was generated, plus a lazily-mmap'd ``videos.npy``
``(N, 3, T, H, W)`` uint8 array. Any dataset built to this convention (see
e.g. ``f_toy.datasets.cubetoy.builder.build_to_disk``) can reuse
``MemmapVideos``/``load_meta`` — only the annotation schema on top differs
per dataset, and that lives in each dataset's own plugin package
(``f_toy/datasets/<name>/disk.py``).
"""
import json
from pathlib import Path

import numpy as np
import torch


class MemmapVideos:
    """Thin wrapper so ``dataset.videos[idx]`` behaves like a torch tensor
    (accepts an int or a torch/numpy index array, returns a torch uint8
    tensor) while the underlying memmap stays lazy/disk-backed — code like
    ``f_toy.sanity.check_dataset`` that indexes ``.videos`` directly works
    unmodified whether the dataset is in-RAM or disk-backed."""

    def __init__(self, memmap):
        self._mm = memmap

    def __len__(self):
        return len(self._mm)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.numpy()
        return torch.from_numpy(np.array(self._mm[idx]))  # np.array() copies -> writable, silences torch warning


def load_meta(root):
    return json.loads((Path(root) / "meta.json").read_text())


def load_memmap_videos(root):
    return MemmapVideos(np.load(Path(root) / "videos.npy", mmap_mode="r"))
