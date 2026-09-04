"""
Place recognition for the baseline: RootSIFT descriptors, a k-means codebook, VLAD vectors,
and the exploration dataset they are built from.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

MOVEMENT = {"FORWARD", "BACKWARD", "LEFT", "RIGHT"}
REVERSE = {"FORWARD": "BACKWARD", "BACKWARD": "FORWARD", "LEFT": "RIGHT", "RIGHT": "LEFT"}


@dataclass(frozen=True)
class Frame:
    path: Path
    action: str
    """The single movement action that produced the *next* frame."""
    trajectory: int


def load_frames(data_dir: Path, subsample: int = 1) -> tuple[list[Frame], list[tuple[int, int]]]:
    """Frames from ``traj_<i>/data_info.json``, keeping only pure single-action steps, then
    every ``subsample``-th one. Returns the frames and ``(start, end)`` per trajectory."""
    trajectories = sorted(
        (d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("traj_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not trajectories:
        raise FileNotFoundError(f"no traj_*/ directories under {data_dir}")

    frames: list[Frame] = []
    bounds: list[tuple[int, int]] = []
    for index, directory in enumerate(trajectories):
        records = json.loads((directory / "data_info.json").read_text())
        kept = [
            Frame(directory / r["image"], r["action"][0], index)
            for r in records
            if len(r["action"]) == 1 and r["action"][0] in MOVEMENT
        ][::subsample]
        bounds.append((len(frames), len(frames) + len(kept)))
        frames.extend(kept)
    return frames, bounds


class VLADExtractor:
    """RootSIFT + VLAD with intra-normalisation and power normalisation. Descriptors and the
    codebook are cached under ``cache_dir``; delete it to recompute."""

    def __init__(self, n_clusters: int = 128, cache_dir: Path = Path("cache")) -> None:
        self.n_clusters = n_clusters
        self.cache_dir = cache_dir
        self.sift = cv2.SIFT_create()
        self.codebook: KMeans | None = None

    @property
    def dim(self) -> int:
        return self.n_clusters * 128

    def describe(self, image: np.ndarray) -> np.ndarray | None:
        _, des = self.sift.detectAndCompute(image, None)
        if des is None or len(des) == 0:
            return None
        des = des / np.sum(des, axis=1, keepdims=True)
        return np.sqrt(des)

    def fit(self, frames: list[Frame]) -> np.ndarray:
        """Build (or load) the codebook and return the ``(N, dim)`` database."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        descriptors = self._descriptors(frames)
        codebook_file = self.cache_dir / f"codebook_k{self.n_clusters}.pkl"
        if codebook_file.exists():
            self.codebook = pickle.loads(codebook_file.read_bytes())
        else:
            stacked = np.vstack([d for d in descriptors if d is not None])
            print(f"fitting k-means (k={self.n_clusters}) on {len(stacked)} descriptors...")
            self.codebook = KMeans(
                n_clusters=self.n_clusters, n_init=3, max_iter=300, random_state=42
            ).fit(stacked)
            codebook_file.write_bytes(pickle.dumps(self.codebook))
        return np.array([self._vlad(d) for d in tqdm(descriptors, desc="VLAD")])

    def extract(self, image: np.ndarray) -> np.ndarray:
        return self._vlad(self.describe(image))

    def _descriptors(self, frames: list[Frame]) -> list[np.ndarray | None]:
        cache_file = self.cache_dir / f"sift_{len(frames)}.pkl"
        if cache_file.exists():
            cached = pickle.loads(cache_file.read_bytes())
            if list(cached) == [str(f.path) for f in frames]:
                return list(cached.values())
        described = {
            str(f.path): self.describe(cv2.imread(str(f.path))) for f in tqdm(frames, desc="SIFT")
        }
        cache_file.write_bytes(pickle.dumps(described))
        return list(described.values())

    def _vlad(self, des: np.ndarray | None) -> np.ndarray:
        assert self.codebook is not None, "call fit() first"
        if des is None:
            return np.zeros(self.dim)
        labels = self.codebook.predict(des)
        centers = self.codebook.cluster_centers_
        vlad = np.zeros((self.n_clusters, des.shape[1]))
        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                vlad[k] = np.sum(des[mask] - centers[k], axis=0)
                norm = np.linalg.norm(vlad[k])
                if norm > 0:
                    vlad[k] /= norm
        vlad = vlad.ravel()
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        norm = np.linalg.norm(vlad)
        return vlad / norm if norm > 0 else vlad
