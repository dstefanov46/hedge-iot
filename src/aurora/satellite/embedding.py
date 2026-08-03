from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _load_patch(path: str | Path) -> np.ndarray:
    from aurora.satellite.preprocess import load_patch
    return load_patch(path)


class FrozenCNNEncoder:
    """Two-block CNN encoder with a deterministic fallback for minimal installs."""

    version = "frozen-cnn-v1"

    def __init__(self, embedding_dim: int = 32, seed: int = 42):
        self.embedding_dim, self.seed = embedding_dim, seed
        try:
            import torch
            import torch.nn as nn

            torch.manual_seed(seed)
            self._torch = torch
            self.model = nn.Sequential(
                nn.Conv2d(12, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, embedding_dim),
            )
            self.model.eval()
        except ImportError:
            self._torch, self.model = None, None
            digest = hashlib.sha256(f"{self.version}:{seed}:{embedding_dim}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            self._projection = rng.standard_normal((12, embedding_dim)).astype(np.float32)

        self.frozen = False

    def fit(
        self, patches: np.ndarray, epochs: int = 1, learning_rate: float = 1e-3
    ) -> FrozenCNNEncoder:
        """Train the encoder on training-period patches, then freeze it."""
        values = np.nan_to_num(np.asarray(patches, dtype=np.float32), nan=0.0)
        if values.ndim != 4 or values.shape[1:] != (12, 64, 64) or len(values) == 0:
            raise ValueError("fit expects non-empty patches with shape (n, 12, 64, 64)")
        if self.model is None:
            # Deterministic fallback has no learnable torch parameters.
            self.frozen = True
            return self
        torch = self._torch
        torch.manual_seed(self.seed)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        inputs = torch.from_numpy(values)
        for _ in range(max(1, epochs)):
            optimizer.zero_grad()
            output = self.model(inputs)
            # A stable self-supervised objective keeps this utility usable without labels.
            loss = (output**2).mean()
            if len(values) > 1:
                loss = loss + 0.01 * (output[1:] - output[:-1]).pow(2).mean()
            loss.backward()
            optimizer.step()
        self.freeze()
        return self

    def freeze(self) -> None:
        if self.model is not None:
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        self.frozen = True

    def save_checkpoint(self, path: str | Path, *, training_interval: dict[str, str] | None = None,
                        normalization: dict[str, object] | None = None) -> dict[str, object]:
        if not self.frozen:
            self.freeze()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self._torch.save(self.model.state_dict(), destination)
        else:
            with destination.open("wb") as handle:
                np.savez(handle, projection=self._projection)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        metadata = {**self.metadata(), "training_interval": training_interval,
                    "normalization": normalization, "checkpoint_sha256": digest}
        destination.with_suffix(destination.suffix + ".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        return metadata

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> FrozenCNNEncoder:
        source = Path(path)
        metadata = json.loads(
            source.with_suffix(source.suffix + ".json").read_text(encoding="utf-8")
        )
        encoder = cls(int(metadata.get("embedding_dim", 32)), int(metadata.get("seed", 42)))
        if encoder.model is not None:
            state = encoder._torch.load(source, map_location="cpu", weights_only=True)
            encoder.model.load_state_dict(state)
        else:
            encoder._projection = np.load(source)["projection"]
        encoder.freeze()
        return encoder

    def encode(self, patch: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(np.asarray(patch, dtype=np.float32), nan=0.0)
        if values.shape != (12, 64, 64):
            raise ValueError("encoder expects a normalized 12x64x64 patch")
        if self.model is not None:
            with self._torch.no_grad():
                return self.model(self._torch.from_numpy(values[None])).numpy()[0]
        return values.mean(axis=(1, 2)) @ self._projection

    def metadata(self) -> dict[str, object]:
        return {
            "model_version": self.version,
            "embedding_dim": self.embedding_dim,
            "seed": self.seed,
            "frozen": self.frozen,
        }


def fit_fold_encoder(
    patch_records: object,
    train_start: object,
    train_end: object,
    checkpoint_path: str | Path,
    *,
    seed: int = 42,
    normalization: dict[str, object] | None = None,
    epochs: int = 1,
) -> tuple[FrozenCNNEncoder, dict[str, object]]:
    """Fit one encoder using only records inside a fold's training interval."""
    import pandas as pd

    records = patch_records.copy()
    timestamps = pd.to_datetime(records["timestamp_utc"], utc=True)
    start, end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    selected = records[(timestamps >= start) & (timestamps <= end)]
    if selected.empty:
        raise ValueError("fold training interval contains no satellite patches")
    patches = np.stack([_load_patch(path) for path in selected["patch_uri"]])
    encoder = FrozenCNNEncoder(seed=seed).fit(patches, epochs=epochs)
    metadata = encoder.save_checkpoint(
        checkpoint_path,
        training_interval={"start": start.isoformat(), "end": end.isoformat()},
        normalization=normalization,
    )
    return encoder, metadata
