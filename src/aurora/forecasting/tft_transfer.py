from __future__ import annotations

import importlib.metadata
import json
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurora.forecasting.baselines import enforce_monotonic_quantiles

QUANTILES = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)
KNOWN_REALS = (
    "year_sin_1",
    "year_sin_2",
    "year_cos_1",
    "year_cos_2",
    "day_sin_1",
    "day_sin_2",
    "day_cos_1",
    "day_cos_2",
    "solar_azimuth",
    "solar_elevation",
    "clear_sky_norm",
)
STATIC_REALS = ("latitude", "longitude", "installed_capacity_mw")
SATELLITE_REALS = tuple(
    [f"satellite_embedding_{i:02d}" for i in range(32)]
    + ["satellite_missing", "satellite_cloud_index", "satellite_irradiance_proxy"]
)
TARGETS = ("target_point", "target_quantiles")


@dataclass(frozen=True)
class TFTTransferConfig:
    schema_version: str = "tft-transfer-v1"
    encoder_length: int = 8
    horizon_steps: int = 8
    hidden_size: int = 32
    attention_head_size: int = 4
    hidden_continuous_size: int = 8
    dropout: float = 0.1
    batch_size: int = 128
    max_epochs: int = 50
    patience: int = 5
    learning_rate: float = 1e-4
    gradient_clip_val: float = 0.1
    max_sequences_per_epoch: int = 10_000
    daylight_weight: float = 1.0
    night_weight: float = 0.1
    random_seed: int = 42
    accelerator: str = "cuda"
    num_workers: int = 0
    satellite_enabled: bool = False
    satellite_embedding_dim: int = 32
    satellite_missing_value: float = 0.0
    satellite_max_alignment_minutes: float = 7.5


def prepare_tft_frame(frame: pd.DataFrame, config: TFTTransferConfig) -> pd.DataFrame:
    satellite_columns = set(SATELLITE_REALS) if config.satellite_enabled else set()
    required = {
        "site_id",
        "time_idx",
        "capacity_factor",
        "observation_available",
        *KNOWN_REALS,
        *STATIC_REALS,
        *satellite_columns,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing TFT columns: {sorted(missing)}")
    out = frame.copy().sort_values(["site_id", "time_idx"]).reset_index(drop=True)
    if config.satellite_enabled and "satellite_issue_timestamp_utc" in out:
        out = mask_future_satellite_values(out)
    target = pd.to_numeric(out["capacity_factor"], errors="coerce")
    out["target_point"] = target.fillna(0).clip(0, 1).astype(float)
    out["target_quantiles"] = out["target_point"]
    observed = out["observation_available"].astype(bool)
    daylight = out["solar_elevation"] > 0
    out["loss_weight"] = np.where(
        observed, np.where(daylight, config.daylight_weight, config.night_weight), 0.0
    ).astype(float)
    out["observation_available_real"] = observed.astype(float)
    if config.satellite_enabled:
        # The frame builder is responsible for masking future values; this is a
        # defensive fill for sparse products and makes missingness explicit.
        for column in SATELLITE_REALS:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(
                config.satellite_missing_value
            )
    out["site_id"] = out["site_id"].astype(str)
    return out


def mask_future_satellite_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Zero satellite values after each row's forecast issue time.

    This helper is intentionally explicit so callers constructing decoder rows
    can apply the same leakage guard before creating a TimeSeriesDataSet.
    """
    required = {"timestamp_utc", "satellite_issue_timestamp_utc"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing satellite leakage-guard columns: {sorted(missing)}")
    out = frame.copy()
    timestamps = pd.to_datetime(out["timestamp_utc"], utc=True)
    issue = pd.to_datetime(out["satellite_issue_timestamp_utc"], utc=True)
    future = timestamps > issue
    for column in SATELLITE_REALS:
        if column in out:
            out.loc[future, column] = 0.0
    if "satellite_missing" in out:
        out.loc[future, "satellite_missing"] = 1.0
    return out


def create_dataset(
    frame: pd.DataFrame,
    config: TFTTransferConfig,
    template: Any | None = None,
    min_prediction_idx: int | None = None,
    predict: bool = False,
) -> Any:
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer

    model_frame = prepare_tft_frame(frame, config)
    if template is not None:
        return TimeSeriesDataSet.from_dataset(
            template,
            model_frame,
            min_prediction_idx=min_prediction_idx,
            predict=predict,
            stop_randomization=True,
        )
    normalizers = [
        TorchNormalizer(method="identity", center=False),
        TorchNormalizer(method="identity", center=False),
    ]
    known_reals = [*KNOWN_REALS, *(SATELLITE_REALS if config.satellite_enabled else ())]
    return TimeSeriesDataSet(
        model_frame,
        time_idx="time_idx",
        target=list(TARGETS),
        group_ids=["site_id"],
        weight="loss_weight",
        min_encoder_length=config.encoder_length,
        max_encoder_length=config.encoder_length,
        min_prediction_length=config.horizon_steps,
        max_prediction_length=config.horizon_steps,
        min_prediction_idx=min_prediction_idx,
        static_reals=list(STATIC_REALS),
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=[*TARGETS, "observation_available_real"],
        target_normalizer=normalizers,
        categorical_encoders={"site_id": NaNLabelEncoder(add_nan=True)},
        allow_missing_timesteps=True,
        constant_fill_strategy={
            "target_point": 0.0,
            "target_quantiles": 0.0,
            "observation_available_real": 0.0,
            "loss_weight": 0.0,
            **{column: config.satellite_missing_value for column in SATELLITE_REALS},
        },
        add_relative_time_idx=True,
        add_encoder_length=True,
        randomize_length=False,
        predict_mode=predict,
    )


def create_model(dataset: Any, config: TFTTransferConfig) -> Any:
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import MultiLoss, QuantileLoss

    from aurora.forecasting.losses import PointMSE

    loss = MultiLoss([PointMSE(), QuantileLoss(quantiles=list(QUANTILES))])
    return TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        output_size=[1, len(QUANTILES)],
        loss=loss,
        reduce_on_plateau_patience=2,
        log_interval=-1,
    )


def transfer_weights(pretrained_checkpoint: Path, dataset: Any, config: TFTTransferConfig) -> Any:
    from pytorch_forecasting import TemporalFusionTransformer

    pretrained = TemporalFusionTransformer.load_from_checkpoint(
        str(pretrained_checkpoint), map_location="cpu"
    )
    model = create_model(dataset, config)
    incompatible = model.load_state_dict(pretrained.state_dict(), strict=False)
    allowed = ("loss.", "logging_metrics.")
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith(allowed)]
    missing = [key for key in incompatible.missing_keys if not key.startswith(allowed)]
    if unexpected or missing:
        raise ValueError(
            "Checkpoint architecture is incompatible with tft-transfer-v1: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}. "
            "Existing one-step checkpoints are legacy artifacts."
        )
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model


def fit_tft(
    train_dataset: Any,
    validation_dataset: Any,
    config: TFTTransferConfig,
    checkpoint_dir: Path,
    pretrained_checkpoint: Path | None = None,
    fixed_epochs: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    seed_everything(config.random_seed, workers=True)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    if config.accelerator in {"cuda", "gpu"}:
        torch.set_float32_matmul_precision("high")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_loader = _training_loader(train_dataset, config)
    val_loader = validation_dataset.to_dataloader(
        train=False,
        batch_size=config.batch_size * 2,
        num_workers=config.num_workers,
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="tft-{epoch:02d}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    callbacks: list[Any] = [checkpoint]
    if fixed_epochs is None:
        callbacks.append(
            EarlyStopping(monitor="val_loss", mode="min", patience=config.patience, min_delta=1e-5)
        )
    trainer = Trainer(
        max_epochs=fixed_epochs or config.max_epochs,
        accelerator=config.accelerator,
        devices=1,
        # TFT's variable-selection interpolation uses a CUDA backward kernel
        # without a strict deterministic implementation. "warn" retains seeded
        # deterministic behavior wherever supported without aborting GPU runs.
        deterministic="warn",
        gradient_clip_val=config.gradient_clip_val,
        callbacks=callbacks,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    model = (
        transfer_weights(pretrained_checkpoint, train_dataset, config)
        if pretrained_checkpoint is not None
        else create_model(train_dataset, config)
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_path = checkpoint.best_model_path
    best_model = model
    if best_path:
        # Loading through the class requires PointMSE to remain importable. The
        # in-memory model plus best state dict avoids custom-metric pickle coupling.
        payload = torch.load(best_path, map_location="cpu", weights_only=False)
        best_model.load_state_dict(payload["state_dict"])
    metadata = {
        "best_model_path": best_path,
        "best_validation_loss": float(checkpoint.best_model_score)
        if checkpoint.best_model_score is not None
        else None,
        "best_epoch": _checkpoint_epoch(best_path),
        "epochs_completed": int(trainer.current_epoch),
        "pretrained_checkpoint": str(pretrained_checkpoint) if pretrained_checkpoint else None,
    }
    return best_model, metadata


def predict_long(
    model: Any,
    dataset: Any,
    source_frame: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    loader = dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    raw = model.predict(loader, mode="raw", return_x=True)
    prediction = raw.output.prediction
    if not isinstance(prediction, (list, tuple)) or len(prediction) != 2:
        raise ValueError("Expected separate point and quantile TFT output heads.")
    point = np.clip(prediction[0].detach().cpu().numpy()[..., 0], 0, 1)
    quantiles = enforce_monotonic_quantiles(prediction[1].detach().cpu().numpy())
    decoder_idx = raw.x["decoder_time_idx"].detach().cpu().numpy().astype(np.int64)
    lookup = source_frame.drop_duplicates("time_idx").set_index("time_idx")
    rows: list[dict[str, Any]] = []
    for sample in range(len(decoder_idx)):
        issue_idx = int(decoder_idx[sample, 0] - 1)
        issue_timestamp = (
            lookup.loc[issue_idx, "timestamp_utc"] if issue_idx in lookup.index else pd.NaT
        )
        for step in range(decoder_idx.shape[1]):
            target_idx = int(decoder_idx[sample, step])
            if target_idx not in lookup.index:
                continue
            target = lookup.loc[target_idx]
            row: dict[str, Any] = {
                "site_id": str(target["site_id"]),
                "issue_timestamp_utc": issue_timestamp,
                "target_timestamp_utc": target["timestamp_utc"],
                "horizon_step": step + 1,
                "actual_value": target["capacity_factor"],
                "solar_elevation": target["solar_elevation"],
                "point_forecast": float(point[sample, step]),
            }
            for index, quantile in enumerate(QUANTILES):
                row[f"q{int(quantile * 100):02d}"] = float(quantiles[sample, step, index])
            rows.append(row)
    return pd.DataFrame(rows)


def save_model_bundle(
    directory: Path,
    model: Any,
    dataset: Any,
    config: TFTTransferConfig,
    metadata: dict[str, Any],
    source_hashes: dict[str, str],
    capacity_metadata: dict[str, float] | None = None,
) -> None:
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, directory / "model_state.pt")
    torch.save(dataset.get_parameters(), directory / "dataset_parameters.pt")
    manifest = {
        "schema_version": config.schema_version,
        "legacy_one_step_checkpoints_compatible": False,
        "config": asdict(config),
        "features": {
            "known_reals": list(KNOWN_REALS)
            + (list(SATELLITE_REALS) if config.satellite_enabled else []),
            "static_reals": list(STATIC_REALS),
            "targets": list(TARGETS),
            "site_id_is_model_feature": False,
            "quantiles": list(QUANTILES),
        },
        "training": metadata,
        "source_hashes": source_hashes,
        "capacity_metadata_mw": capacity_metadata or {},
        "dependencies": dependency_versions(),
        "platform": platform.platform(),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def dependency_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "torch", "pytorch-forecasting", "lightning")
    return {name: importlib.metadata.version(name) for name in packages}


def _training_loader(dataset: Any, config: TFTTransferConfig) -> Any:
    import torch

    sampler = None
    if len(dataset) > config.max_sequences_per_epoch:
        generator = torch.Generator().manual_seed(config.random_seed)
        sampler = torch.utils.data.RandomSampler(
            dataset,
            replacement=True,
            num_samples=config.max_sequences_per_epoch,
            generator=generator,
        )
    kwargs: dict[str, Any] = {}
    if sampler is not None:
        kwargs.update(sampler=sampler, shuffle=False)
    return dataset.to_dataloader(
        train=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        **kwargs,
    )


def _checkpoint_epoch(path: str) -> int | None:
    if not path:
        return None
    name = Path(path).name
    try:
        return int(name.split("epoch=")[1].split("-")[0]) + 1
    except (IndexError, ValueError):
        return None
