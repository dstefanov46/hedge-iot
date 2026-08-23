from __future__ import annotations

import numpy as np
import pandas as pd

from aurora.satellite.config import CHANNEL_NAMES


def derive_patch_features(
    patch: np.ndarray, patch_uri: str | None = None, *, site_id: str | None = None,
    timestamp_utc: object | None = None, product_id: str | None = None,
) -> dict[str, float | str | None]:
    values = np.asarray(patch, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 12:
        raise ValueError("expected (12, height, width) patch")
    result: dict[str, float | str | None] = {
        "satellite_missing": float(np.isnan(values).mean()),
        "satellite_patch_uri": patch_uri,
        "site_id": site_id,
        "timestamp_utc": timestamp_utc,
        "product_id": product_id,
    }
    for i, channel in enumerate(CHANNEL_NAMES):
        sample = values[i][np.isfinite(values[i])]
        statistics = (
            {
                "mean": float(np.mean(sample)),
                "std": float(np.std(sample)),
                "min": float(np.min(sample)),
                "max": float(np.max(sample)),
            }
            if sample.size
            else {name: float("nan") for name in ("mean", "std", "min", "max")}
        )
        for suffix, value in statistics.items():
            result[f"satellite_{channel.lower()}_{suffix}"] = value
    visible = values[[0, 1, 2]]
    infrared = values[[3, 4, 5, 6, 7, 8, 9, 10]]
    visible_sample = visible[np.isfinite(visible)]
    infrared_sample = infrared[np.isfinite(infrared)]
    visible_mean = float(np.mean(visible_sample)) if visible_sample.size else float("nan")
    infrared_mean = float(np.mean(infrared_sample)) if infrared_sample.size else float("nan")
    result["satellite_cloud_index"] = visible_mean / (infrared_mean + 1e-6)
    result["satellite_irradiance_proxy"] = visible_mean
    return result


def align_satellite_features(
    frame: pd.DataFrame,
    satellite: pd.DataFrame,
    tolerance_minutes: float = 7.5,
    issue_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """As-of align products to the 15-minute grid without exposing future products."""
    required = {"site_id", "timestamp_utc"}
    if required - set(frame) or required - set(satellite):
        raise ValueError("both frames require site_id and timestamp_utc")
    left = frame.copy()
    right = satellite.copy()
    timestamp_dtype = "datetime64[ns, UTC]"
    left["timestamp_utc"] = pd.to_datetime(
        left["timestamp_utc"], format="mixed", utc=True
    ).astype(timestamp_dtype)
    right["timestamp_utc"] = pd.to_datetime(
        right["timestamp_utc"], format="mixed", utc=True
    ).astype(timestamp_dtype)
    right["satellite_sensing_timestamp_utc"] = right["timestamp_utc"]
    if issue_time is not None:
        cutoff = pd.Timestamp(issue_time)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        right = right[right["timestamp_utc"] <= cutoff]
    right = right.sort_values(["site_id", "timestamp_utc"])
    merged = pd.merge_asof(
        left.sort_values(["site_id", "timestamp_utc"]),
        right,
        on="timestamp_utc",
        by="site_id",
        direction="backward",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
        suffixes=("", "_satellite"),
    )
    # Keep missing imagery explicit and make the aligned sensing time available
    # to the TFT leakage guard. A null timestamp means no usable product.
    if "satellite_missing" not in merged:
        merged["satellite_missing"] = 1.0
    merged["satellite_missing"] = merged["satellite_missing"].fillna(1.0)
    merged["satellite_issue_timestamp_utc"] = merged.get(
        "satellite_sensing_timestamp_utc", pd.NaT
    )
    return merged.sort_values(["site_id", "timestamp_utc"]).reset_index(drop=True)


def feature_table_from_patch_records(records: list[dict[str, object]]) -> pd.DataFrame:
    """Build the canonical one-row-per-product satellite feature table."""
    rows = []
    for record in records:
        patch_uri = record.get("patch_uri")
        if not patch_uri:
            continue
        source = str(patch_uri)
        from aurora.satellite.preprocess import load_patch
        patch = load_patch(source)
        row = derive_patch_features(
            patch, source, site_id=str(record["site_id"]),
            timestamp_utc=record["timestamp_utc"], product_id=str(record["product_id"]),
        )
        row.update({"quality_status": record.get("quality_status", "ok"),
                    "normalization_version": record.get("normalization_version")})
        rows.append(row)
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    keys = ["site_id", "timestamp_utc", "product_id"]
    if output.duplicated(keys).any():
        raise ValueError("feature table contains duplicate site/timestamp/product rows")
    return output.sort_values(["site_id", "timestamp_utc"]).reset_index(drop=True)
