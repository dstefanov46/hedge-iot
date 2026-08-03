# EUMETSAT satellite ingestion

AURORA uses the `EO:EUM:DAT:MSG:HRSEVIRI` collection. An EUMETSAT account and the
appropriate data licence are required. Copy `configs/satellite.example.toml` to
`configs/satellite.toml`, fill in the credentials, and never commit that local
file or place credentials in logs.

The default local layout is `data/raw/satellite/native/YYYY/MM/DD/` for native
products and `data/processed/satellite/` for patches, manifests, and features.
Set `raw_uri` and `processed_uri` in the TOML file to S3-compatible URIs when
using object storage and install `.[satellite]`.

```powershell
python -m pip install -e ".[satellite]"
aurora satellite-download --config configs/satellite.toml --start 2025-06-01T00:00:00Z --end 2025-06-01T01:00:00Z --site-config configs/site_hirvensalmi.json
aurora satellite-collect --config configs/satellite.toml --site-config configs/site_hirvensalmi.json
aurora satellite-preprocess --config configs/satellite.toml --manifest data/raw/satellite/native/manifest.json --site-config configs/site_hirvensalmi.json
aurora satellite-embed --patch-store data/processed/satellite/patches --output data/processed/satellite/features.parquet
aurora satellite-batch --config configs/satellite.toml --site-config configs/site_hirvensalmi.json --start 2025-01-01T00:00:00Z --end 2025-01-02T00:00:00Z
aurora satellite-batch --config configs/satellite.toml --site-config configs/site_hirvensalmi.json --start 2025-01-01T00:00:00Z --end 2025-01-02T00:00:00Z --delete-native
```

`satellite-collect` uses the configured 2025 calendar window and bounded chunks;
rerunning it is safe because the raw manifest is merged by product ID. Strict
collection mode records invalid ZIPs and download errors for later repair. Fit
`normalization_stats_uri` from each fold's training patches and pass that file to
preprocessing; do not use complete-period statistics for a fold.

`satellite-batch` retains validated physical-unit patches in
`processed/satellite/physical_patches/{product_id}.zarr` and writes append-only
state transitions beside the raw manifest. Failed products remain available for
repair. Native deletion requires a verified `backup_uri` outside both raw and
processed roots; `--dry-run` reports eligible archives without deleting them.

The decoder harmonizes 11 non-HRV bands and HRV onto a common site-centered
grid, extracts a 64x64 patch by default, and records channel count, dimensions,
NaN fraction, normalization version, and product provenance. Missing imagery is
represented by zero-filled TFT inputs plus `satellite_missing`; future decoder
positions are never populated from products after the forecast issue time.

Historical `F:\MAG\satip` archives can be used as external test data by pointing
the raw URI at a copy outside this repository. They are intentionally not part
of Git.
