from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def _fs(uri: str):
    try:
        import fsspec
    except ImportError:
        if "://" in uri and not uri.startswith("file://"):
            raise RuntimeError(
                "fsspec is required for non-local satellite URIs"
            ) from None
        return None, Path(uri.replace("file://", ""))
    fs, path = fsspec.core.url_to_fs(uri)
    return fs, path


def uri_exists(uri: str) -> bool:
    fs, path = _fs(uri)
    return path.exists() if fs is None else fs.exists(path)


def atomic_copy(source: str | Path, destination: str) -> None:
    fs, path = _fs(destination)
    if fs is None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            tmp = Path(handle.name)
        try:
            shutil.copyfile(source, tmp)
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        return
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(source, "rb") as src, fs.open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst)
    fs.mv(tmp, path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(uri: str, value: Any) -> None:
    fs, path = _fs(uri)
    payload = json.dumps(value, indent=2, sort_keys=True).encode()
    if fs is None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(payload)
    else:
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        if parent:
            fs.makedirs(parent, exist_ok=True)
        with fs.open(path, "wb") as handle:
            handle.write(payload)


def read_json(uri: str) -> Any:
    fs, path = _fs(uri)
    if fs is None:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    with fs.open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def backup_artifacts(source_root: str | Path, backup_root: str | Path) -> Path:
    """Copy retained satellite artifacts and write a checksum inventory."""
    source, destination = Path(source_root), Path(backup_root)
    if not source.is_dir():
        raise ValueError(f"processed satellite directory does not exist: {source}")
    source_path, destination_path = source.resolve(), destination.resolve()
    if destination_path == source_path or source_path in destination_path.parents:
        raise ValueError("backup destination must be outside the processed source directory")
    destination.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    for item in source.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        checksums[str(relative).replace("\\", "/")] = sha256_file(target)
    write_json(str(destination / "checksums.json"), checksums)
    return destination
