"""Download the raw REES46 clickstream dataset from Kaggle.

Why Kaggle and not the REES46 origin: data.rees46.com serves an EXPIRED TLS
certificate (verified 2026-08-11, curl error 35 SEC_E_CERT_EXPIRED). We do not
disable certificate verification to pull 15GB from an unauthenticated host, so
Kaggle is the trusted distribution channel.

Requires ~/.kaggle/kaggle.json (see README for how to obtain it).
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)


def _assert_credentials() -> Path:
    cred = Path(os.path.expanduser("~")) / ".kaggle" / "kaggle.json"
    if not cred.exists():
        raise SystemExit(
            f"Kaggle credentials not found at {cred}\n"
            "Get them from https://www.kaggle.com/settings -> API -> "
            "'Create New Token', then place the downloaded kaggle.json there."
        )
    return cred


def list_files(dataset: str) -> list[tuple[str, int]]:
    """Return (filename, size_bytes) for every file in the dataset."""
    _assert_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    out: list[tuple[str, int]] = []
    for f in api.dataset_list_files(dataset).files:
        out.append((f.name, int(f.size) if str(f.size).isdigit() else -1))
    return out


def download(dataset: str, dest: Path, only: str | None = None) -> list[Path]:
    """Download dataset files, unzipping each and removing the archive.

    Downloads file-by-file rather than the whole dataset at once so that peak
    disk usage stays bounded - this machine has ~41GB free on D:.
    """
    _assert_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    dest.mkdir(parents=True, exist_ok=True)

    files = [f.name for f in api.dataset_list_files(dataset).files]
    if only:
        files = [f for f in files if only in f]
    if not files:
        raise SystemExit(f"No files matched in dataset {dataset!r} (filter={only!r})")

    written: list[Path] = []
    for name in files:
        target = dest / name
        if target.exists():
            log.info("already present, skipping: %s (%.2f GB)", name, target.stat().st_size / 1e9)
            written.append(target)
            continue

        log.info("downloading %s ...", name)
        api.dataset_download_file(dataset, name, path=str(dest), force=False, quiet=False)

        # Kaggle hands back either <name> or <name>.zip depending on size.
        zipped = dest / f"{name}.zip"
        if zipped.exists():
            log.info("unzipping %s ...", zipped.name)
            with zipfile.ZipFile(zipped) as zf:
                zf.extractall(dest)
            zipped.unlink()
            log.info("removed archive %s", zipped.name)

        if not target.exists():
            raise SystemExit(f"expected {target} after download but it is missing")
        log.info("ready: %s (%.2f GB)", name, target.stat().st_size / 1e9)
        written.append(target)

    return written


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=settings.kaggle_dataset)
    ap.add_argument("--only", default=None, help="substring filter, e.g. '2019-Oct'")
    ap.add_argument("--list", action="store_true", help="list files and exit")
    args = ap.parse_args()

    if args.list:
        total = 0
        for name, size in list_files(args.dataset):
            total += max(size, 0)
            log.info("%-28s %8.2f GB", name, size / 1e9 if size > 0 else float("nan"))
        log.info("TOTAL %.2f GB", total / 1e9)
        return

    paths = download(args.dataset, settings.raw_dir, only=args.only)
    log.info("downloaded %d file(s) into %s", len(paths), settings.raw_dir)


if __name__ == "__main__":
    main()
