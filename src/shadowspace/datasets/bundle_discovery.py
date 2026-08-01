"""Scanner for discovering and validating pre-fetched dataset bundles on disk."""

from pathlib import Path

from shadowspace.bundle.reader import BundleValidator


def scan_bundle_dir(bundle_dir: Path | str) -> dict[str, Path]:
    """Scan bundle_dir for subdirectories containing valid Shadowspace artifact bundles.

    Parameters
    ----------
    bundle_dir : Path | str
        Path to directory containing bundle subdirectories (e.g., data/bundles/).

    Returns
    -------
    dict[str, Path]
        Mapping from dataset key / bundle_id to manifest.json Path.
    """
    p = Path(bundle_dir)
    if not p.exists() or not p.is_dir():
        return {}

    discovered: dict[str, Path] = {}
    for entry in p.iterdir():
        if entry.is_dir():
            manifest_path = entry / "manifest.json"
            if manifest_path.exists():
                val_result = BundleValidator(entry).validate()
                if val_result.is_valid:
                    try:
                        # Prefer directory name for discovery key mapping
                        discovered[entry.name] = manifest_path
                    except Exception:  # pragma: no cover
                        continue

    return discovered
