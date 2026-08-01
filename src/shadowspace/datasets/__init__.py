"""Automated real dataset fetchers and discovery for Shadowspace."""

from shadowspace.datasets.bundle_discovery import scan_bundle_dir
from shadowspace.datasets.registry import REGISTRY, DatasetSpec

__all__ = ["REGISTRY", "DatasetSpec", "scan_bundle_dir"]
