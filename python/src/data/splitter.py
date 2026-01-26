"""
Dataset Splitter
================

Train/Val/Test split with stratification support.

Features:
- Random split with configurable ratios
- Stratified split by scenario type/source/speed
- Reproducible splits with seed
- Split manifest saving/loading (JSON)

Usage:
    from src.data.splitter import DatasetSplitter

    splitter = DatasetSplitter(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )

    splits = splitter.split(scenario_ids)
    # splits = {"train": [...], "val": [...], "test": [...]}

    # Save split manifest
    splitter.save_manifest(splits, "datasets/splits/split_v1.json")

    # Load split manifest
    splits = splitter.load_manifest("datasets/splits/split_v1.json")
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from .base import Scenario

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    stratify_by: Optional[str] = None  # "source", "speed_bin", "agent_count_bin"
    min_samples_per_stratum: int = 3


class DatasetSplitter:
    """
    Splits dataset into train/val/test sets.
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        stratify_by: Optional[str] = None,
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        self.config = SplitConfig(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            stratify_by=stratify_by,
        )

    def split(
        self,
        scenario_ids: List[str],
        scenarios: Optional[List[Scenario]] = None,
    ) -> Dict[str, List[str]]:
        """
        Split scenario IDs into train/val/test.

        Args:
            scenario_ids: List of scenario IDs to split
            scenarios: Optional list of Scenario objects (for stratification)

        Returns:
            Dict with "train", "val", "test" keys
        """
        rng = np.random.RandomState(self.config.seed)

        if self.config.stratify_by and scenarios:
            return self._stratified_split(scenario_ids, scenarios, rng)
        else:
            return self._random_split(scenario_ids, rng)

    def _random_split(
        self,
        scenario_ids: List[str],
        rng: np.random.RandomState,
    ) -> Dict[str, List[str]]:
        """Simple random split."""
        n = len(scenario_ids)
        indices = rng.permutation(n)

        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        result = {
            "train": [scenario_ids[i] for i in train_idx],
            "val": [scenario_ids[i] for i in val_idx],
            "test": [scenario_ids[i] for i in test_idx],
        }

        self._log_split_stats(result)
        return result

    def _stratified_split(
        self,
        scenario_ids: List[str],
        scenarios: List[Scenario],
        rng: np.random.RandomState,
    ) -> Dict[str, List[str]]:
        """Stratified split maintaining distribution across strata."""
        # Get stratum for each scenario
        strata = self._compute_strata(scenarios)

        # Group by stratum
        stratum_groups: Dict[str, List[int]] = defaultdict(list)
        for i, stratum in enumerate(strata):
            stratum_groups[stratum].append(i)

        # Split within each stratum
        train_ids, val_ids, test_ids = [], [], []

        for stratum, indices in stratum_groups.items():
            rng.shuffle(indices)
            n = len(indices)

            if n < self.config.min_samples_per_stratum:
                # Too few samples, put all in train
                train_ids.extend([scenario_ids[i] for i in indices])
                continue

            n_train = max(1, int(n * self.config.train_ratio))
            n_val = max(1, int(n * self.config.val_ratio))

            train_ids.extend([scenario_ids[i] for i in indices[:n_train]])
            val_ids.extend([scenario_ids[i] for i in indices[n_train:n_train + n_val]])
            test_ids.extend([scenario_ids[i] for i in indices[n_train + n_val:]])

        result = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }

        self._log_split_stats(result)
        return result

    def _compute_strata(self, scenarios: List[Scenario]) -> List[str]:
        """Compute stratum label for each scenario."""
        strata = []

        for s in scenarios:
            if self.config.stratify_by == "source":
                strata.append(s.source)

            elif self.config.stratify_by == "speed_bin":
                speed = np.sqrt(s.ego_trajectory[:, 2]**2 + s.ego_trajectory[:, 3]**2)
                avg_speed = np.mean(speed)
                if avg_speed < 5:
                    strata.append("slow")
                elif avg_speed < 15:
                    strata.append("medium")
                else:
                    strata.append("fast")

            elif self.config.stratify_by == "agent_count_bin":
                n_agents = len(s.agents)
                if n_agents < 5:
                    strata.append("sparse")
                elif n_agents < 15:
                    strata.append("moderate")
                else:
                    strata.append("dense")

            else:
                strata.append("default")

        return strata

    def _log_split_stats(self, splits: Dict[str, List[str]]):
        """Log split statistics."""
        total = sum(len(v) for v in splits.values())
        for split_name, ids in splits.items():
            ratio = len(ids) / total if total > 0 else 0
            logger.info(f"  {split_name}: {len(ids)} scenarios ({ratio:.1%})")

    def save_manifest(
        self,
        splits: Dict[str, List[str]],
        path: str,
        metadata: Optional[Dict] = None,
    ):
        """
        Save split manifest to JSON.

        Args:
            splits: Split dictionary
            path: Output file path
            metadata: Optional metadata to include
        """
        manifest = {
            "config": {
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "test_ratio": self.config.test_ratio,
                "seed": self.config.seed,
                "stratify_by": self.config.stratify_by,
            },
            "stats": {
                split: len(ids) for split, ids in splits.items()
            },
            "splits": splits,
        }

        if metadata:
            manifest["metadata"] = metadata

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved split manifest to {output_path}")

    @staticmethod
    def load_manifest(path: str) -> Dict[str, List[str]]:
        """
        Load split manifest from JSON.

        Args:
            path: Path to manifest file

        Returns:
            Split dictionary
        """
        with open(path, 'r') as f:
            manifest = json.load(f)

        splits = manifest["splits"]
        stats = manifest.get("stats", {})

        logger.info(f"Loaded split manifest from {path}")
        for split_name, count in stats.items():
            logger.info(f"  {split_name}: {count} scenarios")

        return splits


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate dummy scenario IDs
    ids = [f"scenario_{i:04d}" for i in range(1000)]

    # Random split
    splitter = DatasetSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    splits = splitter.split(ids)

    print(f"Train: {len(splits['train'])}")
    print(f"Val: {len(splits['val'])}")
    print(f"Test: {len(splits['test'])}")
    print(f"Total: {sum(len(v) for v in splits.values())}")

    # Save manifest
    splitter.save_manifest(splits, "datasets/splits/test_split.json", metadata={
        "dataset": "test",
        "version": "0.1",
    })

    # Load manifest
    loaded = DatasetSplitter.load_manifest("datasets/splits/test_split.json")
    assert loaded["train"] == splits["train"]
    print("\nManifest save/load verified!")
