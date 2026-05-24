from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from main import build_paths, run_node_featurize_ecfp4_rdkit


def test_ecfp4_rdkit_featurizer_writes_combined_labeled_features(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    paths = build_paths(str(tmp_path / "data"))
    Path(paths["split_dir"]).mkdir(parents=True, exist_ok=True)
    curated_path = Path(paths["curated"])
    pd.DataFrame(
        {
            "__row_index": [10, 11, 12],
            "canonical_smiles": ["CCO", "c1ccccc1", "CC(=O)O"],
            "label": [1, 0, 1],
        }
    ).to_csv(curated_path, index=False)

    context = {
        "paths": paths,
        "curated_path": str(curated_path),
        "target_column": "label",
        "featurize_config": {"radius": 2, "n_bits": 64},
    }

    run_node_featurize_ecfp4_rdkit(context)

    labeled = pd.read_csv(paths["ecfp4_rdkit_labeled"])
    assert context["feature_matrix"] == paths["ecfp4_rdkit_labeled"]
    assert context["labels_matrix"] == paths["ecfp4_rdkit_labeled"]
    assert context["feature_method"] == "ecfp4_rdkit"
    assert labeled["__row_index"].tolist() == [10, 11, 12]
    assert labeled["label"].tolist() == [1, 0, 1]
    assert "fp_0" in labeled.columns
    assert "fp_63" in labeled.columns
    assert "MolWt" in labeled.columns
    assert "canonical_smiles" not in labeled.columns

    metadata = json.loads(Path(paths["ecfp4_rdkit_meta"]).read_text(encoding="utf-8"))
    assert metadata["fingerprint"] == "ecfp4"
    assert metadata["morgan_radius"] == 2
    assert metadata["n_bits"] == 64
