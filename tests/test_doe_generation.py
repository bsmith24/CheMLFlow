from __future__ import annotations

import builtins
import json
import math
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from utilities import doe as doe_module
from utilities.doe import DOEGenerationError, generate_doe


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeCategoricalDistribution:
    def __init__(self, choices):
        self.choices = list(choices)


class _FakeIntDistribution:
    def __init__(self, *, low, high, step=1, log=False):
        self.low = low
        self.high = high
        self.step = step
        self.log = log


class _FakeFloatDistribution:
    def __init__(self, *, low, high, step=None, log=False):
        self.low = low
        self.high = high
        self.step = step
        self.log = log


class _FakeTPESampler:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)


class _FakeOptunaStudy:
    def __init__(self, sampler):
        self.sampler = sampler

    def ask(self, *, fixed_distributions):
        params = {}
        for name, dist in fixed_distributions.items():
            params[name] = self._sample(dist)
        return SimpleNamespace(params=params)

    def _sample(self, dist):
        rng = self.sampler.rng
        if isinstance(dist, _FakeCategoricalDistribution):
            return rng.choice(dist.choices)
        if isinstance(dist, _FakeIntDistribution):
            if dist.log:
                value = math.exp(rng.uniform(math.log(dist.low), math.log(dist.high)))
                return max(dist.low, min(dist.high, int(round(value))))
            choices = list(range(dist.low, dist.high + 1, dist.step))
            return rng.choice(choices)
        if isinstance(dist, _FakeFloatDistribution):
            if dist.log:
                value = math.exp(rng.uniform(math.log(dist.low), math.log(dist.high)))
            else:
                value = rng.uniform(dist.low, dist.high)
            if dist.step is not None:
                value = dist.low + round((value - dist.low) / dist.step) * dist.step
            return max(dist.low, min(dist.high, float(value)))
        raise AssertionError(f"Unknown fake distribution: {dist!r}")


class _FakeOptunaModule:
    distributions = SimpleNamespace(
        CategoricalDistribution=_FakeCategoricalDistribution,
        IntDistribution=_FakeIntDistribution,
        FloatDistribution=_FakeFloatDistribution,
    )
    samplers = SimpleNamespace(
        TPESampler=_FakeTPESampler,
    )

    @staticmethod
    def create_study(*, direction, sampler):
        assert direction in {"maximize", "minimize"}
        return _FakeOptunaStudy(sampler)


def _patch_fake_optuna(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        doe_module,
        "_import_optuna_for_model_search",
        lambda model_type: _FakeOptunaModule,
    )


def _pgp_dataset_path() -> str:
    return str(REPO_ROOT / "tests" / "fixtures" / "data" / "pgp_small.csv")


def _flash_dataset_path() -> str:
    return str(REPO_ROOT / "tests" / "fixtures" / "data" / "flash_small.csv")


def _base_clf_doe(tmp_path: Path) -> dict:
    return {
        "version": 1,
        "dataset": {
            "profile": "clf_local_csv",
            "name": "pgp_small_doe",
            "task_type": "classification",
            "target_column": "label",
            "source": {"type": "local_csv", "path": _pgp_dataset_path()},
            "smiles_column": "SMILES",
            "label_source_column": "Activity",
            "label_map": {
                "positive": [1, "1", "active"],
                "negative": [0, "0", "inactive"],
            },
            "curate": {
                "properties": "Activity",
                "smiles_column": "SMILES",
                "dedupe_strategy": "drop_conflicts",
            },
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.morgan"],
            "pipeline.preprocess": [False],
            "pipeline.select": [False],
            "pipeline.explain": [True],
            "train.model.type": ["catboost_classifier"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "data"),
            "global.runs.enabled": False,
            "global.random_state": 42,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
            "split.stratify": True,
            "split.require_disjoint": True,
            "split.require_full_test_coverage": True,
            "train.tuning.method": "fixed",
            "train.reporting.plot_split_performance": True,
        },
        "output": {"dir": str(tmp_path / "generated")},
    }


def test_generate_doe_skips_invalid_model_task_combos(tmp_path: Path) -> None:
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv",
            "name": "flash_reg_doe",
            "task_type": "regression",
            "target_column": "FP Exp.",
            "source": {"type": "local_csv", "path": _flash_dataset_path()},
            "smiles_column": "SMILES",
            "curate": {
                "properties": "FP Calc.",
                "smiles_column": "SMILES",
                "dedupe_strategy": "first",
                "keep_all_columns": True,
            },
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.none"],
            "pipeline.preprocess": [False],
            "pipeline.select": [False],
            "pipeline.explain": [False],
            "train.model.type": ["random_forest", "catboost_classifier"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "data"),
            "global.runs.enabled": False,
            "global.random_state": 42,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
            "split.stratify": False,
            "split.require_disjoint": True,
            "split.require_full_test_coverage": True,
            "train.tuning.method": "fixed",
        },
        "output": {"dir": str(tmp_path / "generated")},
    }

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["total_cases"] == 2
    assert summary["valid_cases"] == 1
    assert summary["skipped_cases"] == 1
    assert summary["issue_counts"].get("DOE_MODEL_TASK_MISMATCH", 0) == 1
    assert len(result["valid_cases"]) == 1
    assert Path(result["valid_cases"][0]["config_path"]).exists()


def test_generate_doe_allows_tree_models_for_classification(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = [
        "catboost_classifier",
        "random_forest",
        "decision_tree",
        "xgboost",
        "svm",
        "ensemble",
    ]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["total_cases"] == 6
    assert summary["valid_cases"] == 6
    assert summary["skipped_cases"] == 0


def test_generate_doe_requires_validation_split_for_chemprop(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["smiles_native"]
    spec["search_space"]["train.model.type"] = ["chemprop"]
    spec["defaults"]["split.val_size"] = 0.0

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["skipped_cases"] == 1
    assert summary["issue_counts"].get("DOE_VALIDATION_SPLIT_REQUIRED", 0) == 1


def test_generate_doe_propagates_chemprop_legacy_split_flag(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["smiles_native"]
    spec["search_space"]["pipeline.explain"] = [False]
    spec["search_space"]["train.model.type"] = ["chemprop"]
    spec["defaults"]["train.model.allow_legacy_split_positions"] = True

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["train"]["model"]["type"] == "chemprop"
    assert config["train"]["model"]["allow_legacy_split_positions"] is True


def test_generate_doe_allows_chemprop_with_noop_preprocess_and_no_features(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["smiles_native"]
    spec["search_space"]["pipeline.preprocess"] = [True]
    spec["search_space"]["train.model.type"] = ["chemprop"]
    spec["defaults"]["preprocess.scaler"] = "none"
    spec["defaults"]["split.val_size"] = 0.1

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "preprocess.features" in config["pipeline"]["nodes"]
    assert "featurize.none" not in config["pipeline"]["nodes"]
    assert config["preprocess"]["scaler"] == "none"


def test_generate_doe_skips_chemprop_with_non_noop_preprocess(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["smiles_native"]
    spec["search_space"]["pipeline.preprocess"] = [True]
    spec["search_space"]["train.model.type"] = ["chemprop"]
    spec["defaults"]["preprocess.scaler"] = "standard"
    spec["defaults"]["split.val_size"] = 0.1

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_CHEMPROP_PREPROCESS_UNSUPPORTED", 0) == 1
    assert summary["issue_counts"].get("DOE_RUNTIME_SCHEMA_INVALID", 0) == 0


def test_generate_doe_allows_chemprop_for_regression_local_csv(tmp_path: Path) -> None:
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv",
            "name": "flash_reg_chemprop",
            "task_type": "regression",
            "target_column": "FP Exp.",
            "source": {"type": "local_csv", "path": _flash_dataset_path()},
            "smiles_column": "SMILES",
            "curate": {
                "properties": "FP Exp.",
                "smiles_column": "SMILES",
                "dedupe_strategy": "first",
                "keep_all_columns": True,
            },
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["smiles_native"],
            "pipeline.preprocess": [False],
            "pipeline.select": [False],
            "pipeline.explain": [False],
            "train.model.type": ["chemprop"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "data_reg_chemprop"),
            "global.runs.enabled": False,
            "global.random_state": 42,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
            "split.stratify": False,
            "split.require_disjoint": True,
            "split.require_full_test_coverage": True,
            "train.tuning.method": "fixed",
        },
        "output": {"dir": str(tmp_path / "generated_reg_chemprop")},
    }

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    assert summary["skipped_cases"] == 0
    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["global"]["task_type"] == "regression"
    assert config["train"]["model"]["type"] == "chemprop"
    assert "featurize.none" not in config["pipeline"]["nodes"]
    assert config["pipeline"]["feature_input"] == "smiles_native"


def test_generate_doe_skips_chemprop_with_tabular_feature_input(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["featurize.rdkit"]
    spec["search_space"]["train.model.type"] = ["chemprop"]
    spec["defaults"]["split.val_size"] = 0.1

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_CHEMPROP_FEATURE_INPUT_UNSUPPORTED", 0) == 1


def test_generate_doe_skips_chemprop_with_featurize_none_alias(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["featurize.none"]
    spec["search_space"]["train.model.type"] = ["chemprop"]
    spec["defaults"]["split.val_size"] = 0.1

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_CHEMPROP_FEATURE_INPUT_UNSUPPORTED", 0) == 1
    assert summary["issue_counts"].get("DOE_RUNTIME_SCHEMA_INVALID", 0) == 0


def test_generate_doe_allows_chemprop_with_smiles_native_feature_input(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["smiles_native"]
    spec["search_space"]["train.model.type"] = ["chemprop"]
    spec["search_space"]["pipeline.preprocess"] = [True]
    spec["defaults"]["preprocess.scaler"] = "none"
    spec["defaults"]["split.val_size"] = 0.1

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["pipeline"]["feature_input"] == "smiles_native"
    assert "preprocess.features" in config["pipeline"]["nodes"]
    assert "featurize.rdkit" not in config["pipeline"]["nodes"]
    assert "featurize.morgan" not in config["pipeline"]["nodes"]


def test_generate_doe_skips_tabular_models_with_smiles_native_feature_input(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["smiles_native"]
    spec["search_space"]["train.model.type"] = ["random_forest"]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_SMILES_NATIVE_MODEL_UNSUPPORTED", 0) == 1
    assert summary["issue_counts"].get("DOE_RUNTIME_SCHEMA_INVALID", 0) == 0


def test_generate_doe_allows_chemeleon_with_smiles_native_feature_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "chemeleon_mp.pt"
    checkpoint.write_bytes(b"fake-checkpoint")
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["smiles_native"]
    spec["search_space"]["pipeline.preprocess"] = [True]
    spec["search_space"]["train.model.type"] = ["chemeleon"]
    spec["defaults"]["preprocess.scaler"] = "none"
    spec["defaults"]["train.model.foundation_checkpoint"] = str(checkpoint)
    spec["defaults"]["split.val_size"] = 0.1

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["train"]["model"]["type"] == "chemeleon"
    assert config["train"]["model"]["foundation"] == "chemeleon"
    assert config["train"]["model"]["foundation_checkpoint"] == str(checkpoint)
    assert config["pipeline"]["feature_input"] == "smiles_native"


def test_generate_doe_requires_chemeleon_checkpoint(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["smiles_native"]
    spec["search_space"]["train.model.type"] = ["chemeleon"]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_CHEMELEON_CHECKPOINT_REQUIRED", 0) == 1


def test_generate_doe_enforces_split_mode_strategy_compatibility(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["split.mode"] = ["cv"]
    spec["search_space"]["split.strategy"] = ["tdc_scaffold"]
    spec["defaults"]["split.cv.n_splits"] = 5
    spec["defaults"]["split.cv.repeats"] = 1
    spec["defaults"]["split.cv.fold_index"] = 0
    spec["defaults"]["split.cv.repeat_index"] = 0

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_SPLIT_STRATEGY_MODE_INVALID", 0) == 1


def test_generate_doe_rejects_execution_only_axes_in_search_space(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["split.mode"] = ["cv"]
    spec["defaults"]["split.cv.n_splits"] = 3
    spec["defaults"]["split.cv.repeats"] = 1
    spec["search_space"]["split.cv.fold_index"] = [0]

    with pytest.raises(
        DOEGenerationError,
        match="Execution-only split axes must not be placed in search_space",
    ):
        generate_doe(spec)


def test_generate_doe_auto_expands_cv_execution_axes_when_omitted(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["split.mode"] = ["cv"]
    spec["search_space"]["split.strategy"] = ["random", "scaffold"]
    spec["defaults"]["split.cv.n_splits"] = 3
    spec["defaults"]["split.cv.repeats"] = 2

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["total_cases"] == 12
    assert summary["valid_cases"] == 12
    assert summary["skipped_cases"] == 0
    assert summary["total_parent_cases"] == 2
    assert summary["valid_parent_cases"] == 2
    assert summary["total_execution_cases"] == 12
    assert len(result["parent_cases"]) == 2

    first = result["valid_cases"][0]
    assert first["parent_case_id"] == "parent_0001"
    assert first["execution_count"] == 6
    assert first["execution_label"] == "rep0_fold0"
    assert first["scientific_config_id"]
    assert "split.cv.fold_index" not in first["factors"]
    assert "split.cv.repeat_index" not in first["factors"]
    assert set(first["execution_factors"]) == {"split.cv.fold_index", "split.cv.repeat_index"}

    parent_manifest_lines = Path(result["parent_manifest_path"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(parent_manifest_lines) == 2
    first_parent = json.loads(parent_manifest_lines[0])
    assert first_parent["record_type"] == "parent"
    assert first_parent["case_id"] == "parent_0001"
    assert first_parent["execution_count"] == 6
    assert len(first_parent["valid_execution_case_ids"]) == 6
    assert len(first_parent["execution_case_ids"]) == len(first_parent["execution_labels"]) == 6
    assert first_parent["status"] == "valid"

    seen_execution = {
        (
            int(case["execution_factors"]["split.cv.repeat_index"]),
            int(case["execution_factors"]["split.cv.fold_index"]),
        )
        for case in result["valid_cases"]
        if case["factors"]["split.strategy"] == "random"
    }
    assert seen_execution == {
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    }


def test_generate_doe_model_search_grid_expands_params_as_cv_parents(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["split.mode"] = ["cv"]
    spec["search_space"]["split.strategy"] = ["random"]
    spec["search_space"]["train.model.type"] = ["random_forest"]
    spec["defaults"]["split.cv.n_splits"] = 2
    spec["defaults"]["split.cv.repeats"] = 1
    spec["model_search"] = {
        "random_forest": {
            "method": "grid",
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, None],
            },
        }
    }

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["total_parent_cases"] == 4
    assert summary["total_execution_cases"] == 8
    assert summary["valid_cases"] == 8
    assert len(result["parent_cases"]) == 4

    first_parent = result["parent_cases"][0]
    assert first_parent["execution_count"] == 2
    assert first_parent["execution_labels"] == ["rep0_fold0", "rep0_fold1"]
    assert first_parent["factors"]["train.model.params.n_estimators"] == 100
    assert first_parent["factors"]["train.model.params.max_depth"] == 3
    parent_scientific_ids = {case["scientific_config_id"] for case in result["parent_cases"]}
    assert len(parent_scientific_ids) == 4

    first_child = result["valid_cases"][0]
    config = yaml.safe_load(Path(first_child["config_path"]).read_text(encoding="utf-8"))
    assert config["train"]["model"]["params"] == {"max_depth": 3, "n_estimators": 100}
    assert "model_search" not in config
    assert "split.cv.fold_index" not in first_child["factors"]
    assert set(first_child["execution_factors"]) == {"split.cv.fold_index", "split.cv.repeat_index"}
    for parent in result["parent_cases"]:
        child_scientific_ids = {
            case["scientific_config_id"]
            for case in result["valid_cases"]
            if case["parent_case_id"] == parent["case_id"]
        }
        assert child_scientific_ids == {parent["scientific_config_id"]}


def test_generate_doe_model_search_applies_only_to_matching_model(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["random_forest", "xgboost"]
    spec["model_search"] = {
        "random_forest": {
            "method": "grid",
            "params": {"n_estimators": [50, 100]},
        }
    }

    result = generate_doe(spec)

    assert result["summary"]["total_parent_cases"] == 3
    models = [case["model_type"] for case in result["parent_cases"]]
    assert models.count("random_forest") == 2
    assert models.count("xgboost") == 1

    configs = [
        yaml.safe_load(Path(case["config_path"]).read_text(encoding="utf-8"))
        for case in result["valid_cases"]
    ]
    rf_params = [
        cfg["train"]["model"].get("params", {})
        for cfg in configs
        if cfg["train"]["model"]["type"] == "random_forest"
    ]
    xgb_params = [
        cfg["train"]["model"].get("params", {})
        for cfg in configs
        if cfg["train"]["model"]["type"] == "xgboost"
    ]
    assert rf_params == [{"n_estimators": 50}, {"n_estimators": 100}]
    assert xgb_params == [{}]


def test_generate_doe_model_search_optuna_samples_tabular_params(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_optuna(monkeypatch)
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["split.mode"] = ["cv"]
    spec["defaults"]["split.cv.n_splits"] = 2
    spec["defaults"]["split.cv.repeats"] = 1
    spec["search_space"]["train.model.type"] = ["random_forest"]
    spec["model_search"] = {
        "random_forest": {
            "method": "optuna",
            "n_trials": 3,
            "seed": 7,
            "params": {
                "n_estimators": {"type": "int", "low": 50, "high": 150, "step": 50},
                "max_depth": {"type": "categorical", "choices": [None, 4, 8]},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2"]},
            },
        }
    }

    result = generate_doe(spec)

    assert result["summary"]["total_parent_cases"] == 3
    assert result["summary"]["total_execution_cases"] == 6
    assert result["summary"]["valid_cases"] == 6
    assert {parent["execution_count"] for parent in result["parent_cases"]} == {2}
    parent_scientific_ids = {case["scientific_config_id"] for case in result["parent_cases"]}
    assert len(parent_scientific_ids) == 3
    for parent in result["parent_cases"]:
        child_scientific_ids = {
            case["scientific_config_id"]
            for case in result["valid_cases"]
            if case["parent_case_id"] == parent["case_id"]
        }
        assert child_scientific_ids == {parent["scientific_config_id"]}
    configs = [
        yaml.safe_load(Path(case["config_path"]).read_text(encoding="utf-8"))
        for case in result["valid_cases"]
    ]
    seen_params = {json.dumps(cfg["train"]["model"]["params"], sort_keys=True) for cfg in configs}
    assert len(seen_params) == 3
    for config in configs:
        params = config["train"]["model"]["params"]
        assert params["n_estimators"] in {50, 100, 150}
        assert params["max_depth"] in {None, 4, 8}
        assert params["max_features"] in {"sqrt", "log2"}
        assert config["train"]["tuning"]["method"] == "fixed"
        assert "use_hpo" not in config["train"]["tuning"]


def test_generate_doe_model_search_optuna_is_repeatable_byte_for_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_optuna(monkeypatch)
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["random_forest"]
    spec["model_search"] = {
        "random_forest": {
            "method": "optuna",
            "n_trials": 3,
            "seed": 7,
            "params": {
                "n_estimators": {"type": "int", "low": 50, "high": 150, "step": 50},
                "max_depth": {"type": "categorical", "choices": [None, 4, 8]},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2"]},
            },
        }
    }

    def _generated_bytes(result: dict) -> dict[str, bytes]:
        output_dir = Path(spec["output"]["dir"])
        paths = [
            Path(result["summary_path"]),
            Path(result["manifest_path"]),
            Path(result["parent_manifest_path"]),
            output_dir / "doe_spec.input.yaml",
        ]
        git_patch = result["summary"].get("git_diff_snapshot_path")
        if git_patch:
            paths.append(Path(git_patch))
        paths.extend(Path(case["config_path"]) for case in result["valid_cases"])
        return {
            str(path.relative_to(output_dir)): path.read_bytes()
            for path in sorted(paths)
        }

    first = _generated_bytes(generate_doe(spec))
    second = _generated_bytes(generate_doe(spec))

    assert second == first


def test_generate_doe_model_search_optuna_samples_xgboost_params(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_optuna(monkeypatch)
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["xgboost"]
    spec["model_search"] = {
        "xgboost": {
            "method": "optuna",
            "n_trials": 2,
            "seed": 11,
            "params": {
                "n_estimators": {"type": "int", "low": 100, "high": 300, "step": 100},
                "max_depth": {"type": "int", "low": 3, "high": 7},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
            },
        }
    }

    result = generate_doe(spec)

    assert result["summary"]["total_parent_cases"] == 2
    for case in result["valid_cases"]:
        config = yaml.safe_load(Path(case["config_path"]).read_text(encoding="utf-8"))
        params = config["train"]["model"]["params"]
        assert params["n_estimators"] in {100, 200, 300}
        assert 3 <= params["max_depth"] <= 7
        assert 0.01 <= params["learning_rate"] <= 0.2


def test_generate_doe_model_search_optuna_samples_dl_params_without_child_hpo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_optuna(monkeypatch)
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["dl_simple"]
    spec["model_search"] = {
        "dl_simple": {
            "method": "optuna",
            "n_trials": 2,
            "seed": 13,
            "params": {
                "hidden_dim": {"type": "categorical", "choices": [64, 128]},
                "dropout": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
                "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01, "log": True},
                "batch_size": {"type": "categorical", "choices": [32, 64]},
            },
        }
    }

    result = generate_doe(spec)

    assert result["summary"]["total_parent_cases"] == 2
    for case in result["valid_cases"]:
        config = yaml.safe_load(Path(case["config_path"]).read_text(encoding="utf-8"))
        params = config["train"]["model"]["params"]
        assert params["hidden_dim"] in {64, 128}
        assert 0.0 <= params["dropout"] <= 0.5
        assert 0.0001 <= params["learning_rate"] <= 0.01
        assert params["batch_size"] in {32, 64}
        assert config["train"]["tuning"] == {"method": "fixed"}


def test_generate_doe_model_search_optuna_uses_real_optuna_package() -> None:
    pytest.importorskip("optuna")

    samples = doe_module._expand_model_search_for_model(
        "random_forest",
        {
            "method": "optuna",
            "n_trials": 2,
            "seed": 42,
            "params": {
                "n_estimators": {"type": "int", "low": 10, "high": 30, "step": 10},
                "max_depth": {"type": "categorical", "choices": [None, 4, 8]},
            },
        },
    )

    assert len(samples) == 2
    assert len({json.dumps(sample, sort_keys=True) for sample in samples}) == 2
    for sample in samples:
        assert sample["train.model.params.n_estimators"] in {10, 20, 30}
        assert sample["train.model.params.max_depth"] in {None, 4, 8}


@pytest.mark.parametrize(
    ("default_key", "default_value"),
    [
        ("train.tuning.method", "train_cv"),
        ("train.tuning.method", "optuna"),
        ("train.tuning.use_hpo", True),
    ],
)
def test_generate_doe_rejects_child_level_tuning_defaults(
    tmp_path: Path,
    default_key: str,
    default_value: object,
) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["defaults"][default_key] = default_value

    with pytest.raises(DOEGenerationError, match="Runtime child-level tuning setting"):
        generate_doe(spec)


@pytest.mark.parametrize(
    "axis",
    [
        "train.tuning.method",
        "train.tuning.hpo_trials",
        "train_tdc.tuning.use_hpo",
    ],
)
def test_generate_doe_rejects_child_level_tuning_search_axes(
    tmp_path: Path,
    axis: str,
) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"][axis] = ["fixed"]

    with pytest.raises(DOEGenerationError, match="Runtime child-level tuning axes"):
        generate_doe(spec)


def test_generate_doe_model_search_invalid_specs_fail_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_optuna(monkeypatch)
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["random_forest"]
    spec["model_search"] = {
        "random_forest": {
            "method": "optuna",
            "n_trials": 1,
            "params": {
                "n_estimators": {"type": "int", "low": 10},
            },
        }
    }

    with pytest.raises(DOEGenerationError, match="missing required 'high'"):
        generate_doe(spec)


@pytest.mark.parametrize(
    ("search_cfg", "match"),
    [
        (
            {
                "method": "optuna",
                "n_trials": "abc",
                "params": {"n_estimators": {"type": "int", "low": 10, "high": 20}},
            },
            r"model_search\.random_forest\.n_trials must be an integer",
        ),
        (
            {
                "method": "optuna",
                "n_trials": 1,
                "params": {"n_estimators": {"type": "int", "low": "abc", "high": 20}},
            },
            r"model_search\.random_forest\.params\.n_estimators\.low must be an integer",
        ),
        (
            {
                "method": "optuna",
                "n_trials": 1,
                "params": {
                    "learning_rate": {
                        "type": "float",
                        "low": 0.01,
                        "high": 0.1,
                        "step": "abc",
                    }
                },
            },
            r"model_search\.random_forest\.params\.learning_rate\.step must be a number",
        ),
    ],
)
def test_generate_doe_model_search_invalid_numeric_specs_fail_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    search_cfg: dict,
    match: str,
) -> None:
    _patch_fake_optuna(monkeypatch)
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["random_forest"]
    spec["model_search"] = {"random_forest": search_cfg}

    with pytest.raises(DOEGenerationError, match=match):
        generate_doe(spec)


def test_generate_doe_model_search_rejects_random_method_name(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["random_forest"]
    spec["model_search"] = {
        "random_forest": {
            "method": "random",
            "n_trials": 1,
            "params": {"n_estimators": {"type": "int", "low": 10, "high": 20}},
        }
    }

    with pytest.raises(DOEGenerationError, match="method='random' is unsupported"):
        generate_doe(spec)


def test_generate_doe_model_search_optuna_requires_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "optuna":
            raise ImportError("blocked optuna import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["random_forest"]
    spec["model_search"] = {
        "random_forest": {
            "method": "optuna",
            "n_trials": 1,
            "params": {"n_estimators": {"type": "int", "low": 10, "high": 20}},
        }
    }

    with pytest.raises(DOEGenerationError, match="requires the optuna package"):
        generate_doe(spec)


def test_generate_doe_model_search_rejects_unmatched_entries(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["random_forest"]
    spec["model_search"] = {
        "randomforest": {
            "method": "grid",
            "params": {"n_estimators": [50, 100]},
        }
    }

    with pytest.raises(DOEGenerationError, match="did not match.*randomforest"):
        generate_doe(spec)


def test_generate_doe_model_search_applies_to_profile_default_model(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"].pop("train.model.type")
    spec["model_search"] = {
        "random_forest": {
            "method": "grid",
            "params": {"n_estimators": [50, 100]},
        }
    }

    result = generate_doe(spec)

    assert result["summary"]["total_parent_cases"] == 2
    configs = [
        yaml.safe_load(Path(case["config_path"]).read_text(encoding="utf-8"))
        for case in result["valid_cases"]
    ]
    assert [cfg["train"]["model"]["params"] for cfg in configs] == [
        {"n_estimators": 50},
        {"n_estimators": 100},
    ]


def test_generate_doe_model_search_rejects_tdc_profile(tmp_path: Path) -> None:
    spec = {
        "version": 1,
        "dataset": {
            "profile": "clf_tdc_benchmark",
            "name": "pgp_tdc_benchmark",
            "task_type": "classification",
            "source": {
                "type": "tdc",
                "group": "ADMET_Group",
                "name": "Pgp_Broccatelli",
            },
        },
        "search_space": {"train_tdc.model.type": ["catboost_classifier"]},
        "defaults": {"global.base_dir": str(tmp_path / "tdc_data")},
        "output": {"dir": str(tmp_path / "tdc_generated")},
        "model_search": {
            "catboost_classifier": {
                "method": "grid",
                "params": {"depth": [4, 6]},
            }
        },
    }

    with pytest.raises(
        DOEGenerationError,
        match="train_tdc\\.model\\.params search is not supported",
    ):
        generate_doe(spec)


def test_generate_doe_model_search_respects_max_cases_after_expansion(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["random_forest"]
    spec["constraints"] = {"max_cases": 1}
    spec["model_search"] = {
        "random_forest": {
            "method": "grid",
            "params": {"n_estimators": [50, 100]},
        }
    }

    with pytest.raises(DOEGenerationError, match="after model_search expansion"):
        generate_doe(spec)


def test_generate_doe_model_search_rejects_search_space_param_conflict(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["random_forest"]
    spec["search_space"]["train.model.params.n_estimators"] = [50]
    spec["model_search"] = {
        "random_forest": {
            "method": "grid",
            "params": {"n_estimators": [100]},
        }
    }

    with pytest.raises(DOEGenerationError, match="must not redefine axes"):
        generate_doe(spec)


@pytest.mark.parametrize(
    ("search_space_key", "model_search_params"),
    [
        ("train.model.params.optimizer.lr", {"optimizer": ["adam"]}),
        ("train.model.params.optimizer", {"optimizer.lr": [0.001]}),
    ],
)
def test_generate_doe_model_search_rejects_nested_param_conflicts(
    tmp_path: Path,
    search_space_key: str,
    model_search_params: dict,
) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["random_forest"]
    spec["search_space"][search_space_key] = [0.01]
    spec["model_search"] = {
        "random_forest": {
            "method": "grid",
            "params": model_search_params,
        }
    }

    with pytest.raises(DOEGenerationError, match="must not redefine axes"):
        generate_doe(spec)


def test_generate_doe_model_search_rejects_duplicate_grid_values(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["random_forest"]
    spec["model_search"] = {
        "random_forest": {
            "method": "grid",
            "params": {"n_estimators": [50, 50]},
        }
    }

    with pytest.raises(DOEGenerationError, match="duplicate grid values"):
        generate_doe(spec)


def test_generate_doe_respects_explicit_cv_execution_axes_in_defaults(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["split.mode"] = ["cv"]
    spec["defaults"]["split.cv.n_splits"] = 3
    spec["defaults"]["split.cv.repeats"] = 2
    spec["defaults"]["split.cv.fold_index"] = 1
    spec["defaults"]["split.cv.repeat_index"] = 0

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["total_cases"] == 1
    assert summary["valid_cases"] == 1

    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["split"]["cv"]["fold_index"] == 1
    assert config["split"]["cv"]["repeat_index"] == 0
    assert result["valid_cases"][0]["parent_case_id"] == "parent_0001"
    assert summary["total_parent_cases"] == 1
    assert summary["total_execution_cases"] == 1


def test_generate_doe_validates_cv_fold_index_bounds(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["split.mode"] = ["cv"]
    spec["search_space"]["split.strategy"] = ["random"]
    spec["defaults"]["split.cv.n_splits"] = 3
    spec["defaults"]["split.cv.repeats"] = 1
    spec["defaults"]["split.cv.fold_index"] = 3
    spec["defaults"]["split.cv.repeat_index"] = 0

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_SPLIT_PARAM_INVALID", 0) >= 1
    manifest_lines = Path(result["manifest_path"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(manifest_lines) == 1
    payload = json.loads(manifest_lines[0])
    assert payload["status"] == "skipped"
    assert payload["execution_label"] == "rep0_fold3"
    parent_manifest_lines = Path(result["parent_manifest_path"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(parent_manifest_lines) == 1
    parent_payload = json.loads(parent_manifest_lines[0])
    assert parent_payload["execution_case_ids"] == ["case_0001"]
    assert parent_payload["execution_labels"] == ["rep0_fold3"]


def test_generate_doe_supports_auto_task_with_confirmation(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated_auto"
    spec = {
        "version": 1,
        "dataset": {
            "name": "pgp_auto",
            "task_type": "auto",
            "auto_confirmed": True,
            "target_column": "Activity",
            "source": {"type": "local_csv", "path": _pgp_dataset_path()},
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.none"],
            "pipeline.preprocess": [False],
            "pipeline.select": [False],
            "pipeline.explain": [False],
            "pipeline.label_normalize": [False],
            "train.model.type": ["catboost_classifier"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "data_auto"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
            "split.stratify": True,
        },
        "output": {"dir": str(output_dir)},
    }

    result = generate_doe(spec)
    summary = result["summary"]
    assert summary["task_type"] == "classification"
    assert summary["profile"] == "clf_local_csv"
    assert summary["valid_cases"] == 1

    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["global"]["task_type"] == "classification"
    assert config["train"]["model"]["type"] == "catboost_classifier"

    manifest_lines = Path(result["manifest_path"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(manifest_lines) == 1
    payload = json.loads(manifest_lines[0])
    assert payload["record_type"] == "execution_child"
    assert payload["parent_case_id"] == "parent_0001"
    assert payload["execution_label"] == "holdout"
    assert payload["status"] == "valid"

    parent_manifest_lines = Path(result["parent_manifest_path"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(parent_manifest_lines) == 1
    parent_payload = json.loads(parent_manifest_lines[0])
    assert parent_payload["record_type"] == "parent"
    assert parent_payload["case_id"] == "parent_0001"
    assert parent_payload["execution_count"] == 1


def test_generate_doe_isolates_case_artifacts_by_default(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["catboost_classifier", "dl_simple"]
    spec["defaults"]["global.base_dir"] = str(tmp_path / "data_root")
    spec["defaults"]["global.run_dir"] = str(tmp_path / "runs_root")

    result = generate_doe(spec)
    assert result["summary"]["valid_cases"] == 2

    config_paths = [Path(case["config_path"]) for case in result["valid_cases"]]
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in config_paths]

    base_dirs = [cfg["global"]["base_dir"] for cfg in configs]
    run_dirs = [cfg["global"]["run_dir"] for cfg in configs]
    run_ids = [cfg["global"]["runs"]["id"] for cfg in configs]
    namespace = result["summary"]["doe_spec_hash"][:8]

    assert len(set(base_dirs)) == 2
    assert len(set(run_dirs)) == 2
    assert len(set(run_ids)) == 2
    assert any(path.endswith("case_0001") for path in base_dirs)
    assert any(path.endswith("case_0002") for path in base_dirs)
    assert all(Path(path).parts[-2] == namespace for path in base_dirs)
    assert all(Path(path).parts[-2] == namespace for path in run_dirs)


def test_generate_doe_propagates_global_artifact_retention(tmp_path: Path) -> None:
    profile = doe_module.PROFILE_SPECS["clf_local_csv"]
    dataset_cfg = _base_clf_doe(tmp_path)["dataset"]
    merged = {
        "global.base_dir": str(tmp_path / "data"),
        "global.random_state": 42,
        "global.runs.enabled": False,
        "global.artifact_retention": "audit_light",
        "split.mode": "holdout",
        "split.strategy": "random",
        "split.test_size": 0.2,
        "split.val_size": 0.1,
        "split.stratify": True,
        "split.require_disjoint": True,
        "split.require_full_test_coverage": True,
        "pipeline.feature_input": "featurize.morgan",
        "pipeline.preprocess": False,
        "pipeline.select": False,
        "pipeline.explain": True,
        "train.model.type": "catboost_classifier",
    }
    config = doe_module._build_case_config(
        profile=profile,
        dataset_cfg=dataset_cfg,
        merged=merged,
        resolved_task="classification",
        probe={"resolved_smiles_column": "SMILES"},
    )
    assert config["global"]["artifact_retention"] == "audit_light"


def test_hashable_payload_ignores_global_artifact_retention() -> None:
    base_config = {
        "global": {
            "base_dir": "/tmp/data",
            "run_dir": "/tmp/run",
            "artifact_retention": "full",
            "runs": {"enabled": True, "id": "case_001"},
        },
        "pipeline": {"nodes": ["get_data", "train"]},
        "train": {"model": {"type": "random_forest"}},
    }
    audit_config = json.loads(json.dumps(base_config))
    audit_config["global"]["artifact_retention"] = "audit_light"

    base_hash = doe_module._stable_hash(doe_module._hashable_config_payload(base_config))
    audit_hash = doe_module._stable_hash(doe_module._hashable_config_payload(audit_config))
    base_science_hash = doe_module._stable_hash(doe_module._scientific_config_payload(base_config))
    audit_science_hash = doe_module._stable_hash(doe_module._scientific_config_payload(audit_config))

    assert base_hash == audit_hash
    assert base_science_hash == audit_science_hash


def test_generate_doe_isolation_namespaces_paths_by_spec_hash(tmp_path: Path) -> None:
    spec_a = _base_clf_doe(tmp_path)
    spec_a["search_space"]["train.model.type"] = ["catboost_classifier"]
    spec_a["defaults"]["global.base_dir"] = str(tmp_path / "shared_data")
    spec_a["defaults"]["global.run_dir"] = str(tmp_path / "shared_runs")
    spec_a["output"]["dir"] = str(tmp_path / "generated_a")

    spec_b = _base_clf_doe(tmp_path)
    spec_b["search_space"]["train.model.type"] = ["dl_simple"]
    spec_b["defaults"]["global.base_dir"] = str(tmp_path / "shared_data")
    spec_b["defaults"]["global.run_dir"] = str(tmp_path / "shared_runs")
    spec_b["output"]["dir"] = str(tmp_path / "generated_b")

    result_a = generate_doe(spec_a)
    result_b = generate_doe(spec_b)

    cfg_a = yaml.safe_load(Path(result_a["valid_cases"][0]["config_path"]).read_text(encoding="utf-8"))
    cfg_b = yaml.safe_load(Path(result_b["valid_cases"][0]["config_path"]).read_text(encoding="utf-8"))

    ns_a = result_a["summary"]["doe_spec_hash"][:8]
    ns_b = result_b["summary"]["doe_spec_hash"][:8]

    assert ns_a != ns_b
    assert Path(cfg_a["global"]["base_dir"]).parts[-2] == ns_a
    assert Path(cfg_b["global"]["base_dir"]).parts[-2] == ns_b
    assert cfg_a["global"]["base_dir"] != cfg_b["global"]["base_dir"]
    assert cfg_a["global"]["run_dir"] != cfg_b["global"]["run_dir"]


def test_generate_doe_validates_dataset_columns(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["dataset"]["smiles_column"] = "DOES_NOT_EXIST"
    spec["dataset"]["label_source_column"] = "MISSING_LABEL_SOURCE"
    spec["search_space"]["pipeline.label_normalize"] = [True]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_DATASET_COLUMN_MISSING", 0) >= 1


def test_generate_doe_defaults_feature_input_for_non_chemprop_classification(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"].pop("pipeline.feature_input")

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "featurize.morgan" in config["pipeline"]["nodes"]
    assert summary["issue_counts"].get("DOE_FEATURE_INPUT_REQUIRED", 0) == 0


def test_generate_doe_requires_regression_target_column_for_local_csv(tmp_path: Path) -> None:
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv",
            "task_type": "regression",
            "source": {"type": "local_csv", "path": _flash_dataset_path()},
            "smiles_column": "SMILES",
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.rdkit"],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "reg_missing_target"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
        },
        "output": {"dir": str(tmp_path / "generated_reg_missing_target")},
    }

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_TARGET_COLUMN_MISSING", 0) == 1


def test_generate_doe_reg_chembl_defaults_target_column_to_pic50(tmp_path: Path) -> None:
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_chembl_ic50",
            "task_type": "regression",
            "source": {"type": "chembl", "target_name": "Urease"},
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "chembl_default_target"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
        },
        "output": {"dir": str(tmp_path / "generated_chembl_default_target")},
    }

    result = generate_doe(spec)
    assert result["summary"]["valid_cases"] == 1

    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["global"]["target_column"] == "pIC50"


def test_generate_doe_reg_local_csv_ic50_inserts_label_step(tmp_path: Path) -> None:
    raw_csv = tmp_path / "chembl_raw_local.csv"
    raw_csv.write_text(
        "\n".join(
            [
                "canonical_smiles,standard_value,standard_units,standard_relation,standard_type,target_chembl_id",
                "CCO,1000,nM,=,IC50,CHEMBL3885651",
                "CCN,2500,nM,=,IC50,CHEMBL3885651",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv_ic50",
            "task_type": "regression",
            "target_column": "pIC50",
            "source": {"type": "local_csv", "path": str(raw_csv)},
            "smiles_column": "canonical_smiles",
            "curate": {
                "required_non_null_columns": ["standard_value", "standard_units", "standard_relation"],
                "row_filters": {
                    "standard_type": ["IC50"],
                    "standard_units": ["nM"],
                    "standard_relation": ["="],
                },
            },
        },
        "search_space": {
            "split.mode": ["cv"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.rdkit"],
            "pipeline.preprocess": [False],
            "pipeline.select": [False],
            "pipeline.explain": [False],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "chembl_raw_ic50"),
            "global.runs.enabled": False,
            "split.cv.n_splits": 2,
            "split.cv.repeats": 1,
            "split.val_from_train.val_size": 0.1,
        },
        "output": {"dir": str(tmp_path / "generated_chembl_raw_ic50")},
    }

    result = generate_doe(spec)
    assert result["summary"]["valid_cases"] == 2

    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["global"]["target_column"] == "pIC50"
    assert config["curate"]["properties"] == "standard_value"
    assert config["curate"]["keep_all_columns"] is True
    assert config["curate"]["row_filters"] == {
        "standard_type": ["IC50"],
        "standard_units": ["nM"],
        "standard_relation": ["="],
    }
    assert config["pipeline"]["nodes"] == [
        "get_data",
        "curate",
        "label.ic50",
        "featurize.rdkit",
        "split",
        "train",
    ]


def test_generate_doe_reg_local_csv_ic50_requires_standard_value_column(tmp_path: Path) -> None:
    raw_csv = tmp_path / "chembl_raw_local_missing_value.csv"
    raw_csv.write_text(
        "\n".join(
            [
                "canonical_smiles",
                "CCO",
                "CCN",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv_ic50",
            "task_type": "regression",
            "target_column": "pIC50",
            "source": {"type": "local_csv", "path": str(raw_csv)},
            "smiles_column": "canonical_smiles",
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.rdkit"],
            "pipeline.preprocess": [False],
            "pipeline.select": [False],
            "pipeline.explain": [False],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "chembl_raw_ic50_missing_value"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
        },
        "output": {"dir": str(tmp_path / "generated_chembl_raw_ic50_missing_value")},
    }

    result = generate_doe(spec)
    assert result["summary"]["valid_cases"] == 0
    assert result["summary"]["issue_counts"]["DOE_LABEL_IC50_SOURCE_COLUMNS_MISSING"] == 1


def test_generate_doe_reg_local_csv_ic50_rejects_curate_dropping_standard_value(tmp_path: Path) -> None:
    raw_csv = tmp_path / "chembl_raw_local_dropped_value.csv"
    raw_csv.write_text(
        "\n".join(
            [
                "canonical_smiles,standard_value,other_col",
                "CCO,1000,1",
                "CCN,2500,2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv_ic50",
            "task_type": "regression",
            "target_column": "pIC50",
            "source": {"type": "local_csv", "path": str(raw_csv)},
            "smiles_column": "canonical_smiles",
            "curate": {
                "properties": ["other_col"],
                "keep_all_columns": False,
            },
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.rdkit"],
            "pipeline.preprocess": [False],
            "pipeline.select": [False],
            "pipeline.explain": [False],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "chembl_raw_ic50_dropped_value"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
        },
        "output": {"dir": str(tmp_path / "generated_chembl_raw_ic50_dropped_value")},
    }

    result = generate_doe(spec)
    assert result["summary"]["valid_cases"] == 0
    assert result["summary"]["issue_counts"]["DOE_CURATE_LABEL_IC50_SOURCE_DROPPED"] == 1


def test_generate_doe_records_dirty_git_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv",
            "name": "flash_reg_doe",
            "task_type": "regression",
            "target_column": "FP Exp.",
            "source": {"type": "local_csv", "path": _flash_dataset_path()},
            "smiles_column": "SMILES",
            "curate": {
                "properties": "FP Exp.",
                "smiles_column": "SMILES",
                "dedupe_strategy": "first",
                "keep_all_columns": True,
            },
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.none"],
            "pipeline.preprocess": [False],
            "pipeline.select": [False],
            "pipeline.explain": [False],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "data"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
        },
        "output": {"dir": str(tmp_path / "generated")},
    }

    git_outputs = {
        ("git", "rev-parse", "--short", "HEAD"): "abc123\n",
        ("git", "status", "--short"): " M utilities/doe.py\n?? tmp/example.yaml\n",
        ("git", "diff", "--no-ext-diff", "--binary", "HEAD"): "diff --git a/utilities/doe.py b/utilities/doe.py\n",
    }

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        key = tuple(cmd)
        if key not in git_outputs:
            raise AssertionError(f"Unexpected git command: {cmd!r}")
        return subprocess.CompletedProcess(cmd, 0, stdout=git_outputs[key], stderr="")

    monkeypatch.setattr(doe_module.subprocess, "run", fake_run)

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["git_sha"] == "abc123"
    assert summary["git_dirty"] is True
    assert summary["git_status_short"] == [" M utilities/doe.py", "?? tmp/example.yaml"]
    diff_path = Path(summary["git_diff_snapshot_path"])
    assert diff_path.exists()
    diff_text = diff_path.read_text(encoding="utf-8")
    assert "# git status --short" in diff_text
    assert "# git diff --no-ext-diff --binary HEAD" in diff_text
    assert summary["git_diff_hash"]


def test_generate_doe_reg_chembl_propagates_target_pin_and_row_filters(tmp_path: Path) -> None:
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_chembl_ic50",
            "task_type": "regression",
            "source": {"type": "chembl", "target_chembl_id": "CHEMBL3885651"},
            "curate": {
                "row_filters": {
                    "target_chembl_id": ["CHEMBL3885651"],
                    "standard_type": ["IC50"],
                    "standard_units": ["nM"],
                    "standard_relation": ["="],
                }
            },
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["scaffold"],
            "pipeline.feature_input": ["featurize.rdkit"],
            "pipeline.preprocess": [False],
            "pipeline.select": [False],
            "pipeline.explain": [False],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "chembl_row_filters"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
        },
        "output": {"dir": str(tmp_path / "generated_chembl_row_filters")},
    }

    result = generate_doe(spec)
    assert result["summary"]["valid_cases"] == 1

    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    nodes = config["pipeline"]["nodes"]

    assert config["get_data"]["source"]["target_chembl_id"] == "CHEMBL3885651"
    assert config["curate"]["row_filters"] == {
        "target_chembl_id": ["CHEMBL3885651"],
        "standard_type": ["IC50"],
        "standard_units": ["nM"],
        "standard_relation": ["="],
    }
    assert "featurize.lipinski" not in nodes
    assert nodes.index("label.ic50") < nodes.index("featurize.rdkit")
    assert nodes.index("featurize.rdkit") < nodes.index("split")


def test_generate_doe_validates_invalid_dedupe_strategy(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["defaults"]["curate.dedupe_strategy"] = "keepfirst_typo"

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_CURATE_DEDUPE_INVALID", 0) == 1


def test_generate_doe_propagates_curate_drop_controls(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["defaults"]["curate.drop_missing_smiles"] = False
    spec["defaults"]["curate.drop_invalid_smiles"] = False
    spec["defaults"]["curate.drop_missing_target"] = False
    spec["defaults"]["curate.required_non_null_columns"] = "SMILES,Activity"

    result = generate_doe(spec)
    assert result["summary"]["valid_cases"] == 1

    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    curate_cfg = config["curate"]
    assert curate_cfg["drop_missing_smiles"] is False
    assert curate_cfg["drop_invalid_smiles"] is False
    assert curate_cfg["drop_missing_target"] is False
    assert curate_cfg["required_non_null_columns"] == ["SMILES", "Activity"]


def test_generate_doe_rejects_missing_required_non_null_columns(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["defaults"]["curate.required_non_null_columns"] = ["does_not_exist"]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_CURATE_REQUIRED_COLUMNS_MISSING", 0) == 1


def test_generate_doe_allows_required_non_null_canonical_smiles_alias(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["defaults"]["curate.required_non_null_columns"] = ["canonical_smiles", "Activity"]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    assert summary["issue_counts"].get("DOE_CURATE_REQUIRED_COLUMNS_MISSING", 0) == 0


def test_generate_doe_writes_spec_snapshot_and_hash(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    result = generate_doe(spec)
    summary = result["summary"]

    assert summary.get("doe_spec_hash")
    snapshot_path = Path(summary["doe_spec_snapshot_path"])
    assert snapshot_path.exists()
    assert snapshot_path.name == "doe_spec.input.yaml"


def test_generate_doe_auto_task_detects_float_binary_labels(tmp_path: Path) -> None:
    csv_path = tmp_path / "float_binary.csv"
    csv_path.write_text(
        "SMILES,label\nCC,0.0\nCCC,1.0\nCCCC,0.0\n",
        encoding="utf-8",
    )
    spec = {
        "version": 1,
        "dataset": {
            "name": "float_binary_auto",
            "task_type": "auto",
            "auto_confirmed": True,
            "target_column": "label",
            "source": {"type": "local_csv", "path": str(csv_path)},
            "smiles_column": "SMILES",
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.none"],
            "train.model.type": ["catboost_classifier"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "float_binary_data"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
            "split.stratify": True,
        },
        "output": {"dir": str(tmp_path / "generated_float_binary")},
    }

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["task_type"] == "classification"
    assert summary["valid_cases"] == 1


def test_generate_doe_auto_resolves_smiles_column_for_local_csv(tmp_path: Path) -> None:
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv",
            "task_type": "regression",
            "target_column": "FP Exp.",
            "source": {"type": "local_csv", "path": _flash_dataset_path()},
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.rdkit"],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "auto_smiles_data"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
        },
        "output": {"dir": str(tmp_path / "generated_auto_smiles")},
    }

    result = generate_doe(spec)
    assert result["summary"]["valid_cases"] == 1

    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["curate"]["smiles_column"] == "SMILES"


def test_generate_doe_rejects_unresolvable_smiles_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "no_smiles.csv"
    csv_path.write_text("structure,target\nCC,0.1\nCCC,0.2\n", encoding="utf-8")
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv",
            "task_type": "regression",
            "target_column": "target",
            "source": {"type": "local_csv", "path": str(csv_path)},
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.rdkit"],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "no_smiles_data"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
        },
        "output": {"dir": str(tmp_path / "generated_no_smiles")},
    }

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_SMILES_COLUMN_MISSING", 0) == 1


def test_generate_doe_rejects_regression_when_curate_drops_target(tmp_path: Path) -> None:
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv",
            "task_type": "regression",
            "target_column": "FP Exp.",
            "source": {"type": "local_csv", "path": _flash_dataset_path()},
            "smiles_column": "SMILES",
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.rdkit"],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "missing_target_data"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
            "curate.properties": ["FP Calc."],
        },
        "output": {"dir": str(tmp_path / "generated_missing_target")},
    }

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_CURATE_TARGET_DROPPED", 0) == 1


def test_generate_doe_allows_regression_target_when_keep_all_columns(tmp_path: Path) -> None:
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv",
            "task_type": "regression",
            "target_column": "FP Exp.",
            "source": {"type": "local_csv", "path": _flash_dataset_path()},
            "smiles_column": "SMILES",
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.rdkit"],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "keep_all_data"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
            "curate.properties": ["FP Calc."],
            "curate.keep_all_columns": True,
        },
        "output": {"dir": str(tmp_path / "generated_keep_all")},
    }

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    assert summary["issue_counts"].get("DOE_CURATE_TARGET_DROPPED", 0) == 0


def test_generate_doe_featurize_none_defaults_keep_all_columns(tmp_path: Path) -> None:
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv",
            "task_type": "regression",
            "target_column": "FP Exp.",
            "source": {"type": "local_csv", "path": _flash_dataset_path()},
            "smiles_column": "SMILES",
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.none"],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "default_keep_all_data"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
            "curate.properties": ["FP Calc."],
        },
        "output": {"dir": str(tmp_path / "generated_default_keep_all")},
    }

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    assert summary["issue_counts"].get("DOE_CURATE_TARGET_DROPPED", 0) == 0

    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert (config.get("curate") or {}).get("keep_all_columns") is True


def test_generate_doe_accepts_rdkit_labeled_feature_alias(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["featurize.rdkit_labeled"]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    assert summary["issue_counts"].get("DOE_FEATURE_INPUT_NOT_SUPPORTED", 0) == 0
    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "featurize.rdkit_labeled" in config["pipeline"]["nodes"]


def test_generate_doe_accepts_ecfp4_rdkit_feature_input(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["featurize.ecfp4_rdkit"]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    assert summary["issue_counts"].get("DOE_FEATURE_INPUT_NOT_SUPPORTED", 0) == 0
    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["pipeline"]["feature_input"] == "featurize.ecfp4_rdkit"
    assert "featurize.ecfp4_rdkit" in config["pipeline"]["nodes"]
    assert config["featurize"]["radius"] == 2
    assert config["featurize"]["n_bits"] == 2048


def test_generate_doe_normalizes_legacy_curated_feature_alias(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["use.curated_features"]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "featurize.none" in config["pipeline"]["nodes"]
    assert "use.curated_features" not in config["pipeline"]["nodes"]


def test_generate_doe_canonicalizes_parent_identity_for_feature_alias_duplicates(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["featurize.none", "use.curated_features"]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["total_cases"] == 1
    assert summary["valid_cases"] == 1
    assert summary["total_parent_cases"] == 1
    assert len(result["parent_cases"]) == 1
    parent_manifest_lines = Path(result["parent_manifest_path"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(parent_manifest_lines) == 1
    parent_payload = json.loads(parent_manifest_lines[0])
    assert parent_payload["execution_case_ids"] == ["case_0001"]
    assert parent_payload["valid_execution_case_ids"] == ["case_0001"]


def test_generate_doe_requires_max_cases_for_large_grid(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"] = {
        "axis.a": list(range(101)),
        "axis.b": list(range(101)),
    }

    with pytest.raises(DOEGenerationError, match="constraints.max_cases"):
        generate_doe(spec)


# ---------------------------------------------------------------------------
# ts_forecast accepts both local_npy and local_ts_csv (allowed_sources)
# ---------------------------------------------------------------------------


def test_ts_forecast_profile_allows_both_sources() -> None:
    from utilities.doe import PROFILE_SPECS
    ts = PROFILE_SPECS["ts_forecast"]
    assert ts.allows_source("local_npy")
    assert ts.allows_source("local_ts_csv")
    assert not ts.allows_source("local_csv")
    assert set(ts.source_choices()) == {"local_npy", "local_ts_csv"}


def test_validate_case_accepts_local_ts_csv_for_ts_forecast() -> None:
    from utilities.doe import PROFILE_SPECS, _validate_case
    ts = PROFILE_SPECS["ts_forecast"]
    base = {
        "pipeline": {"nodes": ["get_data", "train.timeseries"]},
        "train": {"model": {"type": "dl_adaptive_nvar"}},
        "split": {"warmup_len": 100, "train_len": 500, "val_len": 100, "test_len": 100},
    }
    for src in ("local_npy", "local_ts_csv"):
        cfg = {**base, "get_data": {"data_source": src, "source": {"path": "data/x"}}}
        codes = [i.code for i in _validate_case(ts, cfg, {})]
        assert "DOE_SOURCE_NOT_SUPPORTED_FOR_PROFILE" not in codes, (src, codes)
    # A truly unsupported source is still rejected.
    cfg = {**base, "get_data": {"data_source": "local_csv", "source": {"path": "data/x.csv"}}}
    codes = [i.code for i in _validate_case(ts, cfg, {})]
    assert "DOE_SOURCE_NOT_SUPPORTED_FOR_PROFILE" in codes


def test_other_profiles_unaffected_by_allowed_sources_default() -> None:
    """Profiles without explicit allowed_sources still accept only their default."""
    from utilities.doe import PROFILE_SPECS
    reg = PROFILE_SPECS["reg_local_csv"]
    assert reg.source_choices() == ("local_csv",)
    assert reg.allows_source("local_csv")
    assert not reg.allows_source("local_npy")
