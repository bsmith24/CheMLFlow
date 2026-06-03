from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str
    hint: str | None = None


class ConfigValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        details = []
        for issue in issues:
            hint = f" Hint: {issue.hint}" if issue.hint else ""
            details.append(f"- [{issue.code}] {issue.path}: {issue.message}{hint}")
        msg = "Config validation failed:\n" + "\n".join(details)
        super().__init__(msg)


_NODE_TO_BLOCK = {
    "get_data": "get_data",
    "curate": "curate",
    "label.normalize": "label",
    "split": "split",
    "featurize.none": "featurize",
    "featurize.rdkit": "featurize",
    "featurize.rdkit_labeled": "featurize",
    "featurize.morgan": "featurize",
    "featurize.ecfp4_rdkit": "featurize",
    "preprocess.features": "preprocess",
    "select.features": "preprocess",
    "train": "train",
    "train.tdc": "train_tdc",
    "train.timeseries": "train",
    "analyze.stats": "analyze",
    "analyze.eda": "analyze",
}

_CONFIGLESS_NODE_TO_BLOCK = {
    "featurize.none": "featurize",
}

_ALWAYS_ALLOWED_BLOCKS = {"global", "pipeline"}
_KNOWN_TOP_LEVEL_BLOCKS = {
    *_ALWAYS_ALLOWED_BLOCKS,
    "get_data",
    "curate",
    "label",
    "split",
    "featurize",
    "preprocess",
    "train",
    "train_tdc",
    "analyze",
}
_FEATURE_INPUT_NODES = {
    "featurize.none",
    "featurize.rdkit",
    "featurize.rdkit_labeled",
    "featurize.morgan",
    "featurize.ecfp4_rdkit",
    "use.curated_features",
}
_CHEMPROP_LIKE_MODELS = {"chemprop", "chemeleon"}

# Time-series (Adaptive NVAR) pipeline. These data sources and model types
# belong exclusively to the train.timeseries node; pairing them with the
# tabular `train` node is a configuration error caught in strict validation.
_TIMESERIES_DATA_SOURCES = {"local_npy", "local_ts_csv"}
_TIMESERIES_MODELS = {"dl_adaptive_nvar", "dl_connectome_nvar"}
_FEATURE_INPUT_ALIASES = {
    "use.curated_features": "featurize.none",
}
_CLASSIFICATION_ONLY_MODELS = {"catboost_classifier"}
_DL_WILDCARD = "dl_*"
_ARTIFACT_RETENTION_VALUES = {"full", "audit_light"}
_GET_DATA_SAMPLE_STRATEGIES = {"random", "stratified"}
_GET_DATA_SAMPLE_SOURCE_TYPES = {"local_csv", "chembl", "http_csv"}
_GET_DATA_SAMPLE_KEYS = {"fraction", "seed", "strategy", "stratify_column"}
_RUNTIME_PROFILE_CONTRACTS: dict[str, dict[str, Any]] = {
    "reg_local_csv": {
        "allowed_feature_inputs": (
            "none",
            "smiles_native",
            "featurize.none",
            "featurize.rdkit",
            "featurize.morgan",
            "featurize.ecfp4_rdkit",
        ),
        "allowed_models": ("random_forest", "svm", "decision_tree", "xgboost", "ensemble", "chemprop", "chemeleon", _DL_WILDCARD),
    },
    "reg_local_csv_ic50": {
        "allowed_feature_inputs": (
            "none",
            "smiles_native",
            "featurize.none",
            "featurize.rdkit",
            "featurize.morgan",
            "featurize.ecfp4_rdkit",
        ),
        "allowed_models": ("random_forest", "svm", "decision_tree", "xgboost", "ensemble", "chemprop", "chemeleon", _DL_WILDCARD),
    },
    "reg_chembl_ic50": {
        "allowed_feature_inputs": ("featurize.rdkit",),
        "allowed_models": ("random_forest", "svm", "decision_tree", "xgboost", "ensemble", _DL_WILDCARD),
    },
    "clf_local_csv": {
        "allowed_feature_inputs": (
            "none",
            "smiles_native",
            "featurize.none",
            "featurize.rdkit",
            "featurize.morgan",
            "featurize.ecfp4_rdkit",
        ),
        "allowed_models": (
            "random_forest",
            "decision_tree",
            "xgboost",
            "svm",
            "ensemble",
            "catboost_classifier",
            "chemprop",
            "chemeleon",
            _DL_WILDCARD,
        ),
    },
    "clf_tdc_benchmark": {
        "allowed_feature_inputs": ("none",),
        "allowed_models": ("catboost_classifier",),
    },
}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _normalize_feature_input(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return _FEATURE_INPUT_ALIASES.get(normalized, normalized)


def _feature_input_from_nodes(nodes: list[str]) -> str | None:
    lowered = [str(node).strip().lower() for node in nodes]
    if "featurize.ecfp4_rdkit" in lowered:
        return "featurize.ecfp4_rdkit"
    if "featurize.morgan" in lowered:
        return "featurize.morgan"
    if "featurize.rdkit_labeled" in lowered:
        return "featurize.rdkit_labeled"
    if "featurize.rdkit" in lowered:
        return "featurize.rdkit"
    if "featurize.none" in lowered or "use.curated_features" in lowered:
        return "featurize.none"
    return None


def _allows_model(model_type: str, allowed_models: tuple[str, ...]) -> bool:
    if model_type.startswith("dl_"):
        return _DL_WILDCARD in allowed_models
    return model_type in set(allowed_models)


def _infer_runtime_profile_key(task_type: str, source_type: str, nodes: list[str]) -> str | None:
    if source_type == "tdc":
        if task_type == "classification":
            return "clf_tdc_benchmark"
        return None
    if source_type == "chembl":
        if task_type == "regression":
            return "reg_chembl_ic50"
        return None
    if task_type == "regression":
        if source_type == "local_csv" and "label.ic50" in nodes:
            return "reg_local_csv_ic50"
        return "reg_local_csv"
    if task_type == "classification":
        return "clf_local_csv"
    return None


def _runtime_child_tuning_issue(path: str, value: Any) -> ValidationIssue:
    return ValidationIssue(
        code="CFG_CHILD_LEVEL_HPO_UNSUPPORTED",
        path=path,
        message=(
            f"Runtime child-level hyperparameter search setting {path}={value!r} is disabled. "
            "Use DOE model_search to create parent-level fixed hyperparameter cases."
        ),
    )


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _append_child_level_tuning_issues(
    issues: list[ValidationIssue],
    tuning_cfg: dict[str, Any],
    path_prefix: str,
) -> None:
    if not isinstance(tuning_cfg, dict):
        return
    method = str(tuning_cfg.get("method", "fixed")).strip().lower() or "fixed"
    if method != "fixed":
        issues.append(_runtime_child_tuning_issue(f"{path_prefix}.method", method))
    if _is_truthy(tuning_cfg.get("use_hpo", False)):
        issues.append(_runtime_child_tuning_issue(f"{path_prefix}.use_hpo", tuning_cfg.get("use_hpo")))


def _append_get_data_sample_issues(
    issues: list[ValidationIssue],
    get_data_cfg: dict[str, Any],
    source_type: str,
) -> None:
    sample_cfg = get_data_cfg.get("sample")
    if sample_cfg is None:
        return
    if "max_rows" in get_data_cfg:
        issues.append(
            ValidationIssue(
                code="CFG_GET_DATA_SAMPLE_CONFLICT",
                path="get_data.sample",
                message="get_data.sample cannot be combined with get_data.max_rows.",
            )
        )
    if source_type and source_type not in _GET_DATA_SAMPLE_SOURCE_TYPES:
        issues.append(
            ValidationIssue(
                code="CFG_GET_DATA_SAMPLE_UNSUPPORTED_SOURCE",
                path="get_data.sample",
                message=(
                    "get_data.sample is only supported for CSV-like tabular sources: "
                    + ", ".join(sorted(_GET_DATA_SAMPLE_SOURCE_TYPES))
                    + "."
                ),
            )
        )
    if not isinstance(sample_cfg, dict):
        issues.append(
            ValidationIssue(
                code="CFG_GET_DATA_SAMPLE_INVALID",
                path="get_data.sample",
                message="get_data.sample must be a mapping with fraction, seed, and strategy fields.",
            )
        )
        return

    unknown_keys = sorted(set(sample_cfg) - _GET_DATA_SAMPLE_KEYS)
    if unknown_keys:
        issues.append(
            ValidationIssue(
                code="CFG_GET_DATA_SAMPLE_UNKNOWN_KEY",
                path="get_data.sample",
                message=(
                    "get_data.sample contains unknown key(s): "
                    + ", ".join(unknown_keys)
                    + ". Allowed keys: "
                    + ", ".join(sorted(_GET_DATA_SAMPLE_KEYS))
                    + "."
                ),
            )
        )

    fraction = sample_cfg.get("fraction")
    if isinstance(fraction, bool):
        fraction_value = None
    else:
        try:
            fraction_value = float(fraction)
        except (TypeError, ValueError):
            fraction_value = None
    if fraction_value is None or not 0 < fraction_value <= 1:
        issues.append(
            ValidationIssue(
                code="CFG_GET_DATA_SAMPLE_INVALID",
                path="get_data.sample.fraction",
                message="get_data.sample.fraction must be a number in the interval (0, 1].",
            )
        )

    if "seed" in sample_cfg:
        seed = sample_cfg.get("seed")
        if isinstance(seed, bool):
            seed_valid = False
        elif isinstance(seed, int):
            seed_valid = True
        elif isinstance(seed, float):
            seed_valid = seed.is_integer()
        elif isinstance(seed, str):
            text = seed.strip()
            seed_valid = bool(text) and text.lstrip("+-").isdigit()
        else:
            seed_valid = False
        if not seed_valid:
            issues.append(
                ValidationIssue(
                    code="CFG_GET_DATA_SAMPLE_INVALID",
                    path="get_data.sample.seed",
                    message="get_data.sample.seed must be an integer.",
                )
            )

    strategy = str(sample_cfg.get("strategy", "random") or "random").strip().lower()
    if strategy not in _GET_DATA_SAMPLE_STRATEGIES:
        issues.append(
            ValidationIssue(
                code="CFG_GET_DATA_SAMPLE_INVALID",
                path="get_data.sample.strategy",
                message=(
                    "get_data.sample.strategy must be one of: "
                    + ", ".join(sorted(_GET_DATA_SAMPLE_STRATEGIES))
                    + "."
                ),
            )
        )


def collect_config_issues(config: dict[str, Any], nodes: list[str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    blocks_present = set(config.keys())
    configless_blocks = {
        block for node, block in _CONFIGLESS_NODE_TO_BLOCK.items() if node in nodes
    }
    blocks_required_by_configurable_nodes = {
        block
        for node, block in _NODE_TO_BLOCK.items()
        if node in nodes and node not in _CONFIGLESS_NODE_TO_BLOCK
    }

    allowed_blocks = set(_ALWAYS_ALLOWED_BLOCKS)
    for node in nodes:
        block = _NODE_TO_BLOCK.get(node)
        if block:
            allowed_blocks.add(block)

    # Some nodes draw from multiple top-level blocks. train.timeseries reads
    # both `train` (model + params) and `split` (warmup/train/val/test lengths)
    # without participating in the standard `split` node, so the `split` block
    # is allowed even though the `split` node is forbidden in this pipeline.
    if "train.timeseries" in nodes:
        allowed_blocks.add("split")

    # Top-level block validity.
    for block in blocks_present:
        if block == "model":
            issues.append(
                ValidationIssue(
                    code="CFG_LEGACY_MODEL_BLOCK_FORBIDDEN",
                    path="model",
                    message="Top-level model block is no longer supported.",
                    hint="Move model settings to train.model.*",
                )
            )
            continue
        if block not in _KNOWN_TOP_LEVEL_BLOCKS:
            issues.append(
                ValidationIssue(
                    code="CFG_UNKNOWN_TOP_LEVEL_BLOCK",
                    path=block,
                    message="Unknown top-level config block.",
                )
            )
            continue
        if block not in allowed_blocks:
            if block in configless_blocks:
                continue
            issues.append(
                ValidationIssue(
                    code="CFG_BLOCK_NOT_ALLOWED_FOR_PIPELINE",
                    path=block,
                    message=f"Block '{block}' is present but no corresponding node is in pipeline.nodes.",
                )
            )

    # Configless-node specific checks.
    for node, block in _CONFIGLESS_NODE_TO_BLOCK.items():
        if (
            node in nodes
            and block in blocks_present
            and block not in blocks_required_by_configurable_nodes
        ):
            issues.append(
                ValidationIssue(
                    code="CFG_CONFIGLESS_NODE_HAS_BLOCK",
                    path=block,
                    message=(
                        f"Node {node} is configless and does not accept a top-level {block} block."
                    ),
                )
            )

    # Required blocks for specific nodes.
    if "train" in nodes and "train" not in blocks_present:
        issues.append(
            ValidationIssue(
                code="CFG_MISSING_BLOCK_FOR_NODE",
                path="train",
                message="Pipeline contains train node but train block is missing.",
            )
        )
    if "train.tdc" in nodes and "train_tdc" not in blocks_present:
        issues.append(
            ValidationIssue(
                code="CFG_MISSING_BLOCK_FOR_NODE",
                path="train_tdc",
                message="Pipeline contains train.tdc node but train_tdc block is missing.",
            )
        )

    if "train.timeseries" in nodes:
        # train.timeseries reads the `train` block (model + params) and the
        # `split` block (segment lengths). Both are required; without them the
        # node fails deep inside the trainer with a less helpful error, so we
        # surface the problem here in strict config validation.
        if "train" not in blocks_present:
            issues.append(
                ValidationIssue(
                    code="CFG_MISSING_BLOCK_FOR_NODE",
                    path="train",
                    message="Pipeline contains train.timeseries node but train block is missing.",
                )
            )
        else:
            ts_model_cfg = _as_dict(_as_dict(config.get("train")).get("model"))
            if not ts_model_cfg.get("type"):
                issues.append(
                    ValidationIssue(
                        code="CFG_MISSING_TRAIN_MODEL_TYPE",
                        path="train.model.type",
                        message="train.model.type is required for train.timeseries.",
                    )
                )
        if "split" not in blocks_present:
            issues.append(
                ValidationIssue(
                    code="CFG_MISSING_BLOCK_FOR_NODE",
                    path="split",
                    message="Pipeline contains train.timeseries node but split block is missing.",
                )
            )
        else:
            split_cfg = _as_dict(config.get("split"))
            # Mirror utilities.timeseries_io.parse_split_config exactly so the
            # strict validator accepts precisely what the runtime parser accepts:
            #   * all four keys required and integer-valued
            #   * each length must be >= 0 (NOT strictly positive)
            #   * train_len must be > 0
            #   * at least one of val_len / test_len must be > 0 (need an
            #     evaluation segment); warmup_len == 0 and a single zero
            #     eval segment are both allowed.
            parsed: dict[str, int] = {}
            for key in ("warmup_len", "train_len", "val_len", "test_len"):
                value = split_cfg.get(key)
                if value is None:
                    issues.append(
                        ValidationIssue(
                            code="CFG_MISSING_SPLIT_FIELD",
                            path=f"split.{key}",
                            message=f"split.{key} is required for train.timeseries.",
                        )
                    )
                    continue
                try:
                    ivalue = int(value)
                except (TypeError, ValueError):
                    issues.append(
                        ValidationIssue(
                            code="CFG_INVALID_SPLIT_FIELD",
                            path=f"split.{key}",
                            message=f"split.{key} must be an integer.",
                        )
                    )
                    continue
                if ivalue < 0:
                    issues.append(
                        ValidationIssue(
                            code="CFG_INVALID_SPLIT_FIELD",
                            path=f"split.{key}",
                            message=f"split.{key} must be >= 0.",
                        )
                    )
                    continue
                parsed[key] = ivalue
            # train_len must be strictly positive.
            if parsed.get("train_len") == 0:
                issues.append(
                    ValidationIssue(
                        code="CFG_INVALID_SPLIT_FIELD",
                        path="split.train_len",
                        message="split.train_len must be > 0 for train.timeseries.",
                    )
                )
            # Need at least one evaluation segment.
            if "val_len" in parsed and "test_len" in parsed and parsed["val_len"] == 0 and parsed["test_len"] == 0:
                issues.append(
                    ValidationIssue(
                        code="CFG_INVALID_SPLIT_FIELD",
                        path="split.test_len",
                        message=(
                            "train.timeseries requires val_len > 0 or test_len > 0; "
                            "without an evaluation segment there is nothing to score."
                        ),
                    )
                )

    # Cross-consistency between data source, model type, and train node for the
    # time-series pipeline. The trainer only runs under train.timeseries; using
    # a time-series data source or model under the tabular `train` (or train.tdc)
    # node would either crash or silently mis-handle the data, so reject it here.
    ts_source = str(_as_dict(config.get("get_data")).get("data_source", "")).strip().lower()
    ts_train_model_type = str(_as_dict(_as_dict(config.get("train")).get("model")).get("type", "")).strip().lower()
    if ts_source in _TIMESERIES_DATA_SOURCES and "get_data" in nodes:
        if "train" in nodes or "train.tdc" in nodes:
            issues.append(
                ValidationIssue(
                    code="CFG_TIMESERIES_SOURCE_REQUIRES_TS_NODE",
                    path="pipeline.nodes",
                    message=(
                        f"data_source={ts_source!r} is a time-series source and must be paired with the "
                        "'train.timeseries' node, not 'train' or 'train.tdc'."
                    ),
                )
            )
        elif "train.timeseries" not in nodes:
            issues.append(
                ValidationIssue(
                    code="CFG_TIMESERIES_SOURCE_REQUIRES_TS_NODE",
                    path="pipeline.nodes",
                    message=(
                        f"data_source={ts_source!r} is a time-series source but the pipeline has no "
                        "'train.timeseries' node."
                    ),
                )
            )
    if ts_train_model_type in _TIMESERIES_MODELS and "train" in nodes:
        issues.append(
            ValidationIssue(
                code="CFG_TIMESERIES_MODEL_REQUIRES_TS_NODE",
                path="train.model.type",
                message=(
                    f"model type {ts_train_model_type!r} is a time-series model and must be used with the "
                    "'train.timeseries' node, not the tabular 'train' node."
                ),
            )
        )

    if "train" in blocks_present and not isinstance(config.get("train"), dict):
        issues.append(
            ValidationIssue(
                code="CFG_INVALID_TRAIN_SCHEMA",
                path="train",
                message="train block must be a mapping/object.",
            )
        )
    train_cfg = _as_dict(config.get("train"))
    if "train" in nodes:
        model_value = train_cfg.get("model", None)
        if model_value is None:
            issues.append(
                ValidationIssue(
                    code="CFG_MISSING_TRAIN_MODEL",
                    path="train.model",
                    message="train.model block is required for train node.",
                )
            )
        else:
            model_cfg = _as_dict(model_value)
            if not model_cfg.get("type"):
                issues.append(
                    ValidationIssue(
                        code="CFG_MISSING_TRAIN_MODEL_TYPE",
                        path="train.model.type",
                        message="train.model.type is required.",
                    )
                )

    if "train" in nodes or "train.timeseries" in nodes:
        _append_child_level_tuning_issues(
            issues,
            _as_dict(train_cfg.get("tuning")),
            "train.tuning",
        )

    pipeline_cfg = _as_dict(config.get("pipeline"))
    configured_feature_input = _normalize_feature_input(pipeline_cfg.get("feature_input", ""))
    has_explicit_feature_input = any(node in _FEATURE_INPUT_NODES for node in nodes)
    explicit_feature_input = _feature_input_from_nodes(nodes)
    model_type = str(_as_dict(train_cfg.get("model")).get("type", "")).strip().lower()
    model_cfg = _as_dict(train_cfg.get("model"))
    foundation_mode = str(model_cfg.get("foundation", "none")).strip().lower() or "none"
    foundation_checkpoint = str(model_cfg.get("foundation_checkpoint", "")).strip()
    preprocess_cfg = _as_dict(config.get("preprocess"))
    preprocess_scaler = str(preprocess_cfg.get("scaler", "robust")).strip().lower() or "robust"
    curate_cfg = _as_dict(config.get("curate"))
    global_cfg = _as_dict(config.get("global"))
    task_type = str(global_cfg.get("task_type", "")).strip().lower()
    artifact_retention = str(global_cfg.get("artifact_retention", "")).strip().lower()
    get_data_cfg = _as_dict(config.get("get_data"))
    source_type = str(get_data_cfg.get("data_source", "")).strip().lower()
    if artifact_retention and artifact_retention not in _ARTIFACT_RETENTION_VALUES:
        issues.append(
            ValidationIssue(
                code="CFG_GLOBAL_ARTIFACT_RETENTION_INVALID",
                path="global.artifact_retention",
                message="global.artifact_retention must be one of: full, audit_light.",
            )
        )
    chemprop_like_noop_preprocess = (
        model_type in _CHEMPROP_LIKE_MODELS
        and configured_feature_input == "smiles_native"
        and preprocess_scaler == "none"
        and "preprocess.features" in nodes
        and "select.features" not in nodes
    )
    if configured_feature_input and explicit_feature_input and configured_feature_input != explicit_feature_input:
        issues.append(
            ValidationIssue(
                code="CFG_PIPELINE_FEATURE_INPUT_MISMATCH",
                path="pipeline.feature_input",
                message=(
                    f"pipeline.feature_input={configured_feature_input!r} does not match the explicit feature node "
                    f"{explicit_feature_input!r}."
                ),
            )
        )

    if "select.features" in nodes and "preprocess.features" not in nodes:
        issues.append(
            ValidationIssue(
                code="CFG_SELECT_REQUIRES_PREPROCESS",
                path="pipeline.nodes",
                message="select.features requires preprocess.features in this pipeline.",
            )
        )

    if (
        any(node in nodes for node in ("preprocess.features", "select.features"))
        and not has_explicit_feature_input
        and not chemprop_like_noop_preprocess
    ):
        issues.append(
            ValidationIssue(
                code="CFG_FEATURE_INPUT_NODE_REQUIRED",
                path="pipeline.nodes",
                message=(
                    "preprocess.features/select.features require an explicit feature input node "
                    "("
                    "featurize.none/featurize.rdkit/featurize.rdkit_labeled/"
                    "featurize.morgan/featurize.ecfp4_rdkit"
                    "), "
                    "except for the Chemprop/CheMeleon no-op preprocess branch."
                ),
            )
        )
    if "train" in nodes and not has_explicit_feature_input:
        if model_type and model_type not in _CHEMPROP_LIKE_MODELS:
            issues.append(
                ValidationIssue(
                    code="CFG_FEATURE_INPUT_NODE_REQUIRED",
                    path="pipeline.nodes",
                    message=(
                        "train for non-SMILES-native models requires an explicit feature input node "
                        "("
                        "featurize.none/featurize.rdkit/featurize.rdkit_labeled/"
                        "featurize.morgan/featurize.ecfp4_rdkit"
                        ")."
                    ),
                )
            )

    if task_type == "regression" and model_type in _CLASSIFICATION_ONLY_MODELS:
        issues.append(
            ValidationIssue(
                code="CFG_MODEL_TASK_MISMATCH",
                path="train.model.type",
                message=f"Model {model_type!r} is classification-only but task_type is regression.",
            )
        )
    if model_type in _CHEMPROP_LIKE_MODELS:
        resolved_feature_input = explicit_feature_input or configured_feature_input or ""
        if resolved_feature_input != "smiles_native":
            issues.append(
                ValidationIssue(
                    code="CFG_CHEMPROP_FEATURE_INPUT_UNSUPPORTED",
                    path="pipeline.feature_input",
                    message=(
                        "Chemprop/CheMeleon are SMILES-native and must use pipeline.feature_input=smiles_native "
                        "with no explicit tabular featurizer node."
                    ),
                )
            )
        if "select.features" in nodes or ("preprocess.features" in nodes and preprocess_scaler != "none"):
            issues.append(
                ValidationIssue(
                    code="CFG_CHEMPROP_PREPROCESS_UNSUPPORTED",
                    path="pipeline.nodes",
                    message=(
                        "Chemprop/CheMeleon only support the no-op preprocess branch "
                        "(preprocess.scaler=none, no select.features)."
                    ),
                )
            )
        if model_type == "chemeleon":
            foundation_mode = "chemeleon"
        if foundation_mode == "chemeleon" and not foundation_checkpoint:
            issues.append(
                ValidationIssue(
                    code="CFG_CHEMELEON_CHECKPOINT_REQUIRED",
                    path="train.model.foundation_checkpoint",
                    message="CheMeleon runs require train.model.foundation_checkpoint.",
                )
            )

    train_tdc_model_type = ""
    if "train_tdc" in blocks_present and not isinstance(config.get("train_tdc"), dict):
        issues.append(
            ValidationIssue(
                code="CFG_INVALID_TRAIN_SCHEMA",
                path="train_tdc",
                message="train_tdc block must be a mapping/object.",
            )
        )
    train_tdc_cfg = _as_dict(config.get("train_tdc"))
    if "train.tdc" in nodes:
        model_value = train_tdc_cfg.get("model", None)
        if model_value is None:
            issues.append(
                ValidationIssue(
                    code="CFG_MISSING_TRAIN_MODEL",
                    path="train_tdc.model",
                    message="train_tdc.model block is required for train.tdc node.",
                )
            )
        else:
            model_cfg = _as_dict(model_value)
            if not model_cfg.get("type"):
                issues.append(
                    ValidationIssue(
                        code="CFG_MISSING_TRAIN_MODEL_TYPE",
                        path="train_tdc.model.type",
                        message="train_tdc.model.type is required.",
                    )
                )
            else:
                train_tdc_model_type = str(model_cfg.get("type", "")).strip().lower()

    if "train.tdc" in nodes:
        _append_child_level_tuning_issues(
            issues,
            _as_dict(train_tdc_cfg.get("tuning")),
            "train_tdc.tuning",
        )

    if "get_data" in nodes:
        source_cfg = _as_dict(get_data_cfg.get("source"))
        _append_get_data_sample_issues(issues, get_data_cfg, source_type)
        if source_type == "local_csv" and not str(source_cfg.get("path", "")).strip():
            issues.append(
                ValidationIssue(
                    code="CFG_DATA_SOURCE_CONFIG_INVALID",
                    path="get_data.source.path",
                    message="local_csv source requires get_data.source.path.",
                )
            )
        if source_type == "chembl" and not (
            str(source_cfg.get("target_name", "")).strip()
            or str(source_cfg.get("target_chembl_id", "")).strip()
        ):
            issues.append(
                ValidationIssue(
                    code="CFG_DATA_SOURCE_CONFIG_INVALID",
                    path="get_data.source",
                    message="chembl source requires get_data.source.target_name or get_data.source.target_chembl_id.",
                )
            )
        profile_key = _infer_runtime_profile_key(task_type, source_type or "local_csv", nodes)
        if task_type and source_type and profile_key is None:
            issues.append(
                ValidationIssue(
                    code="CFG_SOURCE_TASK_UNSUPPORTED",
                    path="get_data.data_source",
                    message=(
                        f"source.type={source_type!r} is not supported with task_type={task_type!r} "
                        "in the current runtime profiles."
                    ),
                )
            )
        elif profile_key:
            profile_contract = _RUNTIME_PROFILE_CONTRACTS[profile_key]
            resolved_feature_input = explicit_feature_input or configured_feature_input or "none"
            if resolved_feature_input == "featurize.rdkit_labeled":
                resolved_feature_input = "featurize.rdkit"
            if resolved_feature_input not in set(profile_contract["allowed_feature_inputs"]):
                issues.append(
                    ValidationIssue(
                        code="CFG_FEATURE_INPUT_NOT_SUPPORTED",
                        path="pipeline.feature_input",
                        message=(
                            f"Feature input {resolved_feature_input!r} is not supported for runtime profile "
                            f"{profile_key!r}."
                        ),
                    )
                )
            profile_model_type = train_tdc_model_type if profile_key == "clf_tdc_benchmark" else model_type
            profile_model_path = "train_tdc.model.type" if profile_key == "clf_tdc_benchmark" else "train.model.type"
            if profile_model_type and not _allows_model(profile_model_type, profile_contract["allowed_models"]):
                issues.append(
                    ValidationIssue(
                        code="CFG_MODEL_NOT_SUPPORTED_FOR_PROFILE",
                        path=profile_model_path,
                        message=(
                            f"Model {profile_model_type!r} is not supported for runtime profile {profile_key!r}."
                        ),
                    )
                )
            if profile_key == "clf_tdc_benchmark":
                if "split" in nodes:
                    issues.append(
                        ValidationIssue(
                            code="CFG_PROFILE_NODE_UNSUPPORTED",
                            path="pipeline.nodes",
                            message="TDC benchmark profile does not support the split node.",
                        )
                    )
                if "train.tdc" not in nodes:
                    issues.append(
                        ValidationIssue(
                            code="CFG_PROFILE_TRAIN_NODE_MISMATCH",
                            path="pipeline.nodes",
                            message="TDC benchmark profile requires the train.tdc node.",
                )
            )

    row_filters_cfg = curate_cfg.get("row_filters")
    if row_filters_cfg is not None and not isinstance(row_filters_cfg, dict):
        issues.append(
            ValidationIssue(
                code="CFG_CURATE_ROW_FILTERS_INVALID",
                path="curate.row_filters",
                message="curate.row_filters must be a mapping of column names to allowed value(s).",
            )
        )

    # Legacy preprocess keys that leaked into other nodes.
    if "keep_all_columns" in preprocess_cfg:
        issues.append(
            ValidationIssue(
                code="CFG_LEGACY_PREPROCESS_KEY_FORBIDDEN",
                path="preprocess.keep_all_columns",
                message="preprocess.keep_all_columns is no longer supported.",
                hint="Move to curate.keep_all_columns",
            )
        )
    if "exclude_columns" in preprocess_cfg:
        issues.append(
            ValidationIssue(
                code="CFG_LEGACY_PREPROCESS_KEY_FORBIDDEN",
                path="preprocess.exclude_columns",
                message="preprocess.exclude_columns is no longer supported.",
                hint="Move to train.features.exclude_columns",
            )
        )
    if "scaler" in preprocess_cfg:
        scaler_value = str(preprocess_cfg.get("scaler", "")).strip().lower()
        if scaler_value not in {"robust", "standard", "minmax", "none"}:
            issues.append(
                ValidationIssue(
                    code="CFG_PREPROCESS_SCALER_INVALID",
                    path="preprocess.scaler",
                    message="preprocess.scaler must be one of: robust, standard, minmax, none.",
                )
            )

    return issues


def validate_config_strict(config: dict[str, Any], nodes: list[str]) -> None:
    issues = collect_config_issues(config, nodes)
    if issues:
        raise ConfigValidationError(issues)
