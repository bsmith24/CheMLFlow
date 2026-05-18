---
name: chemlflow-study-runner
description: Coordinate end-to-end CheMLFlow studies across dataset profiling, runtime config design, DOE generation, local or Slurm execution, analysis, and audit. Use when a user asks an agent to run or improve a CheMLFlow experiment workflow rather than only build one config, review one DOE, or audit one analysis bundle.
---

# CheMLFlow Study Runner

## Purpose

Use this as the master CheMLFlow operating skill. It routes agents to the focused skills while keeping the workflow inside CheMLFlow's config, DOE, execution, and analysis system.

## Operating Principles

- Use CheMLFlow runtime configs, DOE generation, DOE execution, and `analysis.py` as the source of truth for scientific runs.
- Do not bypass CheMLFlow with ad hoc sklearn/PyTorch scripts unless the user explicitly asks for an external sanity check.
- Audit before ranking. Do not report "best model" claims until the analysis curator gate passes.
- Separate scientific parent configs from execution children. A runtime CV config is one fold/repeat slice; DOE fanout is the normal path for full K-fold results.
- Preserve generated configs, manifests, run statuses, logs, metrics, and analysis outputs.
- Ask or state assumptions for molecular science choices that change interpretation: Morgan vs RDKit, random vs scaffold, holdout vs CV vs nested CV, and whether SMILES-native models are in scope.

## Routing

1. For one runtime config, use `skills/chemlflow-config-builder`.
2. For a comparison, benchmark, or K-fold result, use `skills/chemlflow-doe-designer`.
3. For local execution, use `scripts/run_doe_local.py`.
4. For Slurm execution, use the repo's Slurm submit/orchestration workflow when present.
5. For local analysis, use `analysis.py --backend local`.
6. For Slurm analysis, use `analysis.py --backend slurm` with the orchestrator job/log inputs.
7. For final result validation, use `skills/chemlflow-analysis-curator`.

## Default Study Flow

1. Profile the dataset: rows, columns, target, SMILES column, missing values, invalid SMILES, duplicates/conflicts, class balance or target distribution.
2. Make the scientific defaults explicit:
   - Quick molecular baseline: Morgan + random split.
   - Chemistry generalization: scaffold CV.
   - Representation comparison: Morgan and RDKit, with balanced row coverage.
   - Final claims: CV, nested CV, or an untouched final holdout depending on the claim.
3. Generate a small pilot DOE or one execution child first.
4. Run locally with:

```bash
python scripts/run_doe_local.py --doe-dir <generated-doe-dir> --limit 1
```

5. Run the full local DOE when the pilot succeeds:

```bash
python scripts/run_doe_local.py --doe-dir <generated-doe-dir> --max-workers 1 --resume
```

6. Analyze local results:

```bash
python analysis.py --backend local --doe-dir <generated-doe-dir> --output-dir <analysis-dir>
```

7. Audit the analysis before summarizing metrics:

```bash
python skills/chemlflow-analysis-curator/scripts/audit_analysis.py <analysis-dir>
```

## Red Flags

- The agent trains directly with sklearn/PyTorch and presents those metrics as CheMLFlow results.
- A user asks for "5-fold CV" and the agent creates or runs only one runtime config without explaining it is one fold slice.
- Local runs require fake Slurm logs or fake `sacct` output.
- Random split results are described as chemistry-generalization results.
- A DOE compares Morgan/RDKit/scaffold branches with unbalanced or missing parent rows.
- Ranking happens before `report.json`, raw metrics rows, aggregate rows, and failed-case files are audited.
