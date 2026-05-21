import argparse
import json
import logging
import math
from typing import List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

ROW_INDEX_COL = "__row_index"


def _find_smiles_column(columns: List[str]) -> Optional[str]:
    candidates = ["canonical_smiles", "smiles", "SMILES", "Smiles", "Drug", "drug"]
    for name in candidates:
        if name in columns:
            return name
    return None


def _clean_descriptor_value(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, ArithmeticError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _descriptor_values(mol: Chem.Mol) -> dict[str, float]:
    values: dict[str, float] = {}
    for desc_name, desc_fn in Descriptors._descList:
        try:
            values[desc_name] = _clean_descriptor_value(desc_fn(mol))
        except Exception:
            values[desc_name] = 0.0
    return values


def _fingerprint_values(mol: Chem.Mol, generator, n_bits: int) -> dict[str, int]:
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return {f"fp_{idx}": int(value) for idx, value in enumerate(arr.tolist())}


def build_features(
    df: pd.DataFrame,
    *,
    smiles_col: str,
    radius: int,
    n_bits: int,
) -> tuple[pd.DataFrame, list[int]]:
    if int(radius) != 2:
        raise ValueError("ECFP4 uses Morgan radius=2; use featurize.morgan for other radii.")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows: list[dict[str, object]] = []
    valid_rows: list[int] = []

    for idx, smiles in enumerate(df[smiles_col].astype(str).tolist()):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        row: dict[str, object] = {}
        row.update(_fingerprint_values(mol, generator, n_bits))
        row.update(_descriptor_values(mol))
        row[ROW_INDEX_COL] = int(df.iloc[idx][ROW_INDEX_COL])
        rows.append(row)
        valid_rows.append(idx)

    if not rows:
        raise ValueError("No valid SMILES for ECFP4+RDKit feature generation.")
    return pd.DataFrame(rows), valid_rows


def main(
    input_file: str,
    output_file: str,
    radius: int,
    n_bits: int,
    labeled_output_file: Optional[str],
    property_columns: List[str],
    metadata_output_file: Optional[str],
) -> None:
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv(input_file)
    smiles_col = _find_smiles_column(list(df.columns))
    if smiles_col is None:
        raise ValueError("Input file must contain a SMILES column.")

    if ROW_INDEX_COL not in df.columns:
        df[ROW_INDEX_COL] = df.index.astype(int)
    else:
        df[ROW_INDEX_COL] = pd.to_numeric(df[ROW_INDEX_COL], errors="raise").astype(int)

    features_df, valid_rows = build_features(
        df,
        smiles_col=smiles_col,
        radius=radius,
        n_bits=n_bits,
    )
    features_df.to_csv(output_file, index=False)
    logging.info("ECFP4+RDKit features saved to %s", output_file)

    if labeled_output_file:
        label_columns = [column for column in property_columns if column in df.columns]
        labels = df.iloc[valid_rows][label_columns].copy()
        combined = pd.concat(
            [
                features_df.reset_index(drop=True),
                labels.reset_index(drop=True),
            ],
            axis=1,
        )
        combined.to_csv(labeled_output_file, index=False)
        logging.info("Labeled ECFP4+RDKit features saved to %s", labeled_output_file)

    if metadata_output_file:
        with open(metadata_output_file, "w", encoding="utf-8") as meta_out:
            json.dump(
                {
                    "fingerprint": "ecfp4",
                    "morgan_radius": int(radius),
                    "n_bits": int(n_bits),
                    "descriptor_count": len(Descriptors._descList),
                    "feature_count": int(n_bits + len(Descriptors._descList)),
                },
                meta_out,
                indent=2,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate concatenated ECFP4 fingerprints and RDKit descriptors from SMILES."
    )
    parser.add_argument("input_file", type=str, help="Input CSV with SMILES.")
    parser.add_argument("output_file", type=str, help="Output CSV for ECFP4+RDKit features.")
    parser.add_argument("--radius", type=int, default=2, help="Morgan radius for ECFP4.")
    parser.add_argument("--n_bits", type=int, default=2048, help="Fingerprint length.")
    parser.add_argument(
        "--labeled-output-file",
        type=str,
        default=None,
        help="Optional output CSV with features + labels.",
    )
    parser.add_argument(
        "--property-columns",
        type=str,
        default=None,
        help="Comma-separated property columns to append to labeled output.",
    )
    parser.add_argument(
        "--metadata-output-file",
        type=str,
        default=None,
        help="Optional metadata JSON output.",
    )

    args = parser.parse_args()
    prop_cols = []
    if args.property_columns:
        prop_cols = [p.strip() for p in args.property_columns.split(",") if p.strip()]

    main(
        args.input_file,
        args.output_file,
        args.radius,
        args.n_bits,
        args.labeled_output_file,
        prop_cols,
        args.metadata_output_file,
    )
