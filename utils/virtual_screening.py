"""
virtual_screening.py
Stage 1 virtual screening: 1285 labeled compounds -> ~13-18 docking candidates.

Method: hard ADMET filter, then Pareto-optimal ranking on 3 composite axes
(bioactivity, ADMET safety, drug-likeness). Pareto avoids defending arbitrary
weights; a weighted score is used only as a tiebreaker when the Pareto set
must be trimmed or filled to hit the target range.

Run: python virtual_screening.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

project_root = Path("../").resolve()

RDLogger.DisableLog("rdApp.*")

# ============================================================ CONFIG ===
INPUT_CSV = project_root / "final_roselle" / 'data' / 'final_dataset' / "final_roselle_dataset.csv"
OUTPUT_CSV = project_root/ "final_roselle" / 'data' / 'final_dataset' / "virtual_screening_survivors.csv"

# Edit these to match your actual column headers.
COL = {
    "smiles": "smiles",
    "bioactivity": ["pIC50 against EGFR", "pIC50 against HER2", "pIC50 against VEGFR2"],
    "solubility": "solubility",      # regression, higher = more soluble
    "cyp3a4": "cyp3a4",         # regression, LOWER = safer (less CYP inhibition)
    "herg_risk": "herg",              # classification prob of blocker, 0-1 (falls back to hard label)
    "mutagenicity_risk": "mutagenicity",
    "qed": "qed",                           # computed fresh if column absent
}

BIOACTIVITY_AGG = "mean"          # "mean" (broad-spectrum) or "max" (best single target)
EXCLUDE_MUTAGENIC = True          # hard filter, applied before ranking
MUTAGENIC_THRESHOLD = 0.5
EXCLUDE_HERG_BLOCKER = False      # soft-penalized in admet_safety_score instead, by default
HERG_THRESHOLD = 0.5

TIEBREAK_WEIGHTS = {"bioactivity": 0.5, "admet_safety": 0.3, "qed": 0.2}
TARGET_MIN, TARGET_MAX = 13, 100


# ============================================================ HELPERS ==
def minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def ensure_qed(df: pd.DataFrame) -> pd.Series:
    col = COL["qed"]
    if col in df.columns and df[col].notna().all():
        return df[col].astype(float)
    out = pd.Series(np.nan, index=df.index)
    for i, smi in df[COL["smiles"]].items():
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        out.loc[i] = Descriptors.qed(mol) if mol is not None else np.nan
    return out


def risk_column(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Returns a 0-1 risk value: probability column if present, else hard 0/1 label."""
    if col_name not in df.columns:
        raise KeyError(f"Missing expected column '{col_name}'")
    return df[col_name].astype(float).clip(0, 1)


def bioactivity_score(df: pd.DataFrame) -> pd.Series:
    normed = pd.concat([minmax(df[c]) for c in COL["bioactivity"]], axis=1)
    return normed.mean(axis=1) if BIOACTIVITY_AGG == "mean" else normed.max(axis=1)


def admet_safety_score(df: pd.DataFrame) -> pd.Series:
    solubility = minmax(df[COL["solubility"]])
    cyp3a4_safety = 1 - minmax(df[COL["cyp3a4"]])          # lower inhibition = safer
    herg_safety = 1 - risk_column(df, COL["herg_risk"])
    mut_safety = 1 - risk_column(df, COL["mutagenicity_risk"])
    return pd.concat([solubility, cyp3a4_safety, herg_safety, mut_safety], axis=1).mean(axis=1)


def pareto_mask(scores: np.ndarray) -> np.ndarray:
    """Non-dominated set for objectives to MAXIMIZE. Vectorized O(n^2)."""
    ge = np.all(scores[:, None, :] >= scores[None, :, :], axis=2)
    gt = np.any(scores[:, None, :] > scores[None, :, :], axis=2)
    dominates = ge & gt
    np.fill_diagonal(dominates, False)
    return ~dominates.any(axis=0)


def apply_hard_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if EXCLUDE_MUTAGENIC:
        mask &= risk_column(df, COL["mutagenicity_risk"]) < MUTAGENIC_THRESHOLD
    if EXCLUDE_HERG_BLOCKER:
        mask &= risk_column(df, COL["herg_risk"]) < HERG_THRESHOLD
    return df[mask].reset_index(drop=True)


# ============================================================ MAIN =====
def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} compounds.")

    df = apply_hard_filters(df)
    print(f"{len(df)} remain after ADMET hard filters.")
    if df.empty:
        raise RuntimeError("No compounds survived the hard filters.")

    df["qed"] = ensure_qed(df)
    df["bioactivity_score"] = bioactivity_score(df)
    df["admet_safety_score"] = admet_safety_score(df)

    axes = df[["bioactivity_score", "admet_safety_score", "qed"]].to_numpy(dtype=float)
    df["pareto_optimal"] = pareto_mask(axes)

    w = TIEBREAK_WEIGHTS
    df["ml_composite_score"] = (
        w["bioactivity"] * df["bioactivity_score"]
        + w["admet_safety"] * df["admet_safety_score"]
        + w["qed"] * df["qed"]
    )

    pareto_df = df[df["pareto_optimal"]].sort_values("ml_composite_score", ascending=False)
    remainder_df = df[~df["pareto_optimal"]].sort_values("ml_composite_score", ascending=False)

    if len(pareto_df) > TARGET_MAX:
        survivors = pareto_df.head(TARGET_MAX)
    elif len(pareto_df) < TARGET_MIN:
        needed = TARGET_MIN - len(pareto_df)
        survivors = pd.concat([pareto_df, remainder_df.head(needed)])
    else:
        survivors = pareto_df

    survivors = survivors.sort_values("ml_composite_score", ascending=False).reset_index(drop=True)
    survivors.to_csv(OUTPUT_CSV, index=False)

    print(f"Selected {len(survivors)} survivors -> {OUTPUT_CSV}")
    print(f"Pareto-optimal set size (pre-trim/fill): {len(pareto_df)}")


if __name__ == "__main__":
    main()