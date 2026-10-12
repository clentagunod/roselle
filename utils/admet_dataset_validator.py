

import math

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi

RDLogger.DisableLog("rdApp.*")


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


class AdmetDatasetValidator:
    def __init__(
        self,
        smiles_column="smiles",
        label_column="label",
        task_type="regression",             
        inchikey_column="inchikey",          
        provenance_columns=("dataset_source", "dataset_name"),
        allowed_classes=(0, 1),
        min_plausible_value=None,
        max_plausible_value=None,
    ):
        if task_type not in ("regression", "classification"):
            raise ValueError("task_type must be 'regression' or 'classification'")

        self.smiles_column = smiles_column
        self.label_column = label_column
        self.task_type = task_type
        self.inchikey_column = inchikey_column
        self.provenance_columns = list(provenance_columns)
        self.allowed_classes = set(allowed_classes)
        self.min_plausible_value = min_plausible_value
        self.max_plausible_value = max_plausible_value

    def _check_row(self, row):
        reasons = []

        smiles = row.get(self.smiles_column)
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            reasons.append("unparseable_smiles")

        if self.inchikey_column in row.index:
            stored_inchikey = row.get(self.inchikey_column)
            if _is_missing(stored_inchikey):
                reasons.append("missing_inchikey")
            elif mol is not None:
                try:
                    recomputed = inchi.MolToInchiKey(mol)
                except Exception:
                    recomputed = None
                if recomputed is None:
                    reasons.append("inchikey_could_not_be_recomputed")
                elif recomputed != stored_inchikey:
                    reasons.append("inchikey_mismatch")
        elif mol is not None:
            
            
            
            try:
                inchi.MolToInchiKey(mol)
            except Exception:
                reasons.append("structure_cannot_produce_valid_inchikey")

        label = row.get(self.label_column)
        if _is_missing(label):
            reasons.append("missing_label")
        elif self.task_type == "regression":
            try:
                value = float(label)
                if self.min_plausible_value is not None and value < self.min_plausible_value:
                    reasons.append("implausible_label_value")
                if self.max_plausible_value is not None and value > self.max_plausible_value:
                    reasons.append("implausible_label_value")
            except (TypeError, ValueError):
                reasons.append("non_numeric_label")
        else:  
            try:
                value = int(float(label))
                if value not in self.allowed_classes:
                    reasons.append("label_outside_allowed_classes")
            except (TypeError, ValueError):
                reasons.append("non_numeric_label")

        for col in self.provenance_columns:
            if col in row.index and _is_missing(row.get(col)):
                reasons.append(f"missing_provenance_{col}")

        return reasons

    def validate(self, df, verbose=True):
        all_reasons = [self._check_row(row) for _, row in df.iterrows()]

        report_df = df.copy().reset_index(drop=True)
        report_df["failure_reasons"] = ["; ".join(r) for r in all_reasons]
        report_df["is_valid"] = [len(r) == 0 for r in all_reasons]

        if verbose:
            n_valid = int(report_df["is_valid"].sum())
            print(f"[AdmetDatasetValidator] {len(df)} rows checked, "
                  f"{n_valid} valid, {len(df) - n_valid} flagged.")
            if len(df) - n_valid:
                flat_reasons = [r for reasons in all_reasons for r in reasons]
                for reason, count in pd.Series(flat_reasons).value_counts().items():
                    print(f"  {reason}: {count}")

        valid_df = (
            report_df[report_df["is_valid"]]
            .drop(columns=["failure_reasons", "is_valid"])
            .reset_index(drop=True)
        )
        return report_df, valid_df