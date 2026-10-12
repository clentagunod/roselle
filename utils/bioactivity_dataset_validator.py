

import math
import re
import time

import pandas as pd
import requests
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


class BioactivityDatasetValidator:

    CHEMBL_ID_PATTERN = re.compile(r"^CHEMBL\d+$")
    CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

    def __init__(
        self,
        activity_value_column="activity_value",
        p_activity_value_column="p_activity_value",
        smiles_column="smiles",
        inchikey_column="inchikey",
        chembl_id_column="chembl_id",
        assay_id_column="assay_chembl_id",
        document_id_column="document_chembl_id",
        max_plausible_activity_nm=1e7,  
        p_scale_tolerance=0.01,
    ):
        self.activity_value_column = activity_value_column
        self.p_activity_value_column = p_activity_value_column
        self.smiles_column = smiles_column
        self.inchikey_column = inchikey_column
        self.chembl_id_column = chembl_id_column
        self.assay_id_column = assay_id_column
        self.document_id_column = document_id_column
        self.max_plausible_activity_nm = max_plausible_activity_nm
        self.p_scale_tolerance = p_scale_tolerance

    def _check_row(self, row):
        reasons = []

        chembl_id = row.get(self.chembl_id_column)
        if not isinstance(chembl_id, str) or not self.CHEMBL_ID_PATTERN.match(chembl_id):
            reasons.append("invalid_chembl_id_format")

        smiles = row.get(self.smiles_column)
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            reasons.append("unparseable_smiles")

        stored_inchikey = row.get(self.inchikey_column)
        if mol is not None and isinstance(stored_inchikey, str):
            try:
                recomputed = inchi.MolToInchiKey(mol)
            except Exception:
                recomputed = None
            if recomputed is None:
                reasons.append("inchikey_could_not_be_recomputed")
            elif recomputed != stored_inchikey:
                reasons.append("inchikey_mismatch")
        elif not isinstance(stored_inchikey, str):
            reasons.append("missing_inchikey")

        raw_activity_value = row.get(self.activity_value_column)
        activity_value = None
        if _is_missing(raw_activity_value):
            reasons.append("missing_or_non_numeric_activity_value")
        else:
            try:
                activity_value = float(raw_activity_value)
                if activity_value <= 0 or activity_value > self.max_plausible_activity_nm:
                    reasons.append("implausible_activity_value")
            except (TypeError, ValueError):
                reasons.append("missing_or_non_numeric_activity_value")

        p_value = row.get(self.p_activity_value_column)
        if activity_value is not None and activity_value > 0:
            if _is_missing(p_value):
                reasons.append("p_scale_not_numeric")
            else:
                try:
                    expected_p = -math.log10(activity_value * 1e-9)
                    if abs(float(p_value) - expected_p) > self.p_scale_tolerance:
                        reasons.append("p_scale_inconsistent_with_activity_value")
                except (TypeError, ValueError):
                    reasons.append("p_scale_not_numeric")

        if _is_missing(row.get(self.assay_id_column)):
            reasons.append("missing_assay_reference")
        if _is_missing(row.get(self.document_id_column)):
            reasons.append("missing_document_reference")

        return reasons

    def validate(self, df, verbose=True):
        all_reasons = [self._check_row(row) for _, row in df.iterrows()]

        report_df = df.copy().reset_index(drop=True)
        report_df["failure_reasons"] = ["; ".join(r) for r in all_reasons]
        report_df["is_valid"] = [len(r) == 0 for r in all_reasons]

        if verbose:
            n_valid = int(report_df["is_valid"].sum())
            print(f"[BioactivityDatasetValidator] {len(df)} rows checked, "
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

    def verify_against_chembl(self, df, sample_size=None, request_sleep_seconds=0.1):
        ids_to_check = df[self.chembl_id_column].dropna().unique().tolist()
        if sample_size is not None:
            ids_to_check = ids_to_check[:sample_size]

        results = {}
        for chembl_id in ids_to_check:
            url = f"{self.CHEMBL_BASE}/molecule/{chembl_id}.json"
            try:
                resp = requests.get(url, timeout=30)
                results[chembl_id] = resp.status_code == 200
            except requests.RequestException:
                results[chembl_id] = False
            time.sleep(request_sleep_seconds)

        n_confirmed = sum(results.values())
        print(f"[BioactivityDatasetValidator] Live ChEMBL check: "
              f"{n_confirmed}/{len(results)} chembl_id(s) confirmed to exist.")
        return results