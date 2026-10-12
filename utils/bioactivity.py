
import os

import pandas as pd

from utils.bioactivity_dataset_validator import BioactivityDatasetValidator
from utils.fetch_bioactivity_dataset import ChEMBLBioactivityFetcher
from utils.rdkit_computation import MolecularFeatureComputer

TARGETS = [
    {
        "target_chembl_id": "CHEMBL203",
        "target_label": "egfr",
    },
    {
        "target_chembl_id": "CHEMBL279",
        "target_label": "vegfr2",
    },
    {
        "target_chembl_id": "CHEMBL1824",
        "target_label": "her2",
    },
    {
        "target_chembl_id": "CHEMBL1907611",
        "target_label": "p53_mdm2",
        "standard_types": ("IC50", "Ki", "EC50"),
    },
]


DAILY_LIMIT = 406
ACTIVE_TARGET_LABELS = ["vegfr2"]

def run_target(target_config, feature_computer, validator):

    label = target_config["target_label"]

    target_id = target_config["target_chembl_id"]

    print(
        f"\n{'=' * 70}\n"
        f"TARGET: {label} ({target_id})\n"
        f"{'=' * 70}"
    )

    
    print(f"\n[{label}] Step 1: Fetching bioactivity data...")

    fetcher = ChEMBLBioactivityFetcher(
        target_chembl_id=target_id,
        target_label=label,
        daily_limit=DAILY_LIMIT,
        standard_types=target_config.get(
            "standard_types",
            ("IC50",)
        ),
    )

    csv_path = fetcher.run()

    print(f"[{label}] Raw dataset: {csv_path}")


    df = pd.read_csv(csv_path)

    if df.empty:

        print(
            f"[{label}] No compounds fetched yet -- "
            f"nothing to validate or featurize."
        )

        return None

    print(
        f"[{label}] Raw rows: {len(df)}"
    )

    identifier_columns = [
        "chembl_id",
        "compound_name",
        "cid",
        "pubchem_cid",
        "inchikey",
        "smiles",
    ]

    available_identifiers = [
        column
        for column in identifier_columns
        if column in df.columns
    ]

    print(
        f"[{label}] Available compound identifiers: "
        f"{available_identifiers}"
    )


    print(
        f"\n[{label}] Step 2: Validating dataset..."
    )

    _report_df, valid_df = validator.validate(df)

    print(
        f"[{label}] "
        f"{len(valid_df)}/{len(df)} rows passed validation."
    )
    if valid_df.empty:

        print(
            f"[{label}] No valid rows to featurize."
        )

        return None

    
    

    print(
        f"\n[{label}] Step 3: Computing RDKit features..."
    )

    features_df = feature_computer.compute_features(
        valid_df,
        smiles_column="smiles",
        drop_invalid=True,
        verbose=True,
    )

    

    if (
        "standard_type" in features_df.columns
        and "p_activity_value" in features_df.columns
    ):

        ic50_mask = (
            features_df["standard_type"]
            .astype(str)
            .str.upper()
            .eq("IC50")
        )

        features_df["pIC50"] = pd.NA

        features_df.loc[
            ic50_mask,
            "pIC50"
        ] = features_df.loc[
            ic50_mask,
            "p_activity_value"
        ]

    

    rdkit_columns = feature_computer.feature_columns

    preferred_metadata_columns = [
        
        "chembl_id",
        "compound_name",
        "cid",
        "pubchem_cid",
        "inchikey",
        "smiles",

        
        "target_chembl_id",
        "target_label",

        
        "assay_chembl_id",
        "assay_type",
        "assay_confidence_score",
        "document_chembl_id",

        
        "standard_type",
        "activity_value",
        "activity_units",
        "p_activity_value",
        "pchembl_value",
        "pIC50",

        
        "chembl_molecular_weight",
        "chembl_alogp",
        "chembl_qed_weighted",
    ]

    metadata_columns = [
        column
        for column in preferred_metadata_columns
        if column in features_df.columns
    ]

    
    remaining_columns = [
        column
        for column in features_df.columns
        if (
            column not in metadata_columns
            and column not in rdkit_columns
        )
    ]

    final_column_order = (
        metadata_columns
        + remaining_columns
        + rdkit_columns
    )

    features_df = features_df[
        final_column_order
    ]

    

    output_directory = os.path.join(
        "data",
        label
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    output_csv = os.path.join(
        output_directory,
        f"{label}_training_features.csv"
    )

    features_df.to_csv(
        output_csv,
        index=False
    )

    print(
        f"\n[{label}] RDKit feature computation complete."
    )

    print(
        f"[{label}] Final dataset shape: "
        f"{features_df.shape}"
    )

    print(
        f"[{label}] "
        f"RDKit feature columns: "
        f"{len(rdkit_columns)}"
    )

    print(
        f"[{label}] "
        f"RDKit descriptors: "
        f"{len(feature_computer.DESCRIPTOR_FUNCTIONS)}"
    )

    print(
        f"[{label}] "
        f"Morgan fingerprint bits: "
        f"{feature_computer.FINGERPRINT_N_BITS}"
    )

    print(
        f"[{label}] Saved final training dataset:"
    )

    print(
        f"        {output_csv}"
    )


    preview_columns = [
        column
        for column in [
            "chembl_id",
            "compound_name",
            "cid",
            "pubchem_cid",
            "smiles",
            "standard_type",
            "activity_value",
            "activity_units",
            "p_activity_value",
            "pchembl_value",
            "pIC50",
        ]
        if column in features_df.columns
    ]

    if preview_columns:

        print(
            f"\n[{label}] Training dataset preview:"
        )

        print(
            features_df[
                preview_columns
            ].head().to_string(index=False)
        )

    return features_df




def main():
    validator = BioactivityDatasetValidator(
        activity_value_column="activity_value",
        p_activity_value_column="p_activity_value",
    )

    feature_computer = MolecularFeatureComputer(
        cache_path="data/rdkit_feature_cache.json"
    )

    results = {}

    for target_config in TARGETS:

        label = target_config["target_label"]

        if (
            ACTIVE_TARGET_LABELS is not None
            and label not in ACTIVE_TARGET_LABELS
        ):
            continue

        results[label] = run_target(
            target_config,
            feature_computer,
            validator,
        )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "SUMMARY -- FINAL TRAINING DATASETS"
    )

    print(
        "=" * 70
    )

    for label, df in results.items():

        if df is None:

            print(
                f"  {label}: no dataset"
            )

            continue

        print(
            f"  {label}: "
            f"{df.shape[0]} compounds x "
            f"{df.shape[1]} columns"
        )

        print(
            f"       RDKit ML features: "
            f"{len(feature_computer.feature_columns)}"
        )

        print(
            f"       Output: "
            f"data/{label}/{label}_training_features.csv"
        )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()