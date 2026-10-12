

import os

import pandas as pd

from utils.fetch_bioactivity_dataset import ChEMBLBioactivityFetcher
from utils.fetch_admet_dataset import TDCPropertyFetcher
from utils.admet_dataset_validator import AdmetDatasetValidator
from utils.rdkit_computation import MolecularFeatureComputer


ADMET_TARGETS = [
    {
        "property_label": "cyp3a4",
        "source": "chembl",
        "target_chembl_id": "CHEMBL340",
        "task_type": "regression",
    },
    {
        "property_label": "solubility",
        "source": "tdc",
        "tdc_category": "ADME",
        "tdc_dataset_name": "Solubility_AqSolDB",
        "task_type": "regression",
    },
    {
        "property_label": "herg",
        "source": "tdc",
        "tdc_category": "Tox",
        "tdc_dataset_name": "hERG",
        "task_type": "classification",
    },
    {
        "property_label": "mutagenicity",
        "source": "tdc",
        "tdc_category": "Tox",
        "tdc_dataset_name": "AMES",
        "task_type": "classification",
    },
]

DAILY_LIMIT = 2120  



ACTIVE_ADMET_LABELS = ["mutagenicity"]


def _fetch_raw_dataset(config):
    label = config["property_label"]

    if config["source"] == "chembl":
        fetcher = ChEMBLBioactivityFetcher(
            target_chembl_id=config["target_chembl_id"],
            target_label=label,
            daily_limit=DAILY_LIMIT,
        )
        csv_path = fetcher.run()
        df = pd.read_csv(csv_path)
        
        
        
        
        if "p_activity_value" in df.columns and "label" not in df.columns:
            df["label"] = df["p_activity_value"]
        return df, csv_path

    elif config["source"] == "tdc":
        fetcher = TDCPropertyFetcher(
            tdc_category=config["tdc_category"],
            tdc_dataset_name=config["tdc_dataset_name"],
            property_label=label,
            task_type=config["task_type"],
            daily_limit=DAILY_LIMIT,
        )
        csv_path = fetcher.run()
        return pd.read_csv(csv_path), csv_path

    else:
        raise ValueError(f"Unknown source '{config['source']}' for {label}")


def run_admet_target(config, feature_computer):
    label = config["property_label"]
    print(f"\n{'=' * 70}\nADMET PROPERTY: {label} (source={config['source']})\n{'=' * 70}")

    df, raw_csv_path = _fetch_raw_dataset(config)
    print(f"[{label}] Raw dataset: {raw_csv_path} ({len(df)} rows)")

    if df.empty:
        print(f"[{label}] No compounds fetched yet -- nothing to validate or featurize.")
        return None

    
    
    
    
    provenance_columns = (
        ("assay_chembl_id", "document_chembl_id")
        if config["source"] == "chembl"
        else ("dataset_source", "dataset_name")
    )

    validator = AdmetDatasetValidator(
        label_column="label",
        task_type=config["task_type"],
        provenance_columns=provenance_columns,
    )
    _report_df, valid_df = validator.validate(df)
    print(f"[{label}] {len(valid_df)}/{len(df)} rows passed validation.")

    if valid_df.empty:
        print(f"[{label}] No valid rows to featurize.")
        return None

    features_df = feature_computer.compute_features(
        valid_df, smiles_column="smiles", drop_invalid=True, verbose=True
    )

    output_dir = os.path.join("data", label)
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"{label}_training_features.csv")
    features_df.to_csv(output_csv, index=False)

    print(f"[{label}] Final dataset shape: {features_df.shape} "
          f"({len(feature_computer.feature_columns)} RDKit feature columns).")
    print(f"[{label}] Saved final training dataset: {output_csv}")

    return features_df


def main():
    feature_computer = MolecularFeatureComputer(cache_path="data/rdkit_feature_cache.json")
    config_by_label = {c["property_label"]: c for c in ADMET_TARGETS}

    results = {}
    for config in ADMET_TARGETS:
        label = config["property_label"]
        if ACTIVE_ADMET_LABELS is not None and label not in ACTIVE_ADMET_LABELS:
            continue
        results[label] = run_admet_target(config, feature_computer)

    print("\n" + "=" * 70)
    print("SUMMARY -- ADMET TRAINING DATASETS")
    print("=" * 70)
    for label, df in results.items():
        task_type = config_by_label[label]["task_type"]
        if df is None:
            print(f"  {label}: no dataset")
        else:
            print(f"  {label}: {df.shape[0]} compounds x {df.shape[1]} columns ({task_type})")
    print("=" * 70)


if __name__ == "__main__":
    main()