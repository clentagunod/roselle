

import json
import os

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, inchi





RDLogger.DisableLog("rdApp.*")


class MolecularFeatureComputer:

    
    
    
    DESCRIPTOR_FUNCTIONS = {  
        "mol_weight": Descriptors.MolWt,
        "logp": Descriptors.MolLogP,
        "tpsa": Descriptors.TPSA,
        "num_h_donors": Descriptors.NumHDonors,
        "num_h_acceptors": Descriptors.NumHAcceptors,
        "num_rotatable_bonds": Descriptors.NumRotatableBonds,
        "num_aromatic_rings": Descriptors.NumAromaticRings,
        "num_rings": Descriptors.RingCount,
        "num_heavy_atoms": Descriptors.HeavyAtomCount,
        "fraction_csp3": Descriptors.FractionCSP3,
        "num_valence_electrons": Descriptors.NumValenceElectrons,
        "molar_refractivity": Descriptors.MolMR,
        "num_heteroatoms": Descriptors.NumHeteroatoms,
        "num_saturated_rings": Descriptors.NumSaturatedRings,
        "balaban_j": Descriptors.BalabanJ,
        "qed": Descriptors.qed,  
    }

    FINGERPRINT_RADIUS = 2
    FINGERPRINT_N_BITS = 1024

    def __init__(self, cache_path=None):
        self.cache_path = cache_path
        self._cache = self._load_cache()
        self.feature_columns = (
            list(self.DESCRIPTOR_FUNCTIONS.keys())
            + [f"fp_{i}" for i in range(self.FINGERPRINT_N_BITS)]
        )

    def _load_cache(self):
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        if self.cache_path:
            cache_dir = os.path.dirname(self.cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)

    def _compute_single(self, smiles):
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            return None, None, "unparseable_smiles"

        try:
            key = inchi.MolToInchiKey(mol)
        except Exception:
            key = None

        if key and key in self._cache:
            return self._cache[key], key, None

        try:
            features = {name: fn(mol) for name, fn in self.DESCRIPTOR_FUNCTIONS.items()}
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.FINGERPRINT_RADIUS, nBits=self.FINGERPRINT_N_BITS
            )
            for i, bit in enumerate(fp):
                features[f"fp_{i}"] = int(bit)
        except Exception as e:
            return None, key, f"descriptor_computation_failed: {e}"

        if key:
            self._cache[key] = features
        return features, key, None

    def compute_features(self, df, smiles_column="smiles", drop_invalid=True, verbose=True):
        records = []
        valid_mask = []
        failures = []

        for idx, smiles in df[smiles_column].items():
            features, _inchikey, error = self._compute_single(smiles)
            if features is None:
                failures.append((idx, smiles, error))
                records.append({col: None for col in self.feature_columns})
                valid_mask.append(False)
            else:
                records.append(features)
                valid_mask.append(True)

        feature_df = pd.DataFrame(records, columns=self.feature_columns)
        combined = pd.concat(
            [df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1
        )
        combined["_rdkit_features_valid"] = valid_mask

        if verbose:
            n_failed = len(failures)
            print(f"[MolecularFeatureComputer] {len(df)} rows in, "
                  f"{n_failed} failed SMILES parsing, "
                  f"{len(self.feature_columns)} feature columns produced.")
            for idx, smiles, error in failures[:10]:
                print(f"  [failed] row {idx} ({smiles!r}): {error}")
            if n_failed > 10:
                print(f"  ... and {n_failed - 10} more (truncated).")

        if drop_invalid:
            combined = (
                combined[combined["_rdkit_features_valid"]]
                .drop(columns=["_rdkit_features_valid"])
                .reset_index(drop=True)
            )
        else:
            combined = combined.rename(columns={"_rdkit_features_valid": "rdkit_features_valid"})

        self._save_cache()
        return combined


    