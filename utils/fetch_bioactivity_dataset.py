

import csv
import json
import math
import os
import time
import datetime

import requests


class ChEMBLBioactivityFetcher:


    CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

    ASSAY_TYPE_PREFERENCE = {  # noqa: RUF012
        "B": 0,   
        "F": 1,   
        "A": 2,   
        "T": 3,   
        "U": 4,   
    }

    def __init__(
        self,
        target_chembl_id,
        target_label,
        output_dir="data",
        daily_limit=5,
        page_size=1000,
        min_confidence_score=8,
        standard_types=("IC50",),
        standard_units="nM",
        request_sleep_seconds=0.1,
        max_retries=3,
    ):
        self.target_chembl_id = target_chembl_id
        self.target_label = target_label
        self.daily_limit = daily_limit
        self.page_size = page_size
        self.min_confidence_score = min_confidence_score
        self.standard_types = list(standard_types)
        self.standard_units = standard_units
        self.request_sleep_seconds = request_sleep_seconds
        self.max_retries = max_retries

        
        
        
        self.output_dir = os.path.join(output_dir, target_label)
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_csv = os.path.join(self.output_dir, f"{target_label}_bioactivity.csv")
        self.cache_file = os.path.join(self.output_dir, f"{target_label}_assay_quality_cache.json")
        self.log_file = os.path.join(self.output_dir, f"{target_label}_fetch_log.txt")

        self._cache = self._load_cache()

    
    
    
    def _log(self, message, also_print=True):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{self.target_label}] {message}"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if also_print:
            print(line)

    
    
    
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2)

    
    
    
    def _load_existing_chembl_ids(self):
        existing = set()
        if os.path.exists(self.output_csv):
            with open(self.output_csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("chembl_id")
                    if cid:
                        existing.add(cid)
        return existing

    
    
    
    def _api_get(self, url, params=None):
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    self._log(f"  Rate limited (429). Waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_exc = e
                wait = 2 ** attempt
                self._log(f"  Request error ({e}). Retry {attempt}/{self.max_retries} after {wait}s...")
                time.sleep(wait)
        raise last_exc

    def _fetch_activity_page(self, offset):
        url = f"{self.CHEMBL_BASE}/activity.json"
        params = {
            "target_chembl_id": self.target_chembl_id,
            "standard_units": self.standard_units,
            "standard_relation": "=",
            "limit": self.page_size,
            "offset": offset,
        }
        
        
        
        
        
        
        
        if len(self.standard_types) == 1:
            params["standard_type"] = self.standard_types[0]
        else:
            params["standard_type__in"] = ",".join(self.standard_types)
        return self._api_get(url, params=params)

    def _get_assay_quality(self, assay_chembl_id):
        if assay_chembl_id in self._cache:
            return self._cache[assay_chembl_id]
        time.sleep(self.request_sleep_seconds)
        url = f"{self.CHEMBL_BASE}/assay/{assay_chembl_id}.json"
        data = self._api_get(url)
        quality = {
            "confidence_score": data.get("confidence_score"),
            "assay_type": data.get("assay_type"),
        }
        self._cache[assay_chembl_id] = quality
        return quality

    def _get_molecule_details(self, molecule_chembl_id):
        time.sleep(self.request_sleep_seconds)
        url = f"{self.CHEMBL_BASE}/molecule/{molecule_chembl_id}.json"
        return self._api_get(url)

    def _activity_quality_key(self, aq):
        assay_type = aq.get("assay_type")
        confidence = aq.get("confidence_score")
        type_rank = self.ASSAY_TYPE_PREFERENCE.get(assay_type, 99)
        confidence_rank = -(confidence if confidence is not None else -1)
        return (type_rank, confidence_rank)

    @staticmethod
    def _to_p_scale(value_nm):
        try:
            value_m = float(value_nm) * 1e-9
            if value_m <= 0:
                return None
            return round(-math.log10(value_m), 4)
        except (TypeError, ValueError):
            return None

    
    
    
    def _build_new_rows(self, existing_ids):
        new_rows = []
        offset = 0
        total_scanned = 0
        total_new_candidates = 0
        total_skipped_existing = 0
        total_skipped_low_quality = 0

        
        
        
        best_per_molecule = {}

        
        
        
        
        while len(best_per_molecule) < self.daily_limit:
            page = self._fetch_activity_page(offset)
            activities = page.get("activities", [])
            page_meta = page.get("page_meta", {})
            total_count = page_meta.get("total_count")

            if not activities:
                self._log("  No more activity pages available from ChEMBL.")
                break

            self._log(f"  Fetched page at offset={offset} ({len(activities)} activities, "
                       f"total available in ChEMBL for this query: {total_count}).")
            total_scanned += len(activities)

            for act in activities:
                molecule_chembl_id = act.get("molecule_chembl_id")
                assay_chembl_id = act.get("assay_chembl_id")
                value = act.get("standard_value")

                if not molecule_chembl_id or not assay_chembl_id or not value:
                    continue
                if act.get("data_validity_comment"):
                    continue  

                if molecule_chembl_id in existing_ids:
                    total_skipped_existing += 1
                    continue

                total_new_candidates += 1

                try:
                    aq = self._get_assay_quality(assay_chembl_id)
                except requests.RequestException as e:
                    self._log(f"  [skip] Could not fetch assay {assay_chembl_id}: {e}")
                    continue

                confidence = aq.get("confidence_score")
                if confidence is None or confidence < self.min_confidence_score:
                    total_skipped_low_quality += 1
                    continue

                key = self._activity_quality_key(aq)
                if (molecule_chembl_id not in best_per_molecule
                        or key < best_per_molecule[molecule_chembl_id][0]):
                    best_per_molecule[molecule_chembl_id] = (key, act, aq)

            
            self._save_cache()

            if len(best_per_molecule) >= self.daily_limit:
                break

            offset += self.page_size
            if total_count is not None and offset >= total_count:
                self._log("  Reached the end of ChEMBL's available activity records for this query.")
                break

        self._log(f"  Scanned {total_scanned} raw activities | "
                   f"{total_new_candidates} were new candidates | "
                   f"{total_skipped_existing} already in dataset (skipped) | "
                   f"{total_skipped_low_quality} failed the confidence-score filter.")

        
        for molecule_chembl_id, (key, act, aq) in best_per_molecule.items():
            if len(new_rows) >= self.daily_limit:
                break

            try:
                mol = self._get_molecule_details(molecule_chembl_id)
            except requests.RequestException as e:
                self._log(f"  [skip] Could not fetch molecule {molecule_chembl_id}: {e}")
                continue

            structures = mol.get("molecule_structures") or {}
            smiles = structures.get("canonical_smiles")
            inchikey = structures.get("standard_inchi_key")
            pref_name = mol.get("pref_name")

            if not smiles or not inchikey:
                continue

            props = mol.get("molecule_properties") or {}
            value = act.get("standard_value")

            row = {
                "chembl_id": molecule_chembl_id,
                "compound_name": pref_name if pref_name else molecule_chembl_id,
                "inchikey": inchikey,
                "smiles": smiles,
                "target_chembl_id": self.target_chembl_id,
                "target_label": self.target_label,
                "assay_chembl_id": act.get("assay_chembl_id"),
                "assay_type": aq.get("assay_type"),
                "assay_confidence_score": aq.get("confidence_score"),
                "document_chembl_id": act.get("document_chembl_id"),
                "standard_type": act.get("standard_type"),
                "activity_value": value,
                "activity_units": act.get("standard_units"),
                "p_activity_value": self._to_p_scale(value),
                "pchembl_value": act.get("pchembl_value"),
                
                
                "chembl_molecular_weight": props.get("full_mwt"),
                "chembl_alogp": props.get("alogp"),
                "chembl_qed_weighted": props.get("qed_weighted"),
            }
            new_rows.append(row)
            self._log(f"  [added] {molecule_chembl_id} ({row['compound_name']}) -> "
                       f"{row['standard_type']}={row['activity_value']} {row['activity_units']}, "
                       f"p-scale={row['p_activity_value']} "
                       f"(assay_type={row['assay_type']}, confidence={row['assay_confidence_score']})")

        return new_rows

    def _append_to_csv(self, rows):
        if not rows:
            self._log("No new rows to add this run.")
            return
        fieldnames = list(rows[0].keys())
        file_exists = os.path.exists(self.output_csv)
        with open(self.output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
        self._log(f"Appended {len(rows)} new compounds to {self.output_csv}.")

    def run(self):
        self._log("=" * 70)
        self._log(f"Starting run. Target={self.target_chembl_id} ({self.target_label}), "
                   f"daily_limit={self.daily_limit}, standard_types={self.standard_types}")

        existing_ids = self._load_existing_chembl_ids()
        self._log(f"Found {len(existing_ids)} compound(s) already in {self.output_csv}.")

        new_rows = self._build_new_rows(existing_ids)
        self._append_to_csv(new_rows)
        self._save_cache()

        total_now = len(existing_ids) + len(new_rows)
        self._log(f"Run complete. Dataset now has {total_now} total compound(s) in {self.output_csv}.")
        self._log("=" * 70)
        return self.output_csv