

import csv
import datetime
import os


class TDCPropertyFetcher:

    def __init__(
        self,
        tdc_category,        
        tdc_dataset_name,    
        property_label,      
        task_type,            
        output_dir="data",
        daily_limit=5,
    ):
        if tdc_category not in ("ADME", "Tox"):
            raise ValueError("tdc_category must be 'ADME' or 'Tox'")
        if task_type not in ("regression", "classification"):
            raise ValueError("task_type must be 'regression' or 'classification'")

        self.tdc_category = tdc_category
        self.tdc_dataset_name = tdc_dataset_name
        self.property_label = property_label
        self.task_type = task_type
        self.daily_limit = daily_limit

        self.output_dir = os.path.join(output_dir, property_label)
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_csv = os.path.join(self.output_dir, f"{property_label}_admet.csv")
        self.log_file = os.path.join(self.output_dir, f"{property_label}_fetch_log.txt")

    def _log(self, message, also_print=True):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{self.property_label}] {message}"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if also_print:
            print(line)

    def _load_tdc_dataframe(self):
        
        
        if self.tdc_category == "ADME":
            from tdc.single_pred import ADME as TDCLoader
        else:
            from tdc.single_pred import Tox as TDCLoader

        self._log(f"Loading TDC dataset '{self.tdc_dataset_name}' ({self.tdc_category})...")
        loader = TDCLoader(name=self.tdc_dataset_name)
        df = loader.get_data()
        self._log(f"TDC returned {len(df)} total compounds for '{self.tdc_dataset_name}'.")
        return df

    def _load_existing_drug_ids(self):
        existing = set()
        if os.path.exists(self.output_csv):
            with open(self.output_csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    drug_id = row.get("drug_id")
                    if drug_id:
                        existing.add(drug_id)
        return existing

    def run(self):
        self._log("=" * 70)
        self._log(f"Starting run. Dataset={self.tdc_dataset_name} ({self.tdc_category}), "
                   f"task_type={self.task_type}, daily_limit={self.daily_limit}")

        raw_df = self._load_tdc_dataframe()
        existing_ids = self._load_existing_drug_ids()
        self._log(f"Found {len(existing_ids)} compound(s) already in {self.output_csv}.")

        rows = []
        for _, r in raw_df.iterrows():
            if len(rows) >= self.daily_limit:
                break
            drug_id = str(r.get("Drug_ID"))
            if drug_id in existing_ids:
                continue
            rows.append({
                "drug_id": drug_id,
                "smiles": r.get("Drug"),
                "property_label": self.property_label,
                "task_type": self.task_type,
                "label": r.get("Y"),
                "dataset_source": "TDC",
                "dataset_name": self.tdc_dataset_name,
            })

        if not rows:
            self._log("No new rows to add this run.")
        else:
            file_exists = os.path.exists(self.output_csv)
            with open(self.output_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerows(rows)
            self._log(f"Appended {len(rows)} new compounds to {self.output_csv}.")

        total_now = len(existing_ids) + len(rows)
        self._log(f"Run complete. Dataset now has {total_now} total compound(s) in {self.output_csv}.")
        self._log("=" * 70)
        return self.output_csv