import pandas as pd 
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "admet" / "mutagenicity" / "mutagenicity_training_features.csv"

df = pd.read_csv(CSV_PATH)


unique_counts = (df['label'] == 1.0).sum()
print(unique_counts)