import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "admet" / "solubility" / 'solubility_molecular_graphs.pt'

graphs = torch.load(
    CSV_PATH,
    weights_only=False
)

graph = graphs[0]

print(graphs[0].num_features)