from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent

pt_file = BASE_DIR / "roselle_analog_molecular_graphs.pt"

graphs = torch.load(
    pt_file,
    weights_only=False
)

print(graphs[0].num_features)