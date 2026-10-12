

import os
import pandas as pd
import torch

from pathlib import Path
from rdkit import Chem
from torch_geometric.data import Data


BASE_DIR = Path(__file__).resolve().parent

CSV_FILE = BASE_DIR.parent / "data" / "roselle_dataset" / "roselle_analog_features.csv"
TARGET = "fp_1023"


OUTPUT_FILE = BASE_DIR.parent / "data" / "roselle_dataset" / "roselle_analog_molecular_graphs.pt"


SMILES_COLUMN_CANDIDATES = [
    "smiles",
    "SMILES",
    "canonical_smiles",
    "canonicalSMILES",
    "isomeric_smiles",
    "isomericSMILES",
]






def find_smiles_column(df):
    column_map = {
        column.strip().lower(): column
        for column in df.columns
    }

    for candidate in SMILES_COLUMN_CANDIDATES:
        candidate_lower = candidate.lower()

        if candidate_lower in column_map:
            return column_map[candidate_lower]

    
    
    for column in df.columns:
        if "smiles" in column.lower():
            return column

    raise ValueError(
        "Could not find a SMILES column.\n"
        f"Available columns: {list(df.columns)}"
    )


def find_target_column(df, target):
    column_map = {
        column.strip().lower(): column
        for column in df.columns
    }

    target_lower = target.strip().lower()

    if target_lower in column_map:
        return column_map[target_lower]

    raise ValueError(
        f"Target column '{target}' was not found.\n"
        f"Available columns: {list(df.columns)}"
    )






def atom_features(atom):
    atomic_number = atom.GetAtomicNum()

    degree = atom.GetDegree()

    formal_charge = atom.GetFormalCharge()

    num_hydrogens = atom.GetTotalNumHs()

    aromatic = int(atom.GetIsAromatic())

    hybridization = atom.GetHybridization()

    hybridization_map = {
        Chem.HybridizationType.SP: 0,
        Chem.HybridizationType.SP2: 1,
        Chem.HybridizationType.SP3: 2,
        Chem.HybridizationType.SP3D: 3,
        Chem.HybridizationType.SP3D2: 4,
    }

    hybridization_value = hybridization_map.get(
        hybridization,
        5
    )

    return [
        atomic_number,
        degree,
        formal_charge,
        num_hydrogens,
        aromatic,
        hybridization_value,
    ]


def bond_features(bond):
    bond_type = bond.GetBondType()

    bond_type_map = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }

    bond_type_value = bond_type_map.get(
        bond_type,
        4
    )

    aromatic = int(bond.GetIsAromatic())

    conjugated = int(bond.GetIsConjugated())

    in_ring = int(bond.IsInRing())

    return [
        bond_type_value,
        aromatic,
        conjugated,
        in_ring,
    ]


def smiles_to_graph(smiles, target):  
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )


    node_features = []

    for atom in molecule.GetAtoms():
        features = atom_features(atom)
        node_features.append(features)

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )

    
    edge_indices = []
    edge_features = []

    for bond in molecule.GetBonds():

        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()

        features = bond_features(bond)


        edge_indices.append([start, end])
        edge_features.append(features)

        edge_indices.append([end, start])
        edge_features.append(features)

    
    if len(edge_indices) == 0:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

        edge_attr = torch.empty(
            (0, 4),
            dtype=torch.float
        )

    else:

        edge_index = torch.tensor(
            edge_indices,
            dtype=torch.long
        ).t().contiguous()

        edge_attr = torch.tensor(
            edge_features,
            dtype=torch.float
        )

    

    y = torch.tensor(
        [float(target)],
        dtype=torch.float
    )

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=None
    )

    return graph



def csv_to_graph_dataset(
    csv_file,
    target,
    output_file
):

    print("=" * 60)
    print("CSV → MOLECULAR GRAPH CONVERTER")
    print("=" * 60)


    print(f"\nReading CSV: {csv_file}")

    df = pd.read_csv(csv_file)

    print(f"Number of rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    
    
    

    smiles_column = find_smiles_column(df)

    print(
        f"\nSMILES column detected: '{smiles_column}'"
    )

    
    
    

    target_column = find_target_column(
        df,
        target
    )

    print(
        f"Target column detected: '{target_column}'"
    )

    
    
    

    before = len(df)

    df = df.dropna(
        subset=[
            smiles_column,
            target_column
        ]
    )

    after = len(df)

    print(
        f"\nRemoved {before - after} rows "
        "with missing SMILES/target."
    )

    
    
    

    graphs = []

    invalid_molecules = 0

    for index, row in df.iterrows():

        smiles = str(row[smiles_column])
        target_value = row[target_column]

        try:

            
            target_value = float(target_value)

            graph = smiles_to_graph(
                smiles,
                target_value
            )

            
            graph.smiles = smiles

            
            graph.csv_index = int(index)

            graphs.append(graph)

        except Exception as error:

            invalid_molecules += 1

            print(
                f"\nSkipping row {index}"
            )

            print(
                f"SMILES: {smiles}"
            )

            print(
                f"Reason: {error}"
            )

    
    
    

    print("\n" + "=" * 60)

    print(
        f"Successfully converted: {len(graphs)} molecules"
    )

    print(
        f"Invalid molecules: {invalid_molecules}"
    )

    torch.save(
        graphs,
        output_file
    )

    print(
        f"Saved graph dataset to: {output_file}"
    )

    print("=" * 60)

    return graphs






if __name__ == "__main__":

    if not os.path.exists(CSV_FILE):

        raise FileNotFoundError(
            f"CSV file not found: {CSV_FILE}"
        )

    graphs = csv_to_graph_dataset(
        csv_file=CSV_FILE,
        target=TARGET,
        output_file=OUTPUT_FILE
    )

    
    
    

    if len(graphs) > 0:

        graph = graphs[0]

        print("\nExample graph:")
        print(graph)

        print("\nNode feature matrix:")
        print(graph.x)

        print("\nEdge index:")
        print(graph.edge_index)

        print("\nEdge features:")
        print(graph.edge_attr)

        print("\nTarget:")
        print(graph.y)

        print("\nSMILES:")
        print(graph.smiles)