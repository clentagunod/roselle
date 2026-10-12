"""
docking.py
============================================================

ROSSELLE VIRTUAL SCREENING - STAGE 2

Dock virtual-screening survivors against:

    EGFR   -> PDB 1M17
    HER2   -> PDB 3PP0
    VEGFR2 -> PDB 4ASD

The script:

    1. Loads virtual-screening survivors from CSV.
    2. Downloads the target PDB structures from RCSB.
    3. Detects the crystallographic ligand automatically.
    4. Uses the ligand centroid as the docking-box center.
    5. Removes alternate conformations.
    6. Creates protein-only PDB files.
    7. Prepares receptors using Meeko.
    8. Handles Meeko template problems in batch mode.
    9. Converts ligand SMILES -> 3D SDF.
   10. Handles disconnected/multi-fragment SMILES.
   11. Converts SDF -> PDBQT using Meeko.
   12. Runs AutoDock Vina.
   13. Extracts docking affinity.
   14. Ranks compounds by mean docking affinity.
   15. Saves the final top N compounds.

IMPORTANT:
    This script intentionally does NOT use OpenBabel.

The major ligand problem fixed here is:

    RDKit molecule has 2 fragments. Must have 1.

For multi-fragment SMILES, the script keeps the largest
fragment and records the removed fragments in the output CSV.

Example:

    CC[NH3+].Cl-

becomes:

    CC[NH3+]

The original SMILES is preserved.

============================================================
REQUIREMENTS
============================================================

Conda environment:

    conda activate docking

Recommended:

    conda install -c conda-forge rdkit pandas numpy requests scipy gemmi
    pip install meeko

AutoDock Vina:

    vina

Meeko executables:

    mk_prepare_receptor.exe
    mk_prepare_ligand.exe

Check:

    where vina
    where mk_prepare_receptor
    where mk_prepare_ligand

============================================================
RUN
============================================================

    python docking.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem


# ============================================================
# SETTINGS
# ============================================================

RDLogger.DisableLog("rdApp.*")


# ============================================================
# PROJECT PATHS
# ============================================================

# docking.py is expected to be inside:
#
# final_roselle/
#     utils/
#         docking.py
#
# Therefore parents[1] = final_roselle/

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "final_dataset"
    / "virtual_screening_survivors.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "final_dataset"
    / "final_docking_survivors.csv"
)

# Intermediate full docking table.
FULL_RESULTS_CSV = (
    PROJECT_ROOT
    / "data"
    / "final_dataset"
    / "docking_all_results.csv"
)

WORKDIR = PROJECT_ROOT / "docking_workdir"


# ============================================================
# DATASET SETTINGS
# ============================================================

SMILES_COL = "smiles"


# ============================================================
# TARGETS
# ============================================================

TARGETS = {
    "egfr": "1M17",
    "her2": "3PP0",
    "vegfr2": "4ASD",
}


# ============================================================
# DOCKING SETTINGS
# ============================================================

BOX_SIZE = (
    25.0,
    25.0,
    25.0,
)

VINA_EXHAUSTIVENESS = 16

FINAL_TOP_N = 50


# ============================================================
# MULTI-FRAGMENT SETTINGS
# ============================================================

# If a SMILES contains:
#
#     molecule.counterion
#
# we keep the largest fragment by heavy-atom count.
#
# This is appropriate for many salts and counterions.
#
# Example:
#
#     C[NH3+].Cl-
#
# -> C[NH3+]
#
# The original SMILES is always retained in the output.

KEEP_LARGEST_FRAGMENT = True


# ============================================================
# NON-LIGAND HETATM RESIDUES
# ============================================================

NON_LIGAND_HET = {

    # --------------------------------------------------------
    # Water
    # --------------------------------------------------------

    "HOH",
    "WAT",

    # --------------------------------------------------------
    # Common ions
    # --------------------------------------------------------

    "NA",
    "CL",
    "K",
    "MG",
    "CA",
    "ZN",
    "MN",
    "FE",
    "NI",
    "CO",
    "CU",

    # --------------------------------------------------------
    # Phosphate / sulfate
    # --------------------------------------------------------

    "SO4",
    "PO4",

    # --------------------------------------------------------
    # Solvents / crystallization additives
    # --------------------------------------------------------

    "GOL",
    "EDO",
    "PEG",
    "DMS",
    "TRS",
    "MES",
    "CIT",
    "ACT",
    "FMT",
    "MPD",
    "BME",
    "IPA",

    # --------------------------------------------------------
    # Other common additives
    # --------------------------------------------------------

    "EOH",
    "MOH",
    "URE",
    "ACE",
    "ACN",
}


# ============================================================
# VINA AFFINITY REGEX
# ============================================================

AFFINITY_RE = re.compile(
    r"REMARK VINA RESULT:\s*([-\d.]+)"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def find_executable(*names: str) -> str:
    """
    Find an executable on PATH.

    Works on Windows, Linux, and macOS.
    """

    for name in names:

        path = shutil.which(name)

        if path is not None:
            return path

    raise EnvironmentError(
        "Could not find any of these executables on PATH:\n"
        + "\n".join(f"    {name}" for name in names)
    )


# ------------------------------------------------------------
# External command runner
# ------------------------------------------------------------

def run_command(
    cmd: list[str],
    description: str,
    raise_on_error: bool = True,
) -> subprocess.CompletedProcess:

    print()
    print(f"  Running: {description}")

    print("  Command:")

    print(
        "    "
        + " ".join(
            f'"{str(x)}"' if " " in str(x) else str(x)
            for x in cmd
        )
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.stdout.strip():

        print("  stdout:")

        print(
            result.stdout.strip()
        )

    if result.returncode != 0:

        print()
        print(
            f"  ERROR: command failed "
            f"with exit code {result.returncode}"
        )

        if result.stderr.strip():

            print()
            print("  stderr:")

            print(
                result.stderr.strip()
            )

        if raise_on_error:

            raise RuntimeError(
                f"{description} failed "
                f"with exit code {result.returncode}"
            )

    return result


# ============================================================
# DEPENDENCY CHECK
# ============================================================

def check_dependencies() -> dict[str, str]:

    print("=" * 70)
    print("CHECKING DEPENDENCIES")
    print("=" * 70)

    vina = find_executable(
        "vina",
        "vina.exe",
    )

    mk_prepare_receptor = find_executable(
        "mk_prepare_receptor",
        "mk_prepare_receptor.exe",
    )

    mk_prepare_ligand = find_executable(
        "mk_prepare_ligand",
        "mk_prepare_ligand.exe",
    )

    executables = {
        "vina": vina,
        "mk_prepare_receptor": mk_prepare_receptor,
        "mk_prepare_ligand": mk_prepare_ligand,
    }

    for name, path in executables.items():

        print(
            f"{name:25s}: {path}"
        )

    print()

    return executables


# ============================================================
# PDB DOWNLOAD
# ============================================================

def download_pdb(
    pdb_id: str,
    out_path: Path,
) -> Path:

    if out_path.exists():

        print(
            f"  Using existing PDB: {out_path}"
        )

        return out_path

    url = (
        f"https://files.rcsb.org/download/"
        f"{pdb_id}.pdb"
    )

    print(
        f"  Downloading {pdb_id}..."
    )

    response = requests.get(
        url,
        timeout=60,
    )

    response.raise_for_status()

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path.write_text(
        response.text,
        encoding="utf-8",
    )

    print(
        f"  Saved: {out_path}"
    )

    return out_path


# ============================================================
# PDB PARSING
# ============================================================

def parse_pdb_atoms(
    pdb_text: str,
):
    """
    Parse PDB ATOM and HETATM records.

    Alternate conformations:

        blank -> kept
        A     -> kept
        B/C/D -> discarded

    Returns:

        atom_lines
        het_groups
    """

    atom_lines = []

    het_groups = {}

    # --------------------------------------------------------
    # Detect alternate locations
    # --------------------------------------------------------

    residue_altlocs = {}

    for line in pdb_text.splitlines():

        record = line[0:6].strip()

        if record not in {
            "ATOM",
            "HETATM",
        }:
            continue

        chain = line[21].strip()

        resnum = line[22:26].strip()

        icode = line[26].strip()

        residue_key = (
            chain,
            resnum,
            icode,
        )

        altloc = line[16].strip()

        if altloc:

            residue_altlocs.setdefault(
                residue_key,
                set(),
            ).add(
                altloc
            )

    alternate_residues = {
        key: sorted(values)
        for key, values in residue_altlocs.items()
        if len(values) > 1
    }

    if alternate_residues:

        print()
        print(
            "  Alternate conformations detected:"
        )

        for residue, altlocs in alternate_residues.items():

            print(
                f"    chain={residue[0] or '-'} "
                f"residue={residue[1]} "
                f"icode={residue[2] or '-'} "
                f"altlocs={altlocs}"
            )

        print(
            "  Keeping blank altloc and altloc A."
        )

    # --------------------------------------------------------
    # Parse records
    # --------------------------------------------------------

    for line in pdb_text.splitlines():

        record = line[0:6].strip()

        if record not in {
            "ATOM",
            "HETATM",
        }:
            continue

        altloc = line[16].strip()

        # ----------------------------------------------------
        # Keep only blank and A conformations
        # ----------------------------------------------------

        if altloc not in {
            "",
            "A",
        }:
            continue

        # ----------------------------------------------------
        # Clean altloc character
        # ----------------------------------------------------

        clean_line = (
            line[:16]
            + " "
            + line[17:]
        )

        # ----------------------------------------------------
        # Protein
        # ----------------------------------------------------

        if record == "ATOM":

            atom_lines.append(
                clean_line
            )

            continue

        # ----------------------------------------------------
        # Heteroatom
        # ----------------------------------------------------

        resname = (
            line[17:20]
            .strip()
            .upper()
        )

        if resname in NON_LIGAND_HET:
            continue

        chain = line[21].strip()

        resnum = line[22:26].strip()

        icode = line[26].strip()

        try:

            x = float(line[30:38])

            y = float(line[38:46])

            z = float(line[46:54])

        except ValueError:

            continue

        key = (
            resname,
            chain,
            resnum,
        )

        het_groups.setdefault(
            key,
            [],
        ).append(
            (
                x,
                y,
                z,
            )
        )

    return (
        atom_lines,
        het_groups,
    )


# ============================================================
# DETECT BINDING LIGAND
# ============================================================

def detect_box_center(
    het_groups: dict,
) -> tuple[float, float, float]:

    if not het_groups:

        raise RuntimeError(
            "No candidate ligand was found "
            "for automatic box detection."
        )

    # --------------------------------------------------------
    # Ignore tiny hetero groups when possible.
    # --------------------------------------------------------

    candidates = {
        key: coords
        for key, coords in het_groups.items()
        if len(coords) >= 4
    }

    if not candidates:

        candidates = het_groups

    # --------------------------------------------------------
    # Select largest heteroatom group
    # --------------------------------------------------------

    ligand_key = max(
        candidates,
        key=lambda key: len(
            candidates[key]
        ),
    )

    coords = np.asarray(
        candidates[ligand_key],
        dtype=float,
    )

    center = coords.mean(
        axis=0
    )

    print()
    print(
        "  Detected binding ligand:"
    )

    print(
        f"    residue = {ligand_key}"
    )

    print(
        f"    atoms   = {len(coords)}"
    )

    print(
        f"    center  = "
        f"{center.round(3).tolist()}"
    )

    return (
        float(center[0]),
        float(center[1]),
        float(center[2]),
    )


# ============================================================
# RECEPTOR PREPARATION
# ============================================================

def prepare_receptor(
    pdb_id: str,
    target_label: str,
    target_dir: Path,
    mk_prepare_receptor: str,
) -> dict:

    print()
    print("=" * 70)
    print(
        f"PREPARING RECEPTOR: "
        f"{target_label.upper()} ({pdb_id})"
    )
    print("=" * 70)

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Download PDB
    # --------------------------------------------------------

    raw_pdb = download_pdb(
        pdb_id,
        target_dir / f"{pdb_id}.pdb",
    )

    pdb_text = raw_pdb.read_text(
        encoding="utf-8",
        errors="replace",
    )

    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    (
        atom_lines,
        het_groups,
    ) = parse_pdb_atoms(
        pdb_text
    )

    if not atom_lines:

        raise RuntimeError(
            f"No ATOM records found in {raw_pdb}"
        )

    # --------------------------------------------------------
    # Detect binding site
    # --------------------------------------------------------

    box_center = detect_box_center(
        het_groups
    )

    # --------------------------------------------------------
    # Protein-only PDB
    # --------------------------------------------------------

    protein_pdb = (
        target_dir
        / f"{pdb_id}_protein.pdb"
    )

    protein_pdb.write_text(
        "\n".join(atom_lines)
        + "\nEND\n",
        encoding="utf-8",
    )

    print(
        f"  Protein-only PDB: {protein_pdb}"
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    receptor_base = (
        target_dir
        / f"{pdb_id}_receptor"
    )

    receptor_pdbqt = (
        target_dir
        / f"{pdb_id}_receptor.pdbqt"
    )

    config_txt = (
        target_dir
        / f"{pdb_id}_receptor.box.txt"
    )

    # --------------------------------------------------------
    # Reuse receptor if already prepared
    # --------------------------------------------------------

    if (
        receptor_pdbqt.exists()
        and config_txt.exists()
    ):

        print()
        print(
            "  Existing prepared receptor found."
        )

        print(
            f"    receptor = {receptor_pdbqt}"
        )

        print(
            f"    box      = {config_txt}"
        )

        return {
            "receptor_pdbqt": receptor_pdbqt,
            "config_txt": config_txt,
            "box_center": box_center,
        }

    # --------------------------------------------------------
    # Meeko receptor preparation
    #
    # --delete_bad_res
    #
    # This handles residues such as:
    #
    #     B:895
    #
    # that Meeko cannot template-match.
    #
    # --default_altloc A
    #
    # ensures a deterministic alternate conformation.
    # --------------------------------------------------------

    cmd = [

        mk_prepare_receptor,

        "--read_pdb",
        str(protein_pdb),

        "-o",
        str(receptor_base),

        "-p",

        "-v",

        "--box_size",
        str(BOX_SIZE[0]),
        str(BOX_SIZE[1]),
        str(BOX_SIZE[2]),

        "--box_center",
        str(box_center[0]),
        str(box_center[1]),
        str(box_center[2]),

        "--default_altloc",
        "A",

        "--delete_bad_res",
    ]

    result = run_command(
        cmd,
        f"Meeko receptor preparation for "
        f"{target_label}",
        raise_on_error=False,
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"Meeko receptor preparation for "
            f"{target_label} failed."
        )

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    if not receptor_pdbqt.exists():

        raise RuntimeError(
            "Meeko completed but receptor PDBQT "
            "was not created:\n"
            f"{receptor_pdbqt}"
        )

    if not config_txt.exists():

        raise RuntimeError(
            "Meeko completed but box configuration "
            "was not created:\n"
            f"{config_txt}"
        )

    print()
    print(
        "  Receptor preparation successful."
    )

    print(
        f"    receptor = {receptor_pdbqt}"
    )

    print(
        f"    box      = {config_txt}"
    )

    return {
        "receptor_pdbqt": receptor_pdbqt,
        "config_txt": config_txt,
        "box_center": box_center,
    }


# ============================================================
# FRAGMENT HANDLING
# ============================================================

def process_ligand_fragments(
    mol: Chem.Mol,
    original_smiles: str,
    tag: str,
):
    """
    Handle disconnected SMILES.

    Example:

        C[NH3+].Cl-

    contains two fragments.

    The largest fragment is retained.

    Returns:

        mol
        docking_smiles
        fragment_count
        removed_fragments
    """

    fragments = Chem.GetMolFrags(
        mol,
        asMols=True,
        sanitizeFrags=True,
    )

    fragment_count = len(
        fragments
    )

    # --------------------------------------------------------
    # Normal single-component molecule
    # --------------------------------------------------------

    if fragment_count == 1:

        docking_smiles = Chem.MolToSmiles(
            mol,
            canonical=True,
        )

        return (
            mol,
            docking_smiles,
            1,
            "",
        )

    # --------------------------------------------------------
    # Multi-component molecule
    # --------------------------------------------------------

    print()
    print(
        f"  [WARNING] {tag} contains "
        f"{fragment_count} molecular fragments."
    )

    print(
        f"  Original SMILES:"
    )

    print(
        f"    {original_smiles}"
    )

    # --------------------------------------------------------
    # Sort largest -> smallest
    # --------------------------------------------------------

    fragments = sorted(
        fragments,
        key=lambda fragment:
            fragment.GetNumHeavyAtoms(),
        reverse=True,
    )

    print(
        "  Fragments:"
    )

    for i, fragment in enumerate(
        fragments,
        start=1,
    ):

        fragment_smiles = Chem.MolToSmiles(
            fragment,
            canonical=True,
        )

        print(
            f"    {i}. "
            f"{fragment_smiles} "
            f"("
            f"{fragment.GetNumHeavyAtoms()} "
            f"heavy atoms)"
        )

    if not KEEP_LARGEST_FRAGMENT:

        raise ValueError(
            f"{tag} contains multiple fragments."
        )

    # --------------------------------------------------------
    # Keep largest
    # --------------------------------------------------------

    main_fragment = fragments[0]

    removed = fragments[1:]

    removed_smiles = [
        Chem.MolToSmiles(
            fragment,
            canonical=True,
        )
        for fragment in removed
    ]

    docking_smiles = Chem.MolToSmiles(
        main_fragment,
        canonical=True,
    )

    print()
    print(
        f"  Keeping largest fragment:"
    )

    print(
        f"    {docking_smiles}"
    )

    print(
        f"  Removed fragments:"
    )

    for fragment_smiles in removed_smiles:

        print(
            f"    {fragment_smiles}"
        )

    return (
        main_fragment,
        docking_smiles,
        fragment_count,
        ".".join(
            removed_smiles
        ),
    )


# ============================================================
# LIGAND PREPARATION
# ============================================================

def prepare_ligand(
    smiles,
    out_dir: Path,
    tag: str,
    mk_prepare_ligand: str,
):

    result_info = {
        "original_smiles": "",
        "docking_smiles": "",
        "fragment_count": np.nan,
        "removed_fragments": "",
        "ligand_status": "FAILED",
    }

    # --------------------------------------------------------
    # Validate SMILES
    # --------------------------------------------------------

    if pd.isna(smiles):

        print(
            f"  [skip] Missing SMILES for {tag}"
        )

        result_info["ligand_status"] = (
            "MISSING_SMILES"
        )

        return None, result_info

    original_smiles = str(
        smiles
    ).strip()

    result_info[
        "original_smiles"
    ] = original_smiles

    if not original_smiles:

        print(
            f"  [skip] Empty SMILES for {tag}"
        )

        result_info["ligand_status"] = (
            "EMPTY_SMILES"
        )

        return None, result_info

    # --------------------------------------------------------
    # RDKit parsing
    # --------------------------------------------------------

    mol = Chem.MolFromSmiles(
        original_smiles
    )

    if mol is None:

        print(
            f"  [skip] Invalid SMILES for {tag}: "
            f"{original_smiles}"
        )

        result_info["ligand_status"] = (
            "INVALID_SMILES"
        )

        return None, result_info

    # --------------------------------------------------------
    # Handle fragments
    # --------------------------------------------------------

    try:

        (
            mol,
            docking_smiles,
            fragment_count,
            removed_fragments,
        ) = process_ligand_fragments(
            mol,
            original_smiles,
            tag,
        )

    except Exception as exc:

        print(
            f"  [skip] Fragment processing failed "
            f"for {tag}: {exc}"
        )

        result_info[
            "ligand_status"
        ] = "FRAGMENT_ERROR"

        return None, result_info

    result_info[
        "docking_smiles"
    ] = docking_smiles

    result_info[
        "fragment_count"
    ] = fragment_count

    result_info[
        "removed_fragments"
    ] = removed_fragments

    # --------------------------------------------------------
    # Add hydrogens
    # --------------------------------------------------------

    try:

        mol = Chem.AddHs(
            mol
        )

    except Exception as exc:

        print(
            f"  [skip] Could not add hydrogens "
            f"for {tag}: {exc}"
        )

        result_info[
            "ligand_status"
        ] = "HYDROGEN_ERROR"

        return None, result_info

    # --------------------------------------------------------
    # 3D embedding
    # --------------------------------------------------------

    params = AllChem.ETKDGv3()

    params.randomSeed = 42

    status = AllChem.EmbedMolecule(
        mol,
        params,
    )

    if status != 0:

        print(
            f"  [skip] 3D embedding failed "
            f"for {tag}"
        )

        result_info[
            "ligand_status"
        ] = "EMBEDDING_FAILED"

        return None, result_info

    # --------------------------------------------------------
    # Geometry optimization
    # --------------------------------------------------------

    optimization_success = False

    # --------------------------------------------------------
    # MMFF
    # --------------------------------------------------------

    try:

        mmff_status = (
            AllChem.MMFFOptimizeMolecule(
                mol
            )
        )

        if mmff_status >= 0:

            optimization_success = True

        else:

            print(
                f"  MMFF optimization failed "
                f"for {tag}; trying UFF."
            )

    except Exception:

        print(
            f"  MMFF optimization failed "
            f"for {tag}; trying UFF."
        )

    # --------------------------------------------------------
    # UFF fallback
    # --------------------------------------------------------

    if not optimization_success:

        try:

            uff_status = (
                AllChem.UFFOptimizeMolecule(
                    mol
                )
            )

            if uff_status >= 0:

                optimization_success = True

        except Exception as exc:

            print(
                f"  [skip] UFF optimization failed "
                f"for {tag}: {exc}"
            )

            result_info[
                "ligand_status"
            ] = "OPTIMIZATION_FAILED"

            return None, result_info

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # SDF
    # --------------------------------------------------------

    sdf_path = (
        out_dir
        / f"{tag}.sdf"
    )

    try:

        writer = Chem.SDWriter(
            str(sdf_path)
        )

        writer.write(
            mol
        )

        writer.close()

    except Exception as exc:

        print(
            f"  [skip] Could not write SDF "
            f"for {tag}: {exc}"
        )

        result_info[
            "ligand_status"
        ] = "SDF_WRITE_FAILED"

        return None, result_info

    # --------------------------------------------------------
    # PDBQT
    # --------------------------------------------------------

    pdbqt_path = (
        out_dir
        / f"{tag}.pdbqt"
    )

    # --------------------------------------------------------
    # Reuse existing PDBQT
    # --------------------------------------------------------

    if pdbqt_path.exists():

        print(
            f"  Existing PDBQT found:"
        )

        print(
            f"    {pdbqt_path}"
        )

        result_info[
            "ligand_status"
        ] = "SUCCESS_REUSED"

        return (
            pdbqt_path,
            result_info,
        )

    # --------------------------------------------------------
    # Meeko
    # --------------------------------------------------------

    cmd = [

        mk_prepare_ligand,

        "-i",
        str(sdf_path),

        "-o",
        str(pdbqt_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        print()
        print(
            f"  [skip] Meeko ligand preparation "
            f"failed for {tag}"
        )

        if result.stdout.strip():

            print(
                "  stdout:"
            )

            print(
                result.stdout.strip()
            )

        if result.stderr.strip():

            print(
                "  stderr:"
            )

            print(
                result.stderr.strip()
            )

        result_info[
            "ligand_status"
        ] = "MEEKO_FAILED"

        return None, result_info

    # --------------------------------------------------------
    # Validate PDBQT
    # --------------------------------------------------------

    if not pdbqt_path.exists():

        print(
            f"  [skip] Meeko completed but "
            f"PDBQT was not created:"
        )

        print(
            f"    {pdbqt_path}"
        )

        result_info[
            "ligand_status"
        ] = "PDBQT_NOT_CREATED"

        return None, result_info

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    result_info[
        "ligand_status"
    ] = "SUCCESS"

    print(
        f"  Ligand preparation successful:"
    )

    print(
        f"    SDF   = {sdf_path}"
    )

    print(
        f"    PDBQT = {pdbqt_path}"
    )

    return (
        pdbqt_path,
        result_info,
    )


# ============================================================
# VINA DOCKING
# ============================================================

def run_vina(
    vina_executable: str,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    config_txt: Path,
    out_pdbqt: Path,
) -> float | None:

    # --------------------------------------------------------
    # Reuse existing Vina output
    # --------------------------------------------------------

    if out_pdbqt.exists():

        affinity = parse_vina_affinity(
            out_pdbqt
        )

        if affinity is not None:

            print(
                "  Existing Vina result found."
            )

            return affinity

    # --------------------------------------------------------
    # Vina command
    # --------------------------------------------------------

    cmd = [

        vina_executable,

        "--receptor",
        str(receptor_pdbqt),

        "--ligand",
        str(ligand_pdbqt),

        "--config",
        str(config_txt),

        "--exhaustiveness",
        str(VINA_EXHAUSTIVENESS),

        "--out",
        str(out_pdbqt),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    # --------------------------------------------------------
    # Failure
    # --------------------------------------------------------

    if result.returncode != 0:

        print()
        print(
            "  [FAIL] Vina failed."
        )

        if result.stdout.strip():

            print(
                "  stdout:"
            )

            print(
                result.stdout.strip()[:3000]
            )

        if result.stderr.strip():

            print(
                "  stderr:"
            )

            print(
                result.stderr.strip()[:3000]
            )

        return None

    # --------------------------------------------------------
    # Check output
    # --------------------------------------------------------

    if not out_pdbqt.exists():

        print(
            "  [FAIL] Vina completed but "
            "output PDBQT was not created."
        )

        return None

    # --------------------------------------------------------
    # Parse affinity
    # --------------------------------------------------------

    affinity = parse_vina_affinity(
        out_pdbqt
    )

    if affinity is None:

        print(
            "  [FAIL] Could not parse "
            "Vina affinity."
        )

        return None

    return affinity


# ============================================================
# VINA AFFINITY PARSER
# ============================================================

def parse_vina_affinity(
    pdbqt_path: Path,
) -> float | None:

    try:

        text = pdbqt_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    except Exception as exc:

        print(
            f"  [FAIL] Could not read "
            f"{pdbqt_path}: {exc}"
        )

        return None

    # --------------------------------------------------------
    # Primary parser
    # --------------------------------------------------------

    match = AFFINITY_RE.search(
        text
    )

    if match:

        try:

            return float(
                match.group(1)
            )

        except ValueError:

            pass

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    for line in text.splitlines():

        if (
            "VINA RESULT"
            in line
        ):

            numbers = re.findall(
                r"-?\d+\.\d+",
                line,
            )

            if numbers:

                try:

                    return float(
                        numbers[0]
                    )

                except ValueError:

                    pass

    return None


# ============================================================
# SAVE FULL RESULTS
# ============================================================

def save_results(
    df: pd.DataFrame,
):

    FULL_RESULTS_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        FULL_RESULTS_CSV,
        index=False,
    )

    print()
    print(
        f"  Intermediate results saved:"
    )

    print(
        f"    {FULL_RESULTS_CSV}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "ROSSELLE VIRTUAL SCREENING - STAGE 2"
    )
    print("=" * 70)

    # ========================================================
    # DEPENDENCIES
    # ========================================================

    executables = check_dependencies()

    vina_executable = (
        executables["vina"]
    )

    mk_prepare_receptor = (
        executables[
            "mk_prepare_receptor"
        ]
    )

    mk_prepare_ligand = (
        executables[
            "mk_prepare_ligand"
        ]
    )

    # ========================================================
    # PATH CHECKS
    # ========================================================

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            "Input CSV not found:\n"
            f"{INPUT_CSV}"
        )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    WORKDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = pd.read_csv(
        INPUT_CSV
    )

    print()
    print(
        f"Loaded {len(df)} survivors."
    )

    if SMILES_COL not in df.columns:

        raise KeyError(
            f"Required column '{SMILES_COL}' "
            "not found.\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    # ========================================================
    # PREPARE RECEPTORS
    # ========================================================

    receptors = {}

    for label, pdb_id in TARGETS.items():

        try:

            receptors[label] = prepare_receptor(
                pdb_id=pdb_id,
                target_label=label,
                target_dir=(
                    WORKDIR
                    / label
                ),
                mk_prepare_receptor=(
                    mk_prepare_receptor
                ),
            )

        except Exception as exc:

            print()
            print(
                "=" * 70
            )

            print(
                f"RECEPTOR PREPARATION FAILED: "
                f"{label.upper()}"
            )

            print(
                "=" * 70
            )

            print(
                str(exc)
            )

            print()
            print(
                "The docking run cannot continue "
                "without this receptor."
            )

            raise

    # ========================================================
    # LIGAND DIRECTORY
    # ========================================================

    ligands_dir = (
        WORKDIR
        / "ligands"
    )

    ligands_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # RESULT COLUMNS
    # ========================================================

    df["original_smiles"] = (
        df[SMILES_COL]
        .astype("string")
    )

    df["docking_smiles"] = ""

    df["fragment_count"] = np.nan

    df["removed_fragments"] = ""

    df["ligand_status"] = ""

    for label in TARGETS:

        df[
            f"{label}_docking_affinity"
        ] = np.nan

    # ========================================================
    # DOCKING
    # ========================================================

    total = len(df)

    successful_ligands = 0

    failed_ligands = 0

    successful_docking_runs = 0

    failed_docking_runs = 0

    # --------------------------------------------------------
    # Iterate compounds
    # --------------------------------------------------------

    for counter, (idx, row) in enumerate(
        df.iterrows(),
        start=1,
    ):

        tag = (
            f"lig_{idx}"
        )

        print()
        print("=" * 70)

        print(
            f"LIGAND {counter}/{total}"
        )

        print(
            f"Row index: {idx}"
        )

        print("=" * 70)

        smiles = row[
            SMILES_COL
        ]

        # ----------------------------------------------------
        # Prepare ligand
        # ----------------------------------------------------

        try:

            (
                ligand_pdbqt,
                ligand_info,
            ) = prepare_ligand(
                smiles=smiles,
                out_dir=ligands_dir,
                tag=tag,
                mk_prepare_ligand=(
                    mk_prepare_ligand
                ),
            )

        except Exception as exc:

            print()
            print(
                f"  [ERROR] Unexpected ligand "
                f"preparation error:"
            )

            print(
                f"    {exc}"
            )

            ligand_pdbqt = None

            ligand_info = {
                "original_smiles": (
                    str(smiles)
                    if not pd.isna(smiles)
                    else ""
                ),
                "docking_smiles": "",
                "fragment_count": np.nan,
                "removed_fragments": "",
                "ligand_status": (
                    "UNEXPECTED_ERROR"
                ),
            }

        # ----------------------------------------------------
        # Save ligand metadata
        # ----------------------------------------------------

        df.at[
            idx,
            "original_smiles"
        ] = ligand_info.get(
            "original_smiles",
            "",
        )

        df.at[
            idx,
            "docking_smiles"
        ] = ligand_info.get(
            "docking_smiles",
            "",
        )

        df.at[
            idx,
            "fragment_count"
        ] = ligand_info.get(
            "fragment_count",
            np.nan,
        )

        df.at[
            idx,
            "removed_fragments"
        ] = ligand_info.get(
            "removed_fragments",
            "",
        )

        df.at[
            idx,
            "ligand_status"
        ] = ligand_info.get(
            "ligand_status",
            "FAILED",
        )

        # ----------------------------------------------------
        # Ligand failed
        # ----------------------------------------------------

        if ligand_pdbqt is None:

            failed_ligands += 1

            print(
                f"  Ligand {tag} skipped."
            )

            # Save immediately.
            save_results(
                df
            )

            continue

        successful_ligands += 1

        # ----------------------------------------------------
        # Dock against every target
        # ----------------------------------------------------

        for label, receptor in (
            receptors.items()
        ):

            print()
            print(
                f"  Docking {tag} -> "
                f"{label.upper()}"
            )

            out_pdbqt = (
                ligands_dir
                / f"{tag}_{label}_out.pdbqt"
            )

            affinity = run_vina(
                vina_executable=(
                    vina_executable
                ),
                receptor_pdbqt=(
                    receptor[
                        "receptor_pdbqt"
                    ]
                ),
                ligand_pdbqt=(
                    ligand_pdbqt
                ),
                config_txt=(
                    receptor[
                        "config_txt"
                    ]
                ),
                out_pdbqt=(
                    out_pdbqt
                ),
            )

            df.at[
                idx,
                f"{label}_docking_affinity"
            ] = affinity

            if affinity is None:

                failed_docking_runs += 1

                print(
                    f"    {label.upper():8s}: "
                    "FAILED"
                )

            else:

                successful_docking_runs += 1

                print(
                    f"    {label.upper():8s}: "
                    f"{affinity:.3f} kcal/mol"
                )

        # ----------------------------------------------------
        # Save after every ligand.
        #
        # This means an interruption will not destroy
        # everything completed previously.
        # ----------------------------------------------------

        save_results(
            df
        )

    # ========================================================
    # RANKING
    # ========================================================

    print()
    print("=" * 70)
    print("CALCULATING FINAL RANKING")
    print("=" * 70)

    affinity_cols = [
        f"{label}_docking_affinity"
        for label in TARGETS
    ]

    # --------------------------------------------------------
    # Number of successful targets
    # --------------------------------------------------------

    df[
        "successful_docking_targets"
    ] = (
        df[
            affinity_cols
        ]
        .notna()
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # Mean docking affinity
    #
    # Pandas ignores NaN values.
    #
    # Example:
    #
    # EGFR  = -8.2
    # HER2  = -7.4
    # VEGFR = NaN
    #
    # mean = (-8.2 - 7.4) / 2
    # --------------------------------------------------------

    df[
        "mean_docking_affinity"
    ] = (
        df[
            affinity_cols
        ]
        .mean(
            axis=1
        )
    )

    # --------------------------------------------------------
    # Minimum affinity
    #
    # Useful for seeing the strongest individual target.
    # --------------------------------------------------------

    df[
        "best_docking_affinity"
    ] = (
        df[
            affinity_cols
        ]
        .min(
            axis=1
        )
    )

    # --------------------------------------------------------
    # Require at least one successful docking.
    # --------------------------------------------------------

    valid = df[
        df[
            "successful_docking_targets"
        ] > 0
    ].copy()

    # --------------------------------------------------------
    # Rank:
    #
    # 1. Mean affinity ascending
    #    more negative = better
    #
    # 2. Number of successful targets descending
    #
    # --------------------------------------------------------

    final = (
        valid
        .sort_values(
            by=[
                "mean_docking_affinity",
                "successful_docking_targets",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .head(
            FINAL_TOP_N
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # SAVE FINAL
    # ========================================================

    final.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # Also save complete results one final time.
    save_results(
        df
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("DOCKING COMPLETE")
    print("=" * 70)

    print(
        f"Input compounds:          {len(df)}"
    )

    print(
        f"Ligands prepared:         "
        f"{successful_ligands}"
    )

    print(
        f"Ligands failed:           "
        f"{failed_ligands}"
    )

    print(
        f"Successful docking runs:  "
        f"{successful_docking_runs}"
    )

    print(
        f"Failed docking runs:      "
        f"{failed_docking_runs}"
    )

    print(
        f"Final top N:              "
        f"{len(final)}"
    )

    print()
    print(
        "Complete results:"
    )

    print(
        f"  {FULL_RESULTS_CSV}"
    )

    print()
    print(
        "Final results:"
    )

    print(
        f"  {OUTPUT_CSV}"
    )

    # ========================================================
    # TOP COMPOUNDS
    # ========================================================

    if len(final) > 0:

        print()
        print("=" * 70)
        print("TOP COMPOUNDS")
        print("=" * 70)

        display_cols = [
            col
            for col in [
                "chembl_id",
                "compound_name",
                SMILES_COL,
                "docking_smiles",
                "fragment_count",
                "removed_fragments",
                "egfr_docking_affinity",
                "her2_docking_affinity",
                "vegfr2_docking_affinity",
                "mean_docking_affinity",
                "best_docking_affinity",
                "successful_docking_targets",
                "ligand_status",
            ]
            if col in final.columns
        ]

        print(
            final[
                display_cols
            ].to_string(
                index=False
            )
        )

    else:

        print()
        print(
            "WARNING: No compounds successfully "
            "completed docking."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Docking interrupted by user."
        )

    except Exception as exc:

        print()
        print("=" * 70)
        print("FATAL ERROR")
        print("=" * 70)

        print(
            str(exc)
        )

        raise
