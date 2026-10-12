# Accompany source directory for research "Quantum-Enhanced Discovery of Anticancer Phytochemicals from Hibiscus sabdariffa: An In Silico Machine Learning Approach"

A Python-based molecular discovery and screening workflow for identifying promising Roselle analog compounds for therapeutic activity and safety assessment. The repository combines dataset acquisition, chemical feature engineering, model training, virtual filtering, molecular docking, and statistical analysis to prioritize compounds across multiple biological targets.

This project is organized around a research pipeline that:

- downloads bioactivity and ADMET datasets from ChEMBL and TDC,
- validates compound records and normalizes target labels,
- computes RDKit molecular descriptors and Morgan fingerprints,
- builds predictive models for EGFR, HER2, VEGFR2, and ADMET endpoints,
- filters candidate compounds through a multi-stage virtual screening process,
- docks the shortlisted compounds against target proteins,
- summarizes data quality and model performance.

## Project goals

The source code supports a drug-discovery style workflow for Roselle-derived compounds, with emphasis on:

- target activity prediction for EGFR, HER2, and VEGFR2,
- ADMET risk prediction such as CYP3A4 inhibition, hERG liability, mutagenicity, and solubility,
- ranking compounds by biological relevance and safety,
- prioritizing a small subset for structure-based docking and further study.

## Repository structure

```text
final_roselle/
├── data/
│   ├── final_dataset/                  # final screening and ranking outputs
│   ├── plots/                          # generated plots for descriptive analysis
│   ├── roselle_dataset/                # Roselle molecule datasets and features
│   ├── training_dataset/               # training-ready feature tables and metadata
│   ├── ames.tab                        # ADMET dataset raw data
│   ├── herg.tab                        # hERG raw data
│   ├── solubility_aqsoldb.tab          # solubility raw data
│   ├── columns.json                    # dataset column metadata
│   ├── rdkit_feature_cache.json        # cached RDKit descriptor fingerprints
│   └── rq4_correlation_result.csv       # final correlation output
├── docking_workdir/
│   ├── egfr/                           # EGFR protein and docking box files
│   ├── her2/                           # HER2 protein and docking box files
│   ├── vegfr2/                        # VEGFR2 protein and docking box files
│   └── ligands/                       # candidate ligands for docking
├── models/
│   ├── best_models/                    # saved/selected model artifacts
│   ├── graph_neural_network/           # GNN-based model notebooks
│   ├── hybrid_quantum_models/          # hybrid quantum ML notebooks
│   ├── random_forest/                  # random forest models and notebooks
│   └── sample.ipynb                    # sample notebook
├── utils/
│   ├── admet.py                       # ADMET dataset construction pipeline
│   ├── bioactivity.py                 # bioactivity dataset construction pipeline
│   ├── bioactivity_dataset_validator.py
│   ├── admet_dataset_validator.py
│   ├── column_helper.py
│   ├── correlation.py                 # correlation and normality testing
│   ├── data_manager.py                # generic dataset loader abstraction
│   ├── descriptives.py               # descriptive statistics for Roselle dataset
│   ├── docking.py                    # receptor preparation + AutoDock Vina workflow
│   ├── fetch_admet_dataset.py        # TDC dataset fetcher
│   ├── fetch_bioactivity_dataset.py  # ChEMBL activity fetcher
│   ├── graph_data_converter.py
│   ├── plotter.py                    # metric plots and visualizations
│   ├── rdkit_computation.py          # descriptor + fingerprint generation
│   ├── virtual_screening.py          # Pareto ranking and survivor selection
│   └── __init__.py
├── MCS-I.1-EDITED.docx
├── MCS-I.1-Revised.docx
├── Programmer_s_Report_revised.docx
├── Rona-Mar-Molecular-Docking.docx
├── Phase 3 Process Flow.docx
├── RESULTS-NEW-BATAAN-NATIONAL-HIGH-SCHOOL.docx-1.pdf
└── README.md
```

## Core source modules

### Data collection and validation

- `utils/fetch_bioactivity_dataset.py`
  - Pulls ChEMBL compound activity data for target proteins.
  - Filters by assay confidence, standard assay type, and daily rate limits.
  - Saves raw CSV outputs and logs.

- `utils/fetch_admet_dataset.py`
  - Loads TDC ADMET datasets such as Solubility, hERG, and mutagenicity.
  - Writes compact property-specific datasets for downstream training.

- `utils/bioactivity_dataset_validator.py` and `utils/admet_dataset_validator.py`
  - Clean and validate records before training.
  - Ensure SMILES, labels, and provenance columns are compatible.

### Chemical featurization

- `utils/rdkit_computation.py`
  - Uses RDKit to compute physicochemical descriptors and Morgan fingerprints.
  - Adds a cache so repeated calculations are faster and more reproducible.

- `utils/column_helper.py`
  - Handles metadata/feature column alignment and processing for datasets.

### Dataset assembly and access

- `utils/data_manager.py`
  - A lightweight loader that reads either tabular or graph datasets based on a dataset path and type.

- `utils/admet.py`
  - Builds ADMET training datasets by combining fetched data and RDKit features.

- `utils/bioactivity.py`
  - Builds bioactivity datasets for selected targets such as EGFR, HER2, and VEGFR2.

### Analysis and model assessment

- `utils/descriptives.py`
  - Produces descriptive statistics for physicochemical and bioactivity features.
  - Summarizes continuous and ADMET-probability distributions.

- `utils/correlation.py`
  - Checks the relationship between a composite model score and docking affinity.
  - Uses Shapiro-Wilk normality tests, Pearson/Spearman selection, and Kendall tau.

- `utils/plotter.py`
  - Generates evaluation visualizations such as ROC curves, PR curves, confusion matrices, and feature importance charts.

### Virtual screening and docking

- `utils/virtual_screening.py`
  - Loads the final Roselle dataset,
  - applies ADMET hard filters,
  - computes bioactivity and safety scores,
  - uses Pareto-optimal ranking to select a manageable set of candidates.

- `utils/docking.py`
  - Prepares target receptors from PDB structures,
  - defines docking boxes around the crystallographic ligand,
  - converts ligands to 3D structures and PDBQT,
  - uses AutoDock Vina to score candidate compounds,
  - saves ranked docking survivors.

## Data workflow

The code follows a typical machine-learning-for-drug-discovery pipeline:

1. Data acquisition
   - ChEMBL bioactivity data is collected per target.
   - TDC ADMET datasets are gathered for toxicity and pharmacokinetic endpoints.

2. Dataset validation and cleaning
   - Rows are filtered for valid SMILES and reliable assay records.
   - Labels are standardised for classification or regression tasks.

3. Feature generation
   - RDKit descriptors and Morgan fingerprints are created.
   - Feature caches reduce repeated computations across targets.

4. Model training
   - The repository contains notebooks for random forest, graph neural network, and hybrid quantum ML workflows.
   - Best-performing models are stored in `models/best_models/`.

5. Virtual screening
   - A candidate library is filtered by safety and then ranked by multi-objective optimisation.

6. Docking
   - Surviving compounds are evaluated against EGFR, HER2, and VEGFR2 receptors.
   - Docking scores refine the final shortlist.

7. Reporting
   - Descriptive statistics, plots, and correlation checks support interpretation of the screening results.

## Main execution scripts

These are the most important entry points found in this repository:

```bash
# Build bioactivity datasets
python utils/bioactivity.py

# Build ADMET datasets
python utils/admet.py

# Run virtual screening and candidate pruning
python utils/virtual_screening.py

# Run receptor preparation and docking
python utils/docking.py

# Correlation analysis between screening score and docking affinity
python utils/correlation.py

# Descriptive statistics for Roselle dataset
python utils/descriptives.py
```

## Notes on data outputs

The project writes many intermediate and final artifacts into `data/`:

- `data/training_dataset/...` contains per-target feature tables and raw fetched data.
- `data/final_dataset/...` contains the final candidate library, screened survivors, and docking outputs.
- `data/plots/...` contains generated figures and summary plots.

These files are the backbone of the research workflow and should be treated as the results layer of the source code.

## Dependencies

The code relies on a standard scientific Python stack plus chemistry tooling, including:

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib / seaborn
- RDKit
- PyTorch
- SciPy
- requests
- AutoDock Vina
- Meeko
- TDC (for dataset fetching)

This project is research-oriented rather than packaged as a formal Python package, so most scripts are meant to be executed from the project root.

## Recommended workflow for a new user

1. Create a Python environment with the dependencies above.
2. Run the data-fetching and feature-building scripts.
3. Open the notebooks under `models/` to train or evaluate models.
4. Run `utils/virtual_screening.py` to generate candidates.
5. Run `utils/docking.py` to evaluate the shortlisted compounds.
6. Use `utils/descriptives.py` and `utils/correlation.py` for reporting and validation.

## Summary

This repository is a complete molecular discovery pipeline for Roselle analog screening. It connects chemistry, machine learning, and structure-based docking into one project, with code structured for dataset generation, model experimentation, and candidate prioritization. The main intellectual work is not just model fitting, but the end-to-end workflow that turns raw molecular data into a ranked shortlist of promising compounds.
