# Quantum-Enhanced Discovery of Anticancer Phytochemicals from *Hibiscus sabdariffa*
## An In Silico Machine Learning Approach

This repository accompanies the research study *“Quantum-Enhanced Discovery of Anticancer Phytochemicals from Hibiscus sabdariffa: An In Silico Machine Learning Approach.”* It contains the datasets, model experiments, and analysis outputs supporting the investigation. The focus of the study is the computational research process and its findings—not a standalone software package.

## Research overview

The study investigates whether machine-learning methods, including hybrid quantum-classical models, can help prioritize *Hibiscus sabdariffa* (roselle)-associated phytochemicals for follow-up as potential anticancer candidates. Predicted activity and developability-related properties are considered together, with molecular docking used as a secondary, structure-based prioritization step.

The work is hypothesis-generating. Model predictions and docking scores do not demonstrate anticancer efficacy, safety, or clinical suitability; experimental validation is required.

## Research questions and scope

The computational investigation centers on:

- Predicting compound activity against three cancer-related protein targets: EGFR, HER2, and VEGFR2.
- Comparing classical Random Forest and graph neural network approaches with hybrid quantum machine-learning experiments.
- Assessing predicted ADMET-related endpoints: CYP3A4 activity, aqueous solubility, hERG liability, and mutagenicity.
- Combining predicted activity, safety-related properties, and drug-likeness to prioritize compounds for docking.
- Exploring the association between machine-learning screening scores and docking affinity.

## Computational approach

1. **Assemble reference data.** Target bioactivity records are collected from ChEMBL; ADMET training data are drawn from ChEMBL and the Therapeutics Data Commons (TDC). Compound records are validated and represented by their SMILES structures.
2. **Represent molecules.** RDKit descriptors and Morgan fingerprints provide tabular features. Molecular graphs are also used for graph-neural-network experiments.
3. **Train and evaluate models.** The notebooks contain Random Forest, graph neural network, and hybrid quantum-model experiments for bioactivity and ADMET endpoints.
4. **Prioritize the roselle compound set.** Predicted activity, ADMET-related values, and QED drug-likeness are used in a multi-objective screen. The current screening utility applies a mutagenicity filter and Pareto ranking, using a weighted score to order or adjust the candidate set.
5. **Dock shortlisted molecules.** AutoDock Vina is used to estimate binding affinity against EGFR (PDB 1M17), HER2 (PDB 3PP0), and VEGFR2 (PDB 4ASD).
6. **Analyze results.** Descriptive statistics and correlation analysis summarize the compound collection and the relationship between screening scores and docking affinities.

## Findings represented in this repository

The included result files are a snapshot of the current analysis:

- `data/final_dataset/final_roselle_dataset.csv` contains 1,284 compound records used in the screening dataset.
- `data/final_dataset/virtual_screening_survivors.csv` contains 77 virtual-screening candidates.
- `data/final_dataset/final_docking_survivors.csv` contains 39 compounds with final docking results.
- `data/rq4_correlation_result.csv` reports an exploratory association analysis on 38 paired observations. It records a negative Spearman correlation between the composite screening score and mean docking affinity (rho = -0.3816, p = 0.0181; Kendall tau = -0.2831, p = 0.0124). This is an association within a small, preselected set—not independent validation of predictive performance.

These counts and statistics describe the checked-in outputs; rerunning or refreshing the workflow may produce different results.

## What is included

```text
data/
  final_dataset/         Curated compound table and screening/docking outputs
  roselle_dataset/       Roselle compound features and molecular-graph data
  training_dataset/      Target-specific bioactivity and ADMET training data
  plots/                 Descriptive and endpoint plots
models/
  random_forest/         Classical baseline experiments
  graph_neural_network/  Molecular graph model experiments
  hybrid_quantum_models/ Hybrid quantum-model experiments
  best_models/           Selected model notebooks and inference metadata
docking_workdir/         Target docking-box files and working data
utils/                   Data preparation, featurization, screening, and analysis utilities
```

The notebooks and output tables are the primary research artifacts. The utilities support dataset preparation, molecular featurization, candidate screening, docking, and statistical analysis.

## Interpretation and limitations

- This is an in-silico prioritization study; predicted activity and docking affinity are not experimental measurements of anticancer activity.
- Docking scores are approximate and depend on receptor structures, preparation choices, and scoring assumptions.
- Screening and correlation results are based on a limited, selected candidate set; they should not be interpreted as proof of general model performance.
- Public training datasets can differ in assay conditions, compound coverage, and label quality. Results depend on the data and model configurations used in the included notebooks.
- Any candidate identified here requires experimental testing, including efficacy, selectivity, and safety assessment, before biological or therapeutic claims can be made.

## Using this companion repository

Treat the checked-in datasets and notebooks as research materials for examining the study workflow and its reported outputs. Reproducing the computational experiments may require the scientific Python and cheminformatics libraries, quantum machine-learning frameworks, and docking tools used by the individual notebooks and utilities. Exact reproduction also depends on matching their data sources, software versions, random seeds, and model settings.

## Citation

If you use these research materials, cite the associated study:

> *Quantum-Enhanced Discovery of Anticancer Phytochemicals from Hibiscus sabdariffa: An In Silico Machine Learning Approach.*

Add the authors, publication venue, year, and persistent identifier here when those details are available.