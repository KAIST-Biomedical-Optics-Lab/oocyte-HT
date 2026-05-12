## Code organization

This repository contains the custom analysis code used for feature extraction, segmentation validation, dimensionality reduction, and statistical analysis of oocyte holotomography data.

### MATLAB analysis code

- `fig4_dice_ssim.m` performs segmentation validation for Fig. 4 by comparing predicted compartment masks with manual reference masks. It computes compartment-level Dice coefficients and 3D SSIM values for segmentation accuracy assessment.

- `fig7_umap.m` is the main MATLAB script for the Fig. 7 modality-comparison analysis, including UMAP visualization, clustering-quality evaluation, PERMANOVA-based analysis, ablation analysis, and figure generation.

- The following MATLAB files are auxiliary function files called by `fig7_umap.m`:
  - `cluster_validity_scores.m`: computes clustering validity metrics, including the Davies–Bouldin index and Calinski–Harabasz index.
  - `knn_mixing_score.m`: computes k-nearest-neighbor mixing scores.
  - `permanova1_raw.m`: performs one-way PERMANOVA.
  - `run_ablation_analysis.m`: performs leave-one-feature-out ablation analysis and summarizes feature-family contributions.

### Python analysis code

- `fig3_singleoocyte_analysis.py` extracts single-oocyte, compartment-level features from segmented HT volumes. The extracted features include morphology, RI/intensity statistics, dry-mass-related metrics, GLCM texture features, SPHARM-derived shape descriptors, and ZP thickness-related metrics.

### Python/Rust accelerated analysis code

The Fig. 6 statistical analysis includes both Python and Rust code:

- Python scripts are used for the main analysis workflow, data handling, feature extraction, and Excel export.
- Rust code is used to accelerate computationally intensive feature calculations, particularly GLCM-based texture feature computation.
- The Rust module is called from Python as a local extension module. If the Rust extension is not available, the corresponding Python implementation is used as a fallback, although with slower runtime.
