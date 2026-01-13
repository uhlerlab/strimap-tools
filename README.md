# StriMap: Discovering TCR-HLA-Epitope interactions with Deep Learning

**strimap-tools** is a package for analysis of peptide-HLA presentation and TCR specificity. It is designed to help researchers understand the interactions between T cell receptors (TCRs) and peptides presented by human leukocyte antigen (HLA) molecules, which play a crucial role in the immune response. 

To facilitate use by biologists and help bridge the gap between the machine-learning and immunology communities, we developed an accessible web portal [**www.strimap.com**](https://www.strimap.com) that enables predictions and finetune models on your own data.

If you prefer to run the package locally, please follow the instructions below to install **strimap-tools**.
## Installation
Create a conda environment. **strimap-tools** requires **Python 3.9** or later.
```bash
conda create -n strimap-env python=3.9
conda activate strimap-env
```
Download the source code from GitHub and install the package along with its dependencies.
```bash
git clone https://github.com/uhlerlab/strimap-tools.git
cd strimap-tools
pip install -r requirements.txt
```

## Usage
## Train and predict peptide–HLA presentation with StriMap

A complete, reproducible workflow for **training and prediction** is provided in:

📓 **[phla_predictor.ipynb](jupyter/phla_predictor.ipynb)**

This notebook demonstrates how to:
- Train a peptide–HLA (pHLA) presentation predictor with **5-fold cross-validation**
- Load a trained checkpoint and run **prediction/inference** (and optional evaluation if labels are available)

**Expected input (CSV format):**

| Column  | Description | Example | Note |
|------------|------------|---------|------|
| `peptide` | Peptide amino acid sequence | `GILGFVFTL` | Required |
| `HLA` | HLA allele | `HLA-A*02:01` | Required |
| `label` | Peptide–HLA presentation label (`0` or `1`) | `1` | Required for training/evaluation |

## Train and predict TCR-pHLA specificity with StriMap

A complete, reproducible workflow for **training and prediction of TCR–pHLA specificity**
(including data loading, HLA normalization, embedding preparation, cross-validation training,
and inference/evaluation on new data) is provided in:

📓 **[tcrphla_predictor.ipynb](jupyter/tcrphla_predictor.ipynb)**

This notebook demonstrates how to:
- Train a TCR–pHLA specificity predictor using cross-validation
- Load trained checkpoints and perform prediction/inference on new TCR–pHLA pairs

**Expected input (CSV format):**

| Column | Description | Example | Note |
|------------|------------|---------|------|
| `cdr3a` | Alpha chain CDR3 sequence | `CARRGAAGNKLTF` | Required |
| `cdr3b` | Beta chain CDR3 sequence | `CASSPSAGDYEQYF` | Required |
| `Va` | Alpha variable gene | `TRAV24*01` | Required |
| `Ja` | Alpha joining gene | `TRAJ17*01` | Required |
| `Vb` | Beta variable gene | `TRBV4-3*01` | Required |
| `Jb` | Beta joining gene | `TRBJ2-7*01` | Required |
| `peptide` | Target peptide sequence | `LLWNGPMAV` | Required |
| `HLA` | Target HLA allele | `HLA-A*02:01` | Required |
| `label` | TCR–pHLA binding label (`0` or `1`) | `1` | Required only for training/validation |


## Citation
If you use **strimap-tools** in your research, please cite the following paper: