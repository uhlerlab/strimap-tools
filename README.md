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

📓 **[phla_predictor.ipynb](phla_predictor.ipynb)**

This notebook demonstrates how to:
- Train a peptide–HLA (pHLA) presentation predictor with **5-fold cross-validation**
- Load a trained checkpoint and run **prediction/inference** (and optional evaluation if labels are available)

**Expected input (CSV format):**
- `peptide`: peptide amino acid sequence (e.g., `GILGFVFTL`)
- `HLA`: HLA allele (e.g., `HLA-A*02:01`)
- `label`: binary label indicating presentation status, optional for inference-only prediction (required for evaluation metrics)
  - `1` = presented (positive)  
  - `0` = not presented (negative)

## Train and predict TCR-pHLA specificity with StriMap

## Citation
If you use **strimap-tools** in your research, please cite the following paper: