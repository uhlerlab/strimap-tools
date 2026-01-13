# StriMap: Discovering TCR-HLA-Epitope interactions with Deep Learning

**strimap-tools** is a package for analysis of peptide-HLA presentation and TCR specificity. It is designed to help researchers understand the interactions between T cell receptors (TCRs) and peptides presented by human leukocyte antigen (HLA) molecules, which play a crucial role in the immune response. 

To facilitate use by biologists and help bridge the gap between the machine-learning and immunology communities, we developed an accessible web portal [www.strimap.com](https://www.strimap.com) that enables users to run predictions and train models on their own data.

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

## Key Features

- **Peptide representation learning with HLA context**  
  Learns latent embeddings of peptides while incorporating HLA background, enabling biologically informed modeling.

- **TCR specificity prediction using full receptor sequences**  
  Supports comprehensive modeling of TCR recognition by leveraging both α and β chain CDR regions.

- **Visualization of critical TCR residues**  
  Highlights key amino acid positions in TCRs that contribute to antigen recognition, aiding biological interpretation.

- **Discovery of disease-associated epitopes**  
  Identifies novel peptides potentially involved in disease by integrating small-scale disease-specific data with large-scale public datasets.

## Citation
If you use **strimap-tools** in your research, please cite the following paper: