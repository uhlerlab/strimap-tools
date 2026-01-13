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
### Train a pHLA Predictor (5-fold Cross-validation)

**Input CSV:** `train_set.csv`

**Required columns:**
- `peptide`: peptide amino acid sequence  
- `HLA`: HLA allele (e.g., `HLA-A*02:01`)  
- `label`: binary label (`1` = positive, `0` = negative)  

```python
from main import StriMap_pHLA, load_train_data
import pandas as pd
from sklearn.model_selection import train_test_split

# Load full dataset (already contains both positive and negative samples)
df_train = pd.read_csv("train_set.csv")

# Create 5 folds (each fold is a train/val split)
num_fold = 5
df_train_folds, df_val_folds = [], []
for i in range(num_fold):
    df_train_fold, df_val_fold = train_test_split(
        df_train,
        test_size=0.1,
        random_state=i,
        stratify=df_train["label"],
    )
    df_train_folds.append(df_train_fold.reset_index(drop=True))
    df_val_folds.append(df_val_fold.reset_index(drop=True))

# Standardize HLA fields and map alleles
df_train_folds, df_val_folds = load_train_data(
    df_train_list=df_train_folds,
    df_val_list=df_val_folds,
    hla_dict_path="HLA_dict.npy",
)

# Combine all folds for embedding preparation (recommended)
df_all = pd.concat([*df_train_folds, *df_val_folds], axis=0).reset_index(drop=True)

# Initialize StriMap
strimap = StriMap_pHLA(
    device="cuda:0",  # or "cpu"
    model_save_path="model_params/phla/best_model.pt",
    cache_dir="phla_cache",
    cache_save=False,
)

# Prepare embeddings (run once; subsequent runs are faster if cached)
strimap.prepare_embeddings(df_all, force_recompute=False)

# K-fold training
all_history = strimap.train_kfold(
    train_folds=[(df_train, df_val) for df_train, df_val in zip(df_train_folds, df_val_folds)],
    epochs=100,
    num_workers=4,
)
```

### Prediction

The trained StriMap pHLA model can be used to generate predictions on new peptide–HLA pairs.

```python
from main import StriMap_pHLA, load_test_data
import pandas as pd

# Load test data
df_test = pd.read_csv("test.csv")

# Standardize HLA fields and map alleles
df_test = load_test_data(
    df_test=df_test,
    hla_dict_path="HLA_dict.npy",
)

# Initialize StriMap with a trained checkpoint
strimap = StriMap_pHLA(
    device="cuda:0",  # or "cpu"
    model_save_path=f"model_params/phla/best_model.pt",
    cache_dir="phla_cache",
)

# Prepare embeddings (cached for faster inference)
strimap.prepare_embeddings(df_test, force_recompute=False,)

# Run prediction / evaluation
y_prob_test, _ = strimap.predict(
    df_test,
    use_kfold=True,
    num_folds=5,
)
df_test["predicted_probability"] = y_prob_test

## Citation
If you use **strimap-tools** in your research, please cite the following paper: