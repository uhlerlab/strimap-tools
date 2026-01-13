import torch
import torch.nn as nn
from typing import List
import numpy as np
from sklearn.preprocessing import StandardScaler

class PhysicochemicalEncoder(nn.Module):
    """Amino Acid Physicochemical Property Encoder (AAindex版本, 向量化优化版)"""
    
    def __init__(self, device, use_aaindex=True, selected_features=None):
        super().__init__()
        self.device = device
        self.use_aaindex = use_aaindex

        # initialize amino acid properties
        if use_aaindex:
            self.aa_properties, self.feature_names = self._load_aaindex_features(selected_features)
            self.n_features = len(list(self.aa_properties['A'].values()))
            print(f"✓ Loaded {self.n_features} AAindex features")
        else:
            self.aa_properties = self._get_basic_properties()
            self.n_features = 5
            print(f"✓ Using {self.n_features} basic features")
        
        # Fit scaler
        self.scaler = self._fit_scaler()

        # ======================== Preprocessing ======================== #
        # 1. Construct lookup table
        aa_list = list(self.aa_properties.keys())
        aa_list.sort()  # Ensure stable order
        self.aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}
        self.pad_idx = len(self.aa_to_idx)  # padding index

        aa_feature_table = []
        for aa in aa_list:
            feats = self._get_aa_features(aa)
            aa_feature_table.append(feats)
        aa_feature_table.append([0.0] * self.n_features)  # padding vector
        self.aa_feature_table = torch.tensor(
            np.array(aa_feature_table),
            dtype=torch.float32
        ).to(self.device)  # [n_aa+1, n_feat]

        # 2. Pre-store standardization parameters as GPU tensors
        self.mean_tensor = torch.tensor(self.scaler.mean_, dtype=torch.float32, device=self.device)
        self.scale_tensor = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=self.device)

    def _load_aaindex_features(self, selected_features=None):
        try:
            from aa_properties_aaindex import AA_PROPERTIES_AAINDEX, FEATURE_DESCRIPTIONS
            if selected_features is not None:
                filtered_props = {}
                for aa, props in AA_PROPERTIES_AAINDEX.items():
                    filtered_props[aa] = {k: v for k, v in props.items() if k in selected_features}
                return filtered_props, selected_features
            else:
                feature_names = list(AA_PROPERTIES_AAINDEX['A'].keys())
                return AA_PROPERTIES_AAINDEX, feature_names
        except ImportError:
            print("⚠ Warning: aa_properties_aaindex.py not found!")
            return self._get_basic_properties(), ['hydro', 'charge', 'volume', 'flex', 'aroma']

    def _get_basic_properties(self):
        return {
            'A': [1.8, 0.0, 88.6, 0.36, 0.0],
            'C': [2.5, 0.0, 108.5, 0.35, 0.0],
            'D': [-3.5, -1.0, 111.1, 0.51, 0.0],
            'E': [-3.5, -1.0, 138.4, 0.50, 0.0],
            'F': [2.8, 0.0, 189.9, 0.31, 1.0],
            'G': [-0.4, 0.0, 60.1, 0.54, 0.0],
            'H': [-3.2, 0.5, 153.2, 0.32, 0.5],
            'I': [4.5, 0.0, 166.7, 0.46, 0.0],
            'K': [-3.9, 1.0, 168.6, 0.47, 0.0],
            'L': [3.8, 0.0, 166.7, 0.37, 0.0],
            'M': [1.9, 0.0, 162.9, 0.30, 0.0],
            'N': [-3.5, 0.0, 114.1, 0.46, 0.0],
            'P': [-1.6, 0.0, 112.7, 0.51, 0.0],
            'Q': [-3.5, 0.0, 143.8, 0.49, 0.0],
            'R': [-4.5, 1.0, 173.4, 0.53, 0.0],
            'S': [-0.8, 0.0, 89.0, 0.51, 0.0],
            'T': [-0.7, 0.0, 116.1, 0.44, 0.0],
            'V': [4.2, 0.0, 140.0, 0.39, 0.0],
            'W': [-0.9, 0.0, 227.8, 0.31, 1.0],
            'Y': [-1.3, 0.0, 193.6, 0.42, 1.0],
            'X': [0.0, 0.0, 120.0, 0.40, 0.0],
        }

    def _fit_scaler(self):
        all_features = []
        for aa in 'ARNDCQEGHILKMFPSTWYV':
            if isinstance(self.aa_properties[aa], dict):
                features = list(self.aa_properties[aa].values())
            else:
                features = self.aa_properties[aa]
            all_features.append(features)
        all_features = np.array(all_features)
        scaler = StandardScaler()
        scaler.fit(all_features)
        return scaler

    def _get_aa_features(self, aa: str):
        aa = aa.upper()
        if aa not in self.aa_properties:
            aa = 'X'
        if isinstance(self.aa_properties[aa], dict):
            return list(self.aa_properties[aa].values())
        else:
            return self.aa_properties[aa]

    def forward(self, sequences: List[str]) -> torch.Tensor:
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)

        # 1) encode sequences to indices with padding
        idx_batch = np.full((batch_size, max_len), self.pad_idx, dtype=np.int64)
        for i, seq in enumerate(sequences):
            idx_seq = [self.aa_to_idx.get(aa.upper(), self.pad_idx) for aa in seq]
            idx_batch[i, :len(idx_seq)] = idx_seq

        idx_tensor = torch.tensor(idx_batch, dtype=torch.long, device=self.device)  # [B, L]

        # 2) lookup properties
        props = self.aa_feature_table[idx_tensor]  # [B, L, n_feat]

        props = (props - self.mean_tensor) / self.scale_tensor

        return props
