# ============================================================================
# Main module for Structure-informed Peptide-HLA Binding Prediction Model
# Author: Kai Cao, PhD
# Date: 2026-01
# ============================================================================
# -*- coding: utf-8 -*-
import os
import warnings
from collections import Counter
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

from strimap.physicochemical import PhysicochemicalEncoder

from strimap.model import (
    negative_sampling_phla,
    ESM2Encoder,
    ESMFoldEncoder,
    PeptideHLABindingPredictor,
    PepHLA_Dataset,
    peptide_hla_collate_fn,
    TCRPeptideHLABindingPredictor,
    TCRPepHLA_Dataset,
    tcr_pep_hla_collate_fn,
    EarlyStopping
)

# ============================================================================
# StriMap for Structure-informed Peptide-HLA Binding Prediction Model
# ============================================================================
class StriMap_pHLA:
    def __init__(
        self,
        device: str = 'cuda:0',
        model_save_path: str = 'model_params/best_model_phla.pt',
        pep_dim: int = 256,
        hla_dim: int = 256,
        bilinear_dim: int = 256,
        loss_fn: str = 'focal',
        alpha: float = 0.5,
        gamma: float = 2.0,
        esm2_layer: int = 33,
        batch_size: int = 256,
        esmfold_cache_dir: str = "esm_cache",
        cache_dir: str = 'cache',
        cache_save: bool = True,
        seed: int = 1,
        pos_weights: Optional[float] = None,
        neg_ratio: Optional[float] = None,
    ):
        """
        Initialize StriMap model
    
        Args:
            device: Device for computation
            model_save_path: Path to save best model
            pep_dim: Peptide embedding dimension
            hla_dim: HLA embedding dimension
            bilinear_dim: Bilinear attention dimension
            loss_fn: Loss function ('bce' or 'focal')
            alpha: Alpha parameter for focal loss
            gamma: Gamma parameter for focal loss
            esm2_layer: ESM2 layer to extract features from
            batch_size: Batch size for embedding and training
            esmfold_cache_dir: Cache directory for ESMFold
            cache_dir: General cache directory
            cache_save: Whether to save embeddings to cache
            seed: Random seed
            pos_weights: Positive class weight for imbalanced data
            neg_ratio: Negative sampling ratio (if any)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_save_path = model_save_path
        if not os.path.exists(os.path.dirname(model_save_path)) and os.path.dirname(model_save_path) != '':
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        self.seed = seed
        self.cache_save = cache_save
        self.batch_size = batch_size
        self.loss_fn_name = loss_fn
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weights = pos_weights
        self.neg_ratio = neg_ratio

        # Set random seeds
        self._set_seed(seed)
        
        # Initialize encoders
        print("Initializing encoders...")
        self.phys_encoder = PhysicochemicalEncoder(device=self.device)
        self.esm2_encoder = ESM2Encoder(device=str(self.device), layer=esm2_layer, cache_dir=cache_dir)
        self.esmfold_encoder = ESMFoldEncoder(esm_cache_dir=esmfold_cache_dir, cache_dir=cache_dir)
        
        # Initialize model
        print("Initializing binding prediction model...")
        self.model = PeptideHLABindingPredictor(
            pep_dim=pep_dim,
            hla_dim=hla_dim,
            bilinear_dim=bilinear_dim,
            loss_fn=self.loss_fn_name,
            alpha=self.alpha,
            gamma=self.gamma,
            device=str(self.device),
            pos_weights=self.pos_weights
        ).to(self.device)
        
        # Embeddings cache
        self.phys_dict = None
        self.esm2_dict = None
        self.struct_dict = None
        
        print(f"✓ StriMap initialized on {self.device}")
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    # =============================== Embedding Preparation ===========================================
    def prepare_embeddings(
        self,
        df: pd.DataFrame,
        force_recompute: bool = False,
    ):
        """
        Prepare all embeddings (physicochemical, ESM2, structure)
        
        Args:
            df: DataFrame containing 'peptide' and 'HLA_full' columns
            force_recompute: Force recomputation even if cache exists
        """
        
        # Extract unique sequences
        all_peptides = sorted(set(df['peptide'].astype(str)))
        all_hlas = sorted(set(df['HLA_full'].astype(str)))
        
        print(f"\n{'='*70}")
        print(f"Preparing embeddings for:")
        print(f"  - {len(all_peptides)} unique peptides")
        print(f"  - {len(all_hlas)} unique HLAs")
        print(f"{'='*70}\n")
        
        # ========================================================================
        # 1. Physicochemical encoder
        # ========================================================================
        self.phys_dict = {
            'pep': self._encode_phys(all_peptides),
            'hla': self._encode_phys(all_hlas)
        }
        
        # ========================================================================
        # 2. ESM2 encoder
        # ========================================================================
        self.esm2_dict = {
            'pep': self._encode_esm2(all_peptides, prefix='pep', re_embed=force_recompute),
            'hla': self._encode_esm2(all_hlas, prefix='hla', re_embed=force_recompute)
        }
        
        # ========================================================================
        # 3. Structure encoder (only for HLA)
        # ========================================================================
        self.struct_dict = self._encode_structure(all_hlas)

        # ========================================================================
        # Summary
        # ========================================================================
        print(f"{'='*70}")
        print("✓ All embeddings prepared!")
        print(f"  - Phys: {len(self.phys_dict['pep'])} peptides, {len(self.phys_dict['hla'])} HLAs")
        print(f"  - ESM2: {len(self.esm2_dict['pep'])} peptides, {len(self.esm2_dict['hla'])} HLAs")
        print(f"  - Struct: {len(self.struct_dict)} HLAs")
        print(f"{'='*70}\n")
    
    def _encode_phys(self, 
                     sequences: List[str]) -> Dict[str, torch.Tensor]:
        """Encode physicochemical properties"""
        emb_dict = {}

        for i in tqdm(range(0, len(sequences), self.batch_size), desc="Phys encoding"):
            batch = sequences[i:i+self.batch_size]
            embs = self.phys_encoder(batch).cpu()  # [B, L, D]
            for seq, emb in zip(batch, embs):
                emb_dict[seq] = emb
        
        return emb_dict

    def _encode_esm2(self, sequences: List[str], prefix: str, re_embed: bool = False) -> Dict[str, torch.Tensor]:
        """Encode with ESM2"""
        df_tmp = pd.DataFrame({'seq': sequences})
        emb_dict = self.esm2_encoder.forward(
            df_tmp,
            seq_col='seq',
            prefix=prefix,
            batch_size=self.batch_size,
            re_embed=re_embed,
            cache_save=self.cache_save
        )
        return emb_dict

    def _encode_structure(self, sequences: List[str], re_embed: bool = False) -> Dict[str, Tuple]:
        """Encode structure with ESMFold"""
        feat_list, coor_list = self.esmfold_encoder.forward(
            pd.DataFrame({'hla': sequences}),
            'hla',
            device=str(self.device),
            re_embed=re_embed,
            cache_save=self.cache_save
        )
        
        struct_dict = {
            seq: (feat, coor)
            for seq, feat, coor in zip(sequences, feat_list, coor_list)
        }
        return struct_dict
    
    # =============================== 
    # Training / Evaluation 
    # ===========================================
    def _epoch_resample_df(self, df_pos: pd.DataFrame, df_neg: pd.DataFrame, neg_ratio: float, epoch: int) -> pd.DataFrame:
        n_pos = len(df_pos)
        n_neg_total = len(df_neg)
        n_neg_target = int(n_pos * neg_ratio)

        if n_neg_target >= n_neg_total:
            df_ep = pd.concat([df_pos, df_neg], ignore_index=True)
        else:
            rng = np.random.default_rng(seed=epoch + 12345)
            idx = rng.choice(n_neg_total, size=n_neg_target, replace=False)
            df_neg_sampled = df_neg.iloc[idx]
            df_ep = pd.concat([df_pos, df_neg_sampled], ignore_index=True)

        return df_ep.sample(frac=1.0, random_state=epoch + 2024).reset_index(drop=True)

    # ===============================
    # Training / Evaluation
    # ===========================================
    def train(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: Optional[pd.DataFrame] = None,
        epochs: int = 100,
        batch_size: int = 256,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-4,
        patience: int = 5,
        num_workers: int = 8,
        fold_id: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model
        
        Args:
            df_train: Training data
            df_val: Validation data
            df_test: Optional test data for evaluation after training
            epochs: Number of epochs
            batch_size: Batch size
            optimizer: Optimizer (if None, AdamW will be used)
            lr: Learning rate
            patience: Early stopping patience
            num_workers: Number of data loading workers
            fold_id: Fold identifier for saving (None for single model)
            
        Returns:
            Dictionary with training history
        """
        # Check if embeddings are prepared
        if self.phys_dict is None or self.esm2_dict is None or self.struct_dict is None:
            raise ValueError("Embeddings not prepared! Call prepare_embeddings() first.")
        
        # Create datasets
        print("Creating datasets...")
        if self.neg_ratio is None:
            train_dataset = PepHLA_Dataset(df_train, self.phys_dict, self.esm2_dict, self.struct_dict)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size, shuffle=True, num_workers=num_workers,
                collate_fn=peptide_hla_collate_fn, pin_memory=True
            )
        else:
            df_pos_full = df_train[df_train["label"] == 1].reset_index(drop=True)
            df_neg_full = df_train[df_train["label"] == 0].reset_index(drop=True)
        
        val_dataset = PepHLA_Dataset(df_val, self.phys_dict, self.esm2_dict, self.struct_dict)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=peptide_hla_collate_fn,
            pin_memory=True
        )
        
        if df_test is not None:
            test_dataset = PepHLA_Dataset(df_test, self.phys_dict, self.esm2_dict, self.struct_dict)
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=peptide_hla_collate_fn,
                pin_memory=True
            )
        
        # Optimizer and early stopping
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # Model save path for this fold
        save_path = self.model_save_path if fold_id is None else \
                   self.model_save_path.replace('.pt', f'_fold{fold_id}.pt')
        
        early_stopping = EarlyStopping(
            patience=patience,
            save_path=save_path
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_prc': []
        }
        
        fold_str = f"Fold {fold_id}" if fold_id is not None else "Single model"
        print(f"\nStarting training for {epochs} epochs [{fold_str}]...")
        print("=" * 70)
        
        for epoch in range(epochs):
            
            if self.neg_ratio is not None:
                df_train_epoch = self._epoch_resample_df(df_pos_full, df_neg_full, self.neg_ratio, epoch)
                train_dataset = PepHLA_Dataset(df_train_epoch, self.phys_dict, self.esm2_dict, self.struct_dict)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    collate_fn=peptide_hla_collate_fn, pin_memory=True
                )

            # Training
            self.model.train()
            train_loss = 0.0
            train_batches = 0

            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, ncols=80)

            for batch in train_iter:
                optimizer.zero_grad()
                probs, loss, _, _ = self.model(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                optimizer.step()
                train_loss += loss.item()
                train_batches += 1
            
            train_loss /= train_batches
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            val_batches = 0
            
            with torch.no_grad():
                val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False, ncols=80)
                for batch in val_iter:
                    probs, loss, _, _ = self.model(batch)
                    val_loss += loss.item()
                    val_batches += 1
                    val_preds.extend(probs)
                    val_labels.extend(batch['label'])

            val_auc = roc_auc_score(val_labels, val_preds)
            val_loss /= val_batches
            val_prc = average_precision_score(val_labels, val_preds)

            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            history['val_prc'].append(val_prc)
            
            # Print metrics
            print(f"[{fold_str}] Epoch [{epoch+1}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"Val PRC: {val_prc:.4f}")
            
            if df_test is not None:
                # Test evaluation
                test_loss = 0.0
                test_preds = []
                test_labels = []
                test_batches = 0

                with torch.no_grad():
                    test_iter = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]", leave=False, ncols=80)
                    for batch in test_iter:
                        probs, loss, _, _ = self.model(batch)
                        test_loss += loss.item()
                        test_batches += 1
                        test_preds.extend(probs)
                        test_labels.extend(batch["label"])

                test_loss /= test_batches

                # Get evaluation metrics
                df_eval = df_test.copy()
                df_eval = df_eval.reset_index(drop=True)
                df_eval["pred"] = test_preds
                df_eval["label"] = test_labels

                # Compute per-HLA AUROC and AUPRC
                records = []
                for hla, group in df_eval.groupby("HLA"):
                    y_true = np.array(group["label"], dtype=int)
                    y_score = np.array(group["pred"], dtype=float)
                    if len(np.unique(y_true)) < 2:
                        continue
                    auroc = roc_auc_score(y_true, y_score)
                    auprc = average_precision_score(y_true, y_score)
                    records.append({
                        "HLA": hla,
                        "AUROC": auroc,
                        "AUPRC": auprc,
                        "n_samples": len(group)
                    })

                df_metrics = pd.DataFrame(records)

                if len(df_metrics) == 0:
                    print(f"[{fold_str}] Epoch [{epoch+1}/{epochs}] | No valid HLA groups for AUROC/AUPRC.")
                else:
                    mean_auroc = df_metrics["AUROC"].mean()
                    mean_auprc = df_metrics["AUPRC"].mean()

                    print(f"[{fold_str}] Epoch [{epoch+1}/{epochs}] | "
                        f"Test Loss: {test_loss:.4f} | "
                        f"Mean AUROC: {mean_auroc:.4f} | "
                        f"Mean AUPRC: {mean_auprc:.4f} | "
                        f"HLA Count: {len(df_metrics)}")

                    print("Top HLA performance:")
                    print(df_metrics.sort_values("AUROC", ascending=False).to_string(index=False))
                
            # Early stopping
            early_stopping(val_prc, self.model)
            
            if early_stopping.early_stop:
                print(f"\n[{fold_str}] Early stopping triggered at epoch {epoch+1}!")
                break
        
        # Load best model
        print(f"\n[{fold_str}] Loading best model from {save_path}...")
        self.model.load_state_dict(torch.load(save_path))
        
        print("=" * 70)
        print(f"✓ Training completed for {fold_str}!")
        
        return history
    
    # ============================================
    # K-Fold Cross-Validation Training
    # ============================================
    def train_kfold(
        self,
        train_folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
        df_test: Optional[pd.DataFrame] = None,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-4,
        patience: int = 5,
        num_workers: int = 8
    ) -> List[Dict[str, List[float]]]:
        """
        Train K-fold cross-validation models
        
        Args:
            train_folds: List of (train_df, val_df) tuples for each fold
            df_test: Optional test data for evaluation after training
            epochs: Number of epochs per fold
            batch_size: Batch size
            lr: Learning rate
            patience: Early stopping patience
            num_workers: Number of data loading workers
            
        Returns:
            List of training histories for each fold
        """
        num_folds = len(train_folds)
        all_histories = []
        
        print("\n" + "=" * 70)
        print(f"Starting {num_folds}-Fold Cross-Validation Training")
        print("=" * 70)
        
        for fold_id, (df_train, df_val) in enumerate(train_folds):
            print(f"\n{'='*70}")
            print(f"Training Fold {fold_id+1}/{num_folds}")
            print(f"Train: {len(df_train)} samples | Val: {len(df_val)} samples")
            print(f"{'='*70}")
            
            self._set_seed(fold_id + self.seed)  # Different seed for each fold
            
            # Reinitialize model for this fold
            self.model = PeptideHLABindingPredictor(
                pep_dim=self.model.pep_dim,
                hla_dim=self.model.hla_dim,
                bilinear_dim=self.model.bilinear_dim,
                loss_fn=self.loss_fn_name,
                alpha=self.alpha,
                gamma=self.gamma,
                device=str(self.device),
                pos_weights=self.pos_weights
            ).to(self.device)

            # Train this fold
            history = self.train(
                df_train,
                df_val,
                df_test=df_test,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                num_workers=num_workers,
                fold_id=fold_id
            )
            
            all_histories.append(history)
        
        print("\n" + "=" * 70)
        print(f"✓ All {num_folds} folds training completed!")
        print("=" * 70)
        
        # Print summary
        print("\nCross-Validation Summary:")
        print("-" * 70)
        for fold_id, history in enumerate(all_histories):
            best_auc = max(history['val_auc'])
            best_epoch = history['val_auc'].index(best_auc) + 1
            print(f"Fold {fold_id}: Best Val AUC = {best_auc:.4f} (Epoch {best_epoch})")
        
        mean_auc = np.mean([max(h['val_auc']) for h in all_histories])
        std_auc = np.std([max(h['val_auc']) for h in all_histories])
        print("-" * 70)
        print(f"Mean Val AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print("=" * 70 + "\n")
        
        return all_histories
    
    # ============================================
    # Prediction
    # ============================================
    def predict(
        self,
        df: pd.DataFrame,
        batch_size: int = 256,
        return_probs: bool = True,
        return_attn: bool = False,
        use_kfold: bool = False,
        num_folds: Optional[int] = None,
        ensemble_method: str = 'mean',
        num_workers: int = 8
    ) -> np.ndarray:
        """
        Make predictions on a dataset
        
        Args:
            df: DataFrame with peptide and HLA_full columns
            batch_size: Batch size for inference
            return_probs: If True, return probabilities; else return binary predictions
            return_attn: If True, return attention maps
            use_kfold: If True, use ensemble of K models
            num_folds: Number of folds (required if use_kfold=True)
            ensemble_method: 'mean' or 'median' for ensemble
            num_workers: Number of data loading workers
            
        Returns:
            Array of predictions
        """
        # Check if embeddings are prepared
        if self.phys_dict is None or self.esm2_dict is None or self.struct_dict is None:
            raise ValueError("Embeddings not prepared! Call prepare_embeddings() first.")
        
        if use_kfold:
            if num_folds is None:
                raise ValueError("num_folds must be specified when use_kfold=True")
            
            return self._predict_ensemble(
                df, 
                batch_size, 
                num_folds, 
                ensemble_method, 
                return_probs,
                return_attn,
                num_workers
            )
        else:
            # load single model
            print(f"\nLoading model from {self.model_save_path} for prediction...")
            self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device), strict=False)
            # Single model prediction
            return self._predict_single(df, batch_size, return_probs, return_attn, num_workers)

    def _pad_attention(self, attns: List[np.ndarray]) -> np.ndarray:
        """Pad attention maps to the same length"""
        max_len = max(a.shape[1] for a in attns)
        attns_padded = []
        for a in attns:
            padding = max_len - a.shape[1]
            pad_width_3d = ((0, 0),        # No padding for N dimension
                            (0, padding),  # Pad Lq dimension
                            (0, 0))        # No padding for Lk dimension
            
            attns_padded.append(np.pad(a, pad_width_3d, mode='constant', constant_values=0.0))
        return np.concatenate(attns_padded, axis=0)

    def _predict_single(
        self,
        df: pd.DataFrame,
        batch_size: int,
        return_probs: bool,
        return_attn: bool = False,
        num_workers: int = 8
    ) -> np.ndarray:
        """Single model prediction
        
        Args:
            df: DataFrame with peptide and HLA_full columns
            batch_size: Batch size for inference
            return_probs: If True, return probabilities; else return binary predictions
            return_attn: If True, return attention maps
            num_workers: Number of data loading workers
        
        Returns:
            Array of predictions
        """
        self.model.eval()
        
        dataset = PepHLA_Dataset(df, self.phys_dict, self.esm2_dict, self.struct_dict)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=peptide_hla_collate_fn,
            pin_memory=True
        )
        
        preds = []
        attns = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                probs, loss, attn, _ = self.model(batch)
                preds.extend(probs.tolist())
                if return_attn:
                    attns.append(attn)
        
        preds = np.array(preds)
        if not return_probs:
            preds = (preds >= 0.5).astype(int)
            
        # padding attns to the same length
        if not return_attn:
            return preds, None
        else:
            return preds, self._pad_attention(attns)

    def _predict_ensemble(
        self,
        df: pd.DataFrame,
        batch_size: int,
        num_folds: int,
        ensemble_method: str,
        return_probs: bool,
        return_attn: bool = False,
        num_workers: int = 8
    ) -> np.ndarray:
        """Ensemble prediction using K-fold models
        
        Args:
            df: DataFrame with peptide and HLA_full columns
            batch_size: Batch size for inference
            num_folds: Number of folds
            ensemble_method: 'mean' or 'median' for ensemble
            return_probs: If True, return probabilities; else return binary predictions
            return_attn: If True, return attention maps
            num_workers: Number of data loading workers
            
            Returns:
            Array of predictions
        """
        
        print(f"\nEnsemble prediction using {num_folds} models...")
        print(f"Ensemble method: {ensemble_method}")
        
        all_preds = []
        all_attns = []
        
        for fold_id in range(num_folds):
            # Load fold model
            fold_model_path = self.model_save_path.replace('.pt', f'_fold{fold_id}.pt')
            
            if not os.path.exists(fold_model_path):
                print(f"⚠ Warning: {fold_model_path} not found, skipping...")
                continue
            
            print(f"Loading model from {fold_model_path}...")
            self.model.load_state_dict(torch.load(fold_model_path, map_location=self.device), strict=False)
            
            # Predict with this fold
            if not return_attn:
                fold_preds, _ = self._predict_single(df, batch_size, return_probs=True, num_workers=num_workers)
            else:
                fold_preds, attn_padded = self._predict_single(df, batch_size, return_probs=True, return_attn=True, num_workers=num_workers)
                all_attns.append(attn_padded)

            all_preds.append(fold_preds)
            
        if len(all_preds) == 0:
            raise ValueError("No fold models found!")
        
        # Ensemble predictions
        all_preds = np.array(all_preds)  # [num_folds, num_samples]
        
        if ensemble_method == 'mean':
            ensemble_preds = np.mean(all_preds, axis=0)
        elif ensemble_method == 'median':
            ensemble_preds = np.median(all_preds, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
        
        print(f"✓ Ensemble prediction completed using {len(all_preds)} models")
        
        if not return_probs:
            ensemble_preds = (ensemble_preds >= 0.5).astype(int)

        if not return_attn:
            return ensemble_preds, None
        else:
            
            # num_attn_each_fold = attns_padded.shape[0] // len(all_preds)
            # # average attns across folds
            # attns_padded = attns_padded.reshape(len(all_preds), num_attn_each_fold, attns_padded.shape[1], attns_padded.shape[2])
            # attns_padded = np.mean(attns_padded, axis=1)
            return ensemble_preds, self._pad_attention(all_attns)

    def evaluate(
        self,
        df: pd.DataFrame,
        batch_size: int = 256,
        threshold: float = 0.5,
        use_kfold: bool = False,
        num_folds: Optional[int] = None,
        ensemble_method: str = 'mean',
        num_workers: int = 8
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset
        
        Args:
            df: DataFrame with peptide, HLA_full, and label columns
            batch_size: Batch size for inference
            threshold: Classification threshold
            use_kfold: If True, use ensemble of K models
            num_folds: Number of folds (required if use_kfold=True)
            ensemble_method: 'mean' or 'median' for ensemble
            num_workers: Number of data loading workers
            
        Returns:
            Dictionary of metrics
        """
        y_true = df['label'].values
        y_prob, _ = self.predict(
            df, 
            batch_size=batch_size, 
            return_probs=True,
            use_kfold=use_kfold,
            num_folds=num_folds,
            ensemble_method=ensemble_method,
            num_workers=num_workers
        )
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
        
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        
        try:
            mcc = ((tp*tn) - (fn*fp)) / np.sqrt(float((tp+fn)*(tn+fp)*(tp+fp)*(tn+fn)))
        except:
            mcc = 0.0
        
        try:
            recall = tp / (tp + fn)
        except:
            recall = 0.0
        
        try:
            precision = tp / (tp + fp)
        except:
            precision = 0.0
        
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except:
            f1 = 0.0
        
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except:
            roc_auc = 0.0
            
        try:
            # prc
            from sklearn.metrics import average_precision_score
            prc_auc = average_precision_score(y_true, y_prob)
        except:
            prc_auc = 0.0
        
        # Print results
        model_type = f"{num_folds}-Fold Ensemble ({ensemble_method})" if use_kfold else "Single Model"
        
        print("\n" + "=" * 70)
        print(f"Evaluation Results [{model_type}]")
        print("=" * 70)
        print(f"tn = {tn}, fp = {fp}, fn = {fn}, tp = {tp}")
        print(f"y_pred: 0 = {Counter(y_pred)[0]} | 1 = {Counter(y_pred)[1]}")
        print(f"y_true: 0 = {Counter(y_true)[0]} | 1 = {Counter(y_true)[1]}")
        print(f"AUC: {roc_auc:.4f} | PRC: {prc_auc:.4f} | ACC: {accuracy:.4f} | MCC: {mcc:.4f} | F1: {f1:.4f}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f}")
        print("=" * 70 + "\n")
        
        return y_prob, {
            'auc': roc_auc,
            'prc': prc_auc,
            'accuracy': accuracy,
            'mcc': mcc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device), strict=False)
        print(f"✓ Model loaded from {path}")
        
# ============================================================================
# TCR-pHLA predictor Classes
# ============================================================================
class StriMap_TCRpHLA:
    """
    Structure-informed TCR(alpha/beta)-peptide-HLA Binding Prediction
    - Reuses encoders from StriMap_pHLA (phys, ESM2, ESMFold)
    - Precomputes peptide-HLA features using pretrained StriMap_pHLA.model (PeptideHLABindingPredictor)
      and injects them into batch during training/inference.
    """
    
    def __init__(
        self,
        pep_hla_system = None,   # already-initialized and pretrained
        pep_hla_params: Optional[list] = None,
        device: str = 'cuda:0',
        model_save_path: str = 'best_model_tcrpHLA.pt',
        tcr_dim: int = 256,
        pep_dim: int = 256,
        hla_dim: int = 256,
        bilinear_dim: int = 256,
        loss_fn: str = 'focal',
        alpha: float = 0.5,
        gamma: float = 2.0,
        resample_negatives: bool = False,
        seed: int = 1,
        pos_weights: Optional[float] = None,
        use_struct: bool = True,
        cache_save: bool = False,
    ):
        '''
        Initialize StriMap_TCRpHLA
        Args:
            pep_hla_system: Pretrained StriMap_pHLA system to reuse encoders
            pep_hla_params: List of peptide-HLA feature names to use from StriMap_pHLA
            device: Device to use ('cuda:0' or 'cpu')
            model_save_path: Path to save the trained model
            tcr_dim: Dimension of TCR embedding
            pep_dim: Dimension of peptide embedding
            hla_dim: Dimension of HLA embedding
            bilinear_dim: Dimension of bilinear layer
            loss_fn: Loss function name ('focal', 'bce', etc.)
            alpha: Alpha parameter for focal loss
            gamma: Gamma parameter for focal loss
            resample_negatives: Whether to resample negatives each epoch
            seed: Random seed for reproducibility
            pos_weights: Positive class weight for loss function
            use_struct: Whether to use structural features
            cache_save: Whether to save embedding caches    
        '''
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_save_path = model_save_path
        self.seed = seed
        self.alpha = alpha
        self.gamma = gamma
        self.loss_fn_name = loss_fn
        self.resample_negatives = resample_negatives
        self.pos_weights = pos_weights
        self.pep_hla_params = pep_hla_params
        self.use_struct = use_struct
        self.cache_save = cache_save

        # seed
        self._set_seed(seed)
        
        if pep_hla_system is None:
            raise ValueError("`pep_hla_system` must be provided — pass a trained StriMap_pHLA instance.")

        # Reuse encoders from StriMap_pHLA
        self.phys_encoder   = pep_hla_system.phys_encoder
        self.esm2_encoder   = pep_hla_system.esm2_encoder
        self.esmfold_encoder= pep_hla_system.esmfold_encoder
        self.pep_hla_model  = pep_hla_system.model   # PeptideHLABindingPredictor with encode_peptide_hla()

        # Initialize TCR–pHLA model
        self.model = TCRPeptideHLABindingPredictor(
            tcr_dim=tcr_dim, 
            pep_dim=pep_dim, 
            hla_dim=hla_dim, 
            bilinear_dim=bilinear_dim, 
            loss_fn=self.loss_fn_name,
            alpha=self.alpha,
            gamma=self.gamma,
            pos_weights=self.pos_weights,
            device=str(self.device),
            use_struct=self.use_struct,
        ).to(self.device)

        # Embedding caches
        self.phys_dict = None
        self.esm2_dict = None
        self.struct_dict = None
        self.pep_hla_feat_dict = {}

        print(f"✓ StriMap_TCRpHLA initialized on {self.device}")

    # ============================================================
    # Utils
    # ============================================================
    def _set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # ============================================================
    # Encoders
    # ============================================================
    def _encode_phys(self, sequences):
        emb_dict = {}
        batch_size = 256
        for i in tqdm(range(0, len(sequences), batch_size), desc="Phys encoding (TCRpHLA)"):
            batch = sequences[i:i+batch_size]
            embs = self.phys_encoder(batch).cpu()  # [B, L, D]
            for seq, emb in zip(batch, embs):
                emb_dict[seq] = emb
        return emb_dict
    
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"✓ Model saved to {path}")
        
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"✓ Model loaded from {path}")

    def _encode_esm2(self, sequences, prefix: str, re_embed: bool=False):
        df_tmp = pd.DataFrame({'seq': sequences})
        return self.esm2_encoder.forward(
            df_tmp, seq_col='seq', prefix=prefix, batch_size=128, re_embed=re_embed, cache_save=self.cache_save
        )

    def _encode_structure(self, sequences, prefix: str, re_embed: bool=False):
        feat_list, coor_list = self.esmfold_encoder.forward(
            pd.DataFrame({prefix: sequences}), prefix, device=str(self.device), re_embed=re_embed, cache_save=self.cache_save
        )
        return {seq: (feat, coor) for seq, feat, coor in zip(sequences, feat_list, coor_list)}

    # ============================================================
    # Prepare embeddings
    # ============================================================
    def prepare_embeddings(self, df: pd.DataFrame, force_recompute: bool=False):
        """
        Prepare per-residue encodings for TCRα, TCRβ, peptide, and HLA.
        Peptide structure is computed via ESMFold as requested.
        """
        all_tcra = sorted(set(df['tcra'].astype(str)))
        all_tcrb = sorted(set(df['tcrb'].astype(str)))
        all_peps = sorted(set(df['peptide'].astype(str)))
        all_hlas = sorted(set(df['HLA_full'].astype(str)))
        
        self.max_pep_len = max(len(p) for p in all_peps)

        print(f"\nPreparing embeddings:")
        print(f"  - TCRα: {len(all_tcra)} | TCRβ: {len(all_tcrb)} | peptides: {len(all_peps)} | HLAs: {len(all_hlas)}\n")

        self.phys_dict = {
            'tcra': self._encode_phys(all_tcra),
            'tcrb': self._encode_phys(all_tcrb),
            'pep':  self._encode_phys(all_peps),
            'hla':  self._encode_phys(all_hlas)
        }
        self.esm2_dict = {
            'tcra': self._encode_esm2(all_tcra, prefix='tcra', re_embed=force_recompute),
            'tcrb': self._encode_esm2(all_tcrb, prefix='tcrb', re_embed=force_recompute),
            'pep':  self._encode_esm2(all_peps, prefix='pep', re_embed=force_recompute),
            'hla':  self._encode_esm2(all_hlas, prefix='hla', re_embed=force_recompute)
        }
        
        # Move everything in phys_dict and esm2_dict to CPU
        for d in [self.phys_dict, self.esm2_dict]:
            for k1 in d.keys():       # tcra / tcrb / pep / hla
                for k2 in d[k1].keys():  # actual sequences
                    if torch.is_tensor(d[k1][k2]):
                        d[k1][k2] = d[k1][k2].cpu()
        
        torch.cuda.empty_cache()
                        
        # IMPORTANT: include peptide structure via ESMFold
        if self.use_struct:
            self.struct_dict = {
                'tcra': self._encode_structure(all_tcra, prefix='tcra', re_embed=force_recompute),
                'tcrb': self._encode_structure(all_tcrb, prefix='tcrb', re_embed=force_recompute),
                'pep':  self._encode_structure(all_peps, prefix='pep',  re_embed=force_recompute),
                'hla':  self._encode_structure(all_hlas, prefix='hla',  re_embed=force_recompute)
            }

            print("✓ Embeddings prepared for TCRα/β, peptide (with ESMFold), and HLA.")
            # Move structure features to CPU
            for part in ['tcra', 'tcrb', 'pep', 'hla']:
                for seq, (feat, coord) in self.struct_dict[part].items():
                    self.struct_dict[part][seq] = (feat.cpu(), coord.cpu())
        else:
            # when use_struct=False, use "pseudo-structure":
            # assign each seq a zero (L, D_feat) and (L, 3) structural feature
            ZERO_FEAT_DIM = 21
            def build_zero_struct_dict(seqs):
                d = {}
                for seq in seqs:
                    L = len(seq)
                    feat = torch.zeros(L, ZERO_FEAT_DIM)   # (L, D_feat)
                    coord = torch.zeros(L, 3)              # (L, 3)
                    d[seq] = (feat, coord)
                return d

            self.struct_dict = {
                'tcra': build_zero_struct_dict(all_tcra),
                'tcrb': build_zero_struct_dict(all_tcrb),
                'pep':  build_zero_struct_dict(all_peps),
                'hla':  self._encode_structure(all_hlas, prefix='hla',  re_embed=force_recompute)
            }
                
        torch.cuda.empty_cache()
        print("✓ Embeddings prepared for TCRα/β, peptide, and HLA.")

    def _pad_or_crop_tensor(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Pad or crop a tensor to the target length along the first dimension.
        """
        L, D = tensor.shape
        if L == target_len:
            return tensor
        if L > target_len:
            return tensor[:target_len]
        pad_len = target_len - L
        padding = tensor.new_zeros(pad_len, D)
        return torch.cat([tensor, padding], dim=0)
        
    # ============================================================
    # Precompute peptide-HLA features
    # ============================================================
    def prepare_pep_hla_features(self, df: pd.DataFrame, batch_size: int = 256):
        """
        Precompute peptide-HLA features using the pretrained PeptideHLABindingPredictor model.
        """
        assert self.phys_dict is not None and self.esm2_dict is not None and self.struct_dict is not None, \
            "Call prepare_embeddings() first."

        pairs = list(set((row['peptide'], row['HLA_full']) for _, row in df.iterrows()))
        
        # 1. Set model to eval mode
        self.pep_hla_model.eval()
        for p in self.pep_hla_model.parameters():
            p.requires_grad = False

        print(f"\nPrecomputing peptide-HLA features for {len(pairs)} unique pairs...")
        
        with torch.no_grad():
            # 2. Manually iterate over batches
            for i in tqdm(range(0, len(pairs), batch_size), desc="pHLA features (batched)"):
                batch_pairs = pairs[i:i+batch_size]
                if not batch_pairs:
                    continue
                
                # 3. Manually assemble batch data
                batch_dict = {
                    'pep_ids': [], 'hla_ids': [],
                    'pep_phys_list': [], 'pep_esm_list': [],
                    'hla_phys_list': [], 'hla_esm_list': [], 'hla_struct_list': [], 'hla_coord_list': [],
                    'pep_lens_list': [],
                }
                
                for pep, hla in batch_pairs:
                    batch_dict['pep_ids'].append(pep)
                    batch_dict['hla_ids'].append(hla)
                    batch_dict['pep_lens_list'].append(len(pep))
                    
                    # Retrieve data
                    pep_phys, pep_esm = self.phys_dict['pep'][pep], self.esm2_dict['pep'][pep]
                    hla_phys, hla_esm = self.phys_dict['hla'][hla], self.esm2_dict['hla'][hla]
                    hla_struct, hla_coord = self.struct_dict['hla'][hla]
                    
                    # Pad Peptide (HLA is fixed length, no need to pad)
                    batch_dict['pep_phys_list'].append(self._pad_or_crop_tensor(pep_phys, self.max_pep_len))
                    batch_dict['pep_esm_list'].append(self._pad_or_crop_tensor(pep_esm, self.max_pep_len))
                    
                    batch_dict['hla_phys_list'].append(hla_phys)
                    batch_dict['hla_esm_list'].append(hla_esm)
                    batch_dict['hla_struct_list'].append(hla_struct)
                    batch_dict['hla_coord_list'].append(hla_coord)

                # 4. Convert lists to tensors
                final_batch = {
                    'pep_lens': torch.tensor(batch_dict['pep_lens_list'], dtype=torch.long),
                    'pep_phys': torch.stack(batch_dict['pep_phys_list']),
                    'pep_esm': torch.stack(batch_dict['pep_esm_list']),
                    'hla_phys': torch.stack(batch_dict['hla_phys_list']),
                    'hla_esm': torch.stack(batch_dict['hla_esm_list']),
                    'hla_struct': torch.stack(batch_dict['hla_struct_list']),
                    'hla_coord': torch.stack(batch_dict['hla_coord_list']),
                }
                
                # 5. Run model (K-fold ensemble or S-fold)
                if self.pep_hla_params is None:
                    # Note: Here we call the new batch-supported encode_peptide_hla
                    pep_feat_batch, hla_feat_batch = self.pep_hla_model.encode_peptide_hla_batch(
                        final_batch, self.max_pep_len
                    )
                else:
                    pep_feature_list, hla_feature_list = [], []
                    for param in self.pep_hla_params:
                        self.pep_hla_model.load_state_dict(torch.load(param, map_location=self.device), strict=False)
                        pep_f, hla_f = self.pep_hla_model.encode_peptide_hla_batch(
                            final_batch, self.max_pep_len
                        )
                        pep_feature_list.append(pep_f)
                        hla_feature_list.append(hla_f)
                    pep_feat_batch = torch.mean(torch.stack(pep_feature_list, dim=0), dim=0)
                    hla_feat_batch = torch.mean(torch.stack(hla_feature_list, dim=0), dim=0)
                
                # 6. Store results back to dictionary (on CPU)
                pep_feat_cpu = pep_feat_batch.cpu() # [B, Lp, D]
                hla_feat_cpu = hla_feat_batch.cpu() # [B, Lh, D]

                for idx, (pep, hla) in enumerate(batch_pairs):
                    self.pep_hla_feat_dict[(pep, hla)] = {
                        'pep_feat_pretrain': pep_feat_cpu[idx],
                        'hla_feat_pretrain': hla_feat_cpu[idx]
                    }

        print("✓ Pretrained peptide-HLA features prepared.")

    # ============================================================
    # Training
    # ============================================================
    def train(
        self,
        df_train: pd.DataFrame,
        df_val: Optional[pd.DataFrame] = None,
        df_test: Optional[pd.DataFrame] = None,
        df_add: Optional[pd.DataFrame] = None,
        epochs: int = 100,
        batch_size: int = 128,
        lr: float = 1e-4,
        optimizer: Optional[torch.optim.Optimizer] = None,
        patience: int = 5,
        num_workers: int = 8,
    ):
        """
        Train the TCR-pHLA model.

        Args:
            df_train: Training data.
            df_val: Optional validation data.
            df_test: Optional test data for evaluation after each epoch.
            df_add: Optional additional data for training. Set when resample_negatives=True.
            epochs: Number of epochs.
            batch_size: Batch size.
            lr: Learning rate.
            patience: Early stopping patience.
            num_workers: Data loading workers.

        Returns:
            history: Dict containing training and validation metrics.
        """

        # ---- Prepare embeddings ----
        print("Preparing peptide-HLA features...")
        all_dfs = [df for df in [df_train, df_val, df_test, df_add] if df is not None]
        self.prepare_pep_hla_features(pd.concat(all_dfs, axis=0))

        # ---- Validation loader (optional) ----
        if df_val is not None:
            val_ds = TCRPepHLA_Dataset(df_val, self.phys_dict, self.esm2_dict, self.struct_dict, self.pep_hla_feat_dict)
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                collate_fn=tcr_pep_hla_collate_fn, pin_memory=True, drop_last=True
            )
            stopper = EarlyStopping(patience=patience, save_path=self.model_save_path)
        else:
            val_loader, stopper = None, None

        # ---- Optimizer ----
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # ---- Metric history ----
        history = {'train_loss': [], 'train_auc': []}
        if df_val is not None:
            history.update({'val_loss': [], 'val_auc': [], 'val_prc': []})

        print("\nStart training TCR-pHLA model...")
        df_train_pos = df_train[df_train['label'] == 1].copy().reset_index(drop=True)

        for epoch in range(epochs):
            # ---------- Training ----------
            if self.resample_negatives:
                df_train_neg = negative_sampling_phla(df_train_pos, random_state=epoch)
                df_train_resample = pd.concat([df_train_pos, df_train_neg], axis=0).reset_index(drop=True)
                if df_add is not None:
                    df_train_resample = pd.concat([df_train_resample, df_add], axis=0).reset_index(drop=True)
                train_ds = TCRPepHLA_Dataset(df_train_resample, self.phys_dict, self.esm2_dict, self.struct_dict, self.pep_hla_feat_dict)
            else:
                train_ds = TCRPepHLA_Dataset(df_train, self.phys_dict, self.esm2_dict, self.struct_dict, self.pep_hla_feat_dict)

            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                collate_fn=tcr_pep_hla_collate_fn, pin_memory=True, drop_last=True
            )

            self.model.train()
            train_labels, train_preds = [], []
            epoch_loss = 0.0

            for ibatch, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
                optimizer.zero_grad()
                probs, loss, _, _ = self.model(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                optimizer.step()

                epoch_loss += loss.item()
                train_labels.extend(batch['label'].cpu().numpy().tolist())
                train_preds.extend(probs.detach().cpu().numpy().tolist())
                
            train_loss = epoch_loss / (ibatch + 1)
            history['train_loss'].append(train_loss)

            # if only one class present in y_true, roc_auc_score will raise an error
            if len(set(train_labels)) > 1:
                train_auc = roc_auc_score(train_labels, train_preds)
                history['train_auc'].append(train_auc)
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train AUC: N/A (only one class present)")

            # ---------- Validation ----------
            if df_val is not None:
                self.model.eval()
                val_loss_sum, val_labels, val_preds = 0.0, [], []
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                        probs, loss, _, _ = self.model(batch)
                        val_loss_sum += loss.item()
                        val_labels.extend(batch['label'].cpu().numpy().tolist())
                        val_preds.extend(probs.detach().cpu().numpy().tolist())

                val_loss = val_loss_sum / len(val_loader)
                history['val_loss'].append(val_loss)
                if len(set(val_labels)) > 1:
                    val_auc = roc_auc_score(val_labels, val_preds)
                    val_prc = average_precision_score(val_labels, val_preds)
                    history['val_auc'].append(val_auc)
                    history['val_prc'].append(val_prc)
                    print(f"Epoch {epoch+1}/{epochs} | Val AUC: {val_auc:.4f} | Val PRC: {val_prc:.4f} | Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} | Val AUC: N/A (only one class present) | Val Loss: {val_loss:.4f}")

                stopper(val_auc, self.model)
                if stopper.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # ---------- Optional Test ----------
            if df_test is not None:
                test_ds = TCRPepHLA_Dataset(df_test, self.phys_dict, self.esm2_dict, self.struct_dict, self.pep_hla_feat_dict)
                test_loader = torch.utils.data.DataLoader(
                    test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    collate_fn=tcr_pep_hla_collate_fn, pin_memory=True
                )
                self.model.eval()
                test_labels, test_preds = [], []
                with torch.no_grad():
                    for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]"):
                        probs, _, _, _ = self.model(batch)
                        test_labels.extend(batch['label'].cpu().numpy().tolist())
                        test_preds.extend(probs.detach().cpu().numpy().tolist())
                test_auc = roc_auc_score(test_labels, test_preds)
                test_prc = average_precision_score(test_labels, test_preds)
                print(f"Epoch {epoch+1}/{epochs} | Test AUC: {test_auc:.4f} | Test PRC: {test_prc:.4f}")

        # ---- Load best model only if validation used ----
        if df_val is not None and os.path.exists(self.model_save_path):
            self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
            print(f"✓ Training finished. Best model loaded from {self.model_save_path}")
        else:
            print("✓ Training finished (no validation set used).")
            # save final model
            torch.save(self.model.state_dict(), self.model_save_path)
            print(f"✓ Final model saved to {self.model_save_path}")

        return history
    
    # ================================================================
    # K-Fold Cross-Validation Training
    # ================================================================
    def train_kfold(
        self,
        train_folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
        df_test: Optional[pd.DataFrame] = None,
        df_add: Optional[pd.DataFrame] = None,
        epochs: int = 100,
        batch_size: int = 128,
        lr: float = 1e-4,
        patience: int = 8,
        num_workers: int = 8,
    ) -> List[Dict[str, List[float]]]:
        """
        K-fold cross-validation training for TCR-pHLA model.

        Args:
            train_folds: list of (train_df, val_df) tuples for each fold
            df_test: optional test data for evaluation after each epoch
            df_add: optional additional samples for training. Set when resample_negatives=True.
            epochs: training epochs
            batch_size: batch size
            lr: learning rate
            patience: early stopping patience
            num_workers: dataloader workers

        Returns:
            List of training histories for each fold
        """
        num_folds = len(train_folds)
        all_histories = []

        print("\n" + "=" * 70)
        print(f"Starting {num_folds}-Fold Cross-Validation Training (TCR-pHLA)")
        print("=" * 70)

        for fold_id, (df_train, df_val) in enumerate(train_folds):
            print(f"\n{'='*70}")
            print(f"Training Fold {fold_id+1}/{num_folds}")
            print(f"{'='*70}")

            self._set_seed(self.seed + fold_id)

            self.model = TCRPeptideHLABindingPredictor(
                tcr_dim=self.model.tcr_dim,
                pep_dim=self.model.pep_dim,
                hla_dim=self.model.hla_dim,
                bilinear_dim=self.model.bilinear_dim,
                loss_fn=self.loss_fn_name,
                alpha=self.alpha,
                gamma=self.gamma,
                pos_weights=self.pos_weights,
                device=str(self.device),
            ).to(self.device)

            fold_save_path = self.model_save_path.replace(".pt", f"_fold{fold_id}.pt")

            history = self.train(
                df_train=df_train,
                df_val=df_val,
                df_test=df_test,
                df_add=df_add,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                num_workers=num_workers,
            )

            torch.save(self.model.state_dict(), fold_save_path)
            print(f"✓ Saved fold {fold_id} model to {fold_save_path}")

            all_histories.append(history)

        print("\n" + "=" * 70)
        print(f"✓ All {num_folds} folds training completed (TCR-pHLA)")
        print("=" * 70)

        if df_val is not None:
            print("\nCross-Validation Summary:")
            print("-" * 70)
            for fold_id, hist in enumerate(all_histories):
                best_auc = max(hist['val_auc'])
                best_prc = max(hist['val_prc'])
                best_epoch = hist['val_auc'].index(best_auc) + 1
                print(f"Fold {fold_id}: Best Val AUC = {best_auc:.4f}, Best Val PRC = {best_prc:.4f}, (Epoch {best_epoch})")

            mean_auc = np.mean([max(h['val_auc']) for h in all_histories])
            std_auc = np.std([max(h['val_auc']) for h in all_histories])
            print("-" * 70)
            print(f"Mean Val AUC: {mean_auc:.4f} ± {std_auc:.4f}")
            print("=" * 70 + "\n")

        return all_histories    
    
    # ================================================================
    # Fine-tuning with Validation monitoring
    # ================================================================
    def finetune(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-5,
        num_workers: int = 0
    ):
        print("\nStarting fine-tuning with Validation monitoring...")
        
        # 1. Prepare (Train & Val)
        self.prepare_pep_hla_features(df_train)
        self.prepare_pep_hla_features(df_val)
        
        # 2. Load base weights
        self.load_model(self.model_save_path)

        # 3. Build DataLoaders
        train_ds = TCRPepHLA_Dataset(df_train, self.phys_dict, self.esm2_dict, self.struct_dict, self.pep_hla_feat_dict)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            collate_fn=tcr_pep_hla_collate_fn, pin_memory=True
        )

        val_ds = TCRPepHLA_Dataset(df_val, self.phys_dict, self.esm2_dict, self.struct_dict, self.pep_hla_feat_dict)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, # Shuffle=False for val
            collate_fn=tcr_pep_hla_collate_fn, pin_memory=True
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # Record history
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # === Training Phase ===
            self.model.train()
            train_loss_sum = 0.0
            train_steps = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                probs, loss, _, _ = self.model(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                optimizer.step()
                
                train_loss_sum += loss.item()
                train_steps += 1
            
            avg_train_loss = train_loss_sum / max(1, train_steps)
            history["train_loss"].append(avg_train_loss)

            # === Validation Phase ===
            self.model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    # model forward returns: probs, loss, fused_feat, attn_weights
                    _, v_loss, _, _ = self.model(batch)
                    val_loss_sum += v_loss.item()
                    val_steps += 1
            
            avg_val_loss = val_loss_sum / max(1, val_steps)
            history["val_loss"].append(avg_val_loss)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        print("✓ Fine-tuning completed.")
        return history
    
    # ================================================================
    # Attention padding utility
    # ================================================================
    def _pad_attention(self, attn_list: List[np.ndarray]) -> np.ndarray:
        """
        Pad and merge a list of 3D attention maps (from different batches).
        
        Input: [[B1, L1, W1], [B2, L2, W2], ...]
        Output: [B_total, L_max, W_max]
        """
        
        # 1. Determine max L and W
        max_L = 0
        max_W = 0
        valid_batches = [] # Filter out empty/invalid batches
        
        for batch_arr in attn_list:
            # batch_arr might be an empty np.array
            if batch_arr is not None and batch_arr.ndim == 3 and batch_arr.shape[0] > 0:
                max_L = max(max_L, batch_arr.shape[1])
                max_W = max(max_W, batch_arr.shape[2])
                valid_batches.append(batch_arr)

        if not valid_batches:
             return np.empty((0, 0, 0), dtype=np.float32)

        # 2. Pad each batch and collect
        padded_batches = []
        for batch_arr in valid_batches:
            B, L, W = batch_arr.shape
            pad_L = max_L - L
            pad_W = max_W - W
            
            # np.pad's padding width format: ((dim0_before, dim0_after), (dim1_before, dim1_after), ...)
            pad_width = ((0, 0),       # No padding for B dimension
                         (0, pad_L),   # Pad after in L dimension (dim 1)
                         (0, pad_W))   # Pad after in W dimension (dim 2)
                         
            padded_arr = np.pad(batch_arr, pad_width, mode='constant', constant_values=0.0)
            padded_batches.append(padded_arr)

        # 3. Concatenate all batches along B dimension (axis=0)
        return np.concatenate(padded_batches, axis=0)

    # ================================================================
    # Single model prediction
    # ================================================================
    def _predict_single(
        self, df: pd.DataFrame, 
        batch_size: int = 128, 
        return_probs: bool = True, 
        num_workers: int = 8
    ):
        self.model.eval()
        ds = TCRPepHLA_Dataset(df, self.phys_dict, self.esm2_dict, self.struct_dict, self.pep_hla_feat_dict)
        loader = torch.utils.data.DataLoader(
            ds, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=tcr_pep_hla_collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )

        preds = []
        fused_feat = []
        attn_all = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting (TCR-pHLA)"):
                probs, _, fused, attn_dict = self.model(batch)
                preds.extend(probs.tolist())
                fused_feat.append(fused)
                attn_all.append(attn_dict)
        
        # merge attn_all to a single dict of lists
        merged = {k: [] for k in attn_all[0].keys()}

        for ad in attn_all:
            for k, v in ad.items():
                merged[k].append(v)

        # padding + concat
        final_attn = {}
        for k, v_list in merged.items():
            final_attn[k] = self._pad_attention(v_list)


        if not return_probs:
            preds = (preds >= 0.5).astype(int)

        return preds, fused_feat, final_attn

    # ================================================================
    # Ensemble prediction
    # ================================================================
    def _predict_ensemble(
        self,
        df: pd.DataFrame,
        batch_size: int = 128,
        num_folds: int = 5,
        ensemble_method: str = 'mean',
        return_probs: bool = True,
        num_workers: int = 8
    ) -> np.ndarray:
        """
        Ensemble prediction using multiple fold models.
        """
        print(f"\nEnsemble prediction using {num_folds} TCR–pHLA models...")
        print(f"Ensemble method: {ensemble_method}")

        fused_feat_folds = []
        attn_dict_folds = []
        all_preds = []
        for fold_id in range(num_folds):
            fold_model_path = self.model_save_path.replace(".pt", f"_fold{fold_id}.pt")
            if not os.path.exists(fold_model_path):
                print(f"⚠ Warning: {fold_model_path} not found, skipping...")
                continue

            print(f"Loading model from {fold_model_path}...")
            self.model.load_state_dict(torch.load(fold_model_path, map_location=self.device), strict=False)

            # Predict for this fold
            fold_preds, fold_fused_feat, fold_attn_dict = self._predict_single(
                df, batch_size=batch_size, return_probs=True, num_workers=num_workers
            )
            all_preds.append(fold_preds)
            fused_feat_folds.append(fold_fused_feat)
            attn_dict_folds.append(fold_attn_dict)
            
        if len(all_preds) == 0:
            raise ValueError("No fold models found!")

        if ensemble_method == 'mean':
            ensemble_preds = np.mean(all_preds, axis=0)
        elif ensemble_method == 'median':
            ensemble_preds = np.median(all_preds, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")

        print(f"✓ Ensemble prediction completed using {len(all_preds)} folds")

        if not return_probs:
            ensemble_preds = (ensemble_preds >= 0.5).astype(int)

        return ensemble_preds, fused_feat_folds, attn_dict_folds


    # ================================================================
    # Unified predict() with ensemble support
    # ================================================================
    def predict(
        self,
        df: pd.DataFrame,
        batch_size: int = 128,
        return_probs: bool = True,
        use_kfold: bool = False,
        num_folds: Optional[int] = None,
        ensemble_method: str = 'mean',
        num_workers: int = 8
    ) -> Tuple[np.ndarray, List, List]:
        """
        Predict binding probabilities or binary labels.

        If use_kfold=True, averages predictions across fold models.
        """
        print('Preparing peptide-HLA features for prediction set...')
        self.prepare_pep_hla_features(df)

        if use_kfold:
            if num_folds is None:
                raise ValueError("num_folds must be specified when use_kfold=True")
            return self._predict_ensemble(
                df=df,
                batch_size=batch_size,
                num_folds=num_folds,
                ensemble_method=ensemble_method,
                return_probs=return_probs,
                num_workers=num_workers
            )
        else:
            return self._predict_single(df, batch_size=batch_size, return_probs=return_probs, num_workers=num_workers)


    # ================================================================
    # Unified evaluate() with ensemble support
    # ================================================================
    def evaluate(
        self,
        df: pd.DataFrame,
        batch_size: int = 128,
        threshold: float = 0.5,
        use_kfold: bool = False,
        num_folds: Optional[int] = None,
        ensemble_method: str = 'mean',
        num_workers: int = 8
    ) -> Dict[str, float]:
        """
        Evaluate model performance on a dataset.

        If use_kfold=True, performs ensemble evaluation across folds.
        """
        y_true = df['label'].values
        y_prob, _, _ = self.predict(
            df,
            batch_size=batch_size,
            return_probs=True,
            use_kfold=use_kfold,
            num_folds=num_folds,
            ensemble_method=ensemble_method,
            num_workers=num_workers
        )
        y_prob = np.array(y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
        accuracy = (tp + tn) / (tn + fp + fn + tp + 1e-9)
        try:
            mcc = ((tp*tn) - (fn*fp)) / np.sqrt(float((tp+fn)*(tn+fp)*(tp+fp)*(tn+fn)) + 1e-9)
        except:
            mcc = 0.0
        recall = tp / (tp + fn + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.0
        
        try:
            # prc
            from sklearn.metrics import average_precision_score
            prc_auc = average_precision_score(y_true, y_prob)
        except:
            prc_auc = 0.0

        print("\n" + "=" * 70)
        print(f"Evaluation Results [{'K-Fold Ensemble' if use_kfold else 'Single Model'}]")
        print("=" * 70)
        print(f"tn={tn}, fp={fp}, fn={fn}, tp={tp}")
        print(f"AUC={auc:.4f} | PRC={prc_auc:.4f} | ACC={accuracy:.4f} | MCC={mcc:.4f} | F1={f1:.4f} | P={precision:.4f} | R={recall:.4f}" )
        print("=" * 70 + "\n")

        return y_prob, dict(
            auc=auc, prc=prc_auc, accuracy=accuracy, mcc=mcc, f1=f1,
            precision=precision, recall=recall,
            tn=tn, fp=fp, fn=fn, tp=tp
        )