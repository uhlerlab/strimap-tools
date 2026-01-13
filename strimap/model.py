import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from torch.nn.utils.parametrizations import weight_norm
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import esm
import pandas as pd
from tqdm import tqdm

import tempfile
from pathlib import Path
import mdtraj as md
import os

from egnn_pytorch import EGNN
from transformers import AutoTokenizer, EsmForProteinFolding

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils import *
from dataset import *

class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0.0, save_path='checkpoint.pt'):
        """
        Early stopping based on both val_loss and val_auc.
        The model is saved whenever EITHER:
            - val_loss decreases by more than delta, OR
            - val_auc increases by more than delta.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self.save_path = save_path
        
        self.best_loss = np.inf
        self.best_auc = -np.inf

    def __call__(self, val_auc, model):
        improved = False
        
        # Check auc improvement
        if val_auc > self.best_auc + self.delta:
            self.best_auc = val_auc
            improved = True

        if improved:
            self.save_checkpoint(model, val_auc)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model, val_auc):
        """Save current best model."""
        if self.verbose:
            print(f"Validation improved → Saving model (Score={val_auc:.4f}) to {self.save_path}")
        torch.save(model.state_dict(), self.save_path)

# ============================================================================
# ESM2 Embedding via HuggingFace
# ============================================================================
class ESM2Encoder(nn.Module):
    def __init__(self, 
                 device="cuda:0", 
                 layer=33,
                 cache_dir='cache'):
        """
        Initialize an ESM2 encoder.

        Args:
            model_name (str): Name of the pretrained ESM2 model (e.g., 'esm2_t33_650M_UR50D').
            device (str): Device to run on, e.g. 'cuda:0', 'cuda:1', or 'cpu'.
            layer (int): Layer number from which to extract representations.
        """
        super().__init__()
        self.device = device
        self.layer = layer
        
        if cache_dir is None:
            cache_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model = self.model.eval().to(device)

    def _cache_path(self, prefix):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = base_dir + "/" + self.cache_dir
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, f"{prefix}_esm2_layer{self.layer}.pt")

    def save_obj(self, obj, path):
        """Save object to a file (no compression)."""
        torch.save(obj, path)

    def load_obj(self, path):
        """Load object from a file (no compression)."""
        return torch.load(path, map_location="cpu", weights_only=False)
    
    @torch.no_grad()
    def _embed_batch(self, batch_data):
        batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_data)
        batch_tokens = batch_tokens.to(self.device)
        results = self.model(batch_tokens, repr_layers=[self.layer], return_contacts=False)
        token_representations = results["representations"][self.layer]
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        seq_reprs = []
        for i, tokens_len in enumerate(batch_lens):
            seq_repr = token_representations[i, 1:tokens_len-1].cpu()
            seq_reprs.append(seq_repr)
        return seq_reprs

    @torch.no_grad()
    def forward(self, df, seq_col, prefix, batch_size=64, re_embed=False, cache_save=True):
        """
        Add or update embeddings for sequences in a DataFrame.
        - If there are new sequences, automatically update the dictionary and save.
        - If re_embed=True, force re-computation of all sequences.
        """
        cache_path = self._cache_path(prefix)
        emb_dict = {}

        if os.path.exists(cache_path) and not re_embed:
            print(f"[ESM2] Loading cached embeddings from {cache_path}")
            emb_dict = self.load_obj(cache_path)
        else:
            if re_embed:
                print(f"[ESM2] Re-embedding all sequences for {prefix}")
            else:
                print(f"[ESM2] No existing cache for {prefix}, will create new.")

        seqs = [str(s).strip().upper() for s in df[seq_col].tolist() if isinstance(s, str)]
        unique_seqs = sorted(set(seqs))
        new_seqs = [s for s in unique_seqs if s not in emb_dict]

        if new_seqs:
            print(f"[ESM2] Found {len(new_seqs)} new sequences → computing embeddings...")
            data = [(str(i), s) for i, s in enumerate(new_seqs)]
            for i in tqdm(range(0, len(data), batch_size), desc=f"ESM2 update ({prefix})"):
                batch = data[i:i+batch_size]
                embs = self._embed_batch(batch)
                for (_, seq), emb in zip(batch, embs):
                    emb_dict[seq] = emb.clone()
            if cache_save:
                print(f"[ESM2] Updating cache with new sequences")
                self.save_obj(emb_dict, cache_path)
        else:
            print(f"[ESM2] No new sequences for {prefix}, using existing cache")

        return emb_dict

# ============================================================================
# ESMFold (transformers)
# ============================================================================
class ESMFoldPredictorHF(nn.Module):
    def __init__(self, 
                 model_name="facebook/esmfold_v1", 
                 cache_dir=None, 
                 device='cpu', 
                 allow_tf32=True):
        super().__init__()
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        if allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # tokenizer and model
        print(f"Loading ESMFold model {model_name} on {device}... {'with' if cache_dir else 'without'} cache_dir: {cache_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = EsmForProteinFolding.from_pretrained(
            model_name, low_cpu_mem_usage=True, cache_dir=cache_dir
        ).eval().to(self.device)

    @torch.no_grad()
    def infer_pdb_str(self, seq: str) -> str:
        pdb_str = self.model.infer_pdb(seq)
        return pdb_str

    @torch.no_grad()
    def forward_raw(self, seq: str):
        inputs = self.tokenizer([seq], return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs  # ESMFoldOutput

MAX_ASA_TIEN = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}
SS8_INDEX = {"H":0,"B":1,"E":2,"G":3,"I":4,"T":5,"S":6,"C":7,"-":7}

class StructureFeatureExtractorNoDSSP(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

        self.in_dim = 6 + 8 + 1 + 1 + 1  # 17

        self.to(torch.device(self.device))

    @torch.no_grad()
    def _angles(self, traj):

        L = traj.n_residues

        sphi  = np.zeros(L, dtype=np.float32); cphi  = np.zeros(L, dtype=np.float32)
        spsi  = np.zeros(L, dtype=np.float32); cpsi  = np.zeros(L, dtype=np.float32)
        someg = np.zeros(L, dtype=np.float32); comeg = np.zeros(L, dtype=np.float32)

        # 1) phi: (C_{i-1}, N_i, CA_i, C_i) —— Current residue i can be located using atoms[1] (N_i)
        phi_idx, phi_vals = md.compute_phi(traj)          # phi_vals: (1, n_phi)
        if phi_vals.size > 0:
            for k, atoms in enumerate(phi_idx):
                res_i = traj.topology.atom(int(atoms[1])).residue.index  # N_i residue index
                if 0 <= res_i < L:
                    ang = float(phi_vals[0, k])
                    sphi[res_i] = np.sin(ang); cphi[res_i] = np.cos(ang)

        # 2) psi: (N_i, CA_i, C_i, N_{i+1}) —— Current residue i can be located using atoms[1] (CA_i)
        psi_idx, psi_vals = md.compute_psi(traj)
        if psi_vals.size > 0:
            for k, atoms in enumerate(psi_idx):
                res_i = traj.topology.atom(int(atoms[1])).residue.index  # CA_i
                if 0 <= res_i < L:
                    ang = float(psi_vals[0, k])
                    spsi[res_i] = np.sin(ang); cpsi[res_i] = np.cos(ang)

        # 3) omega: (CA_i, C_i, N_{i+1}, CA_{i+1}) —— Current residue i can be located using atoms[0] (CA_i)
        omg_idx, omg_vals = md.compute_omega(traj)
        if omg_vals.size > 0:
            for k, atoms in enumerate(omg_idx):
                res_i = traj.topology.atom(int(atoms[0])).residue.index  # CA_i
                if 0 <= res_i < L:
                    ang = float(omg_vals[0, k])
                    someg[res_i] = np.sin(ang); comeg[res_i] = np.cos(ang)

        angles_feat = np.stack([sphi, cphi, spsi, cpsi, someg, comeg], axis=-1)  # [L, 6]
        return angles_feat.astype(np.float32)

    @torch.no_grad()
    def _ss8(self, traj: md.Trajectory):
        ss = md.compute_dssp(traj, simplified=False)[0]
        L = traj.n_residues
        onehot = np.zeros((L, 8), dtype=np.float32)
        for i, ch in enumerate(ss):
            onehot[i, SS8_INDEX.get(ch, 7)] = 1.0
        return onehot

    @torch.no_grad()
    def _rsa(self, traj: md.Trajectory):
        asa = md.shrake_rupley(traj, mode="residue")[0]  # (L,)
        rsa = np.zeros_like(asa, dtype=np.float32)
        for i, res in enumerate(traj.topology.residues):
            max_asa = MAX_ASA_TIEN.get(res.name.upper(), None)
            rsa[i] = 0.0 if not max_asa else float(asa[i] / max_asa)
        return np.clip(rsa, 0.0, 1.0)[:, None]

    @torch.no_grad()
    def _contact_count(self, traj: md.Trajectory, cutoff_nm=0.8):
        L = traj.n_residues
        ca_atoms = traj.topology.select("name CA")
        if len(ca_atoms) == L:
            coors = traj.xyz[0, ca_atoms, :]  # nm
        else:
            xyz = traj.xyz[0]
            coors = []
            for res in traj.topology.residues:
                idxs = [a.index for a in res.atoms]
                coors.append(xyz[idxs, :].mean(axis=0))
            coors = np.array(coors, dtype=np.float32)
        diff = coors[:, None, :] - coors[None, :, :]
        dist = np.sqrt((diff**2).sum(-1))  # nm
        mask = (dist < cutoff_nm).astype(np.float32)
        np.fill_diagonal(mask, 0.0)
        cnt = mask.sum(axis=1)
        return cnt[:, None].astype(np.float32)

    @torch.no_grad()
    def _plddt(self, pdb_file: str):
        # Use Biopython to read PDB B-factor (ESMFold/AlphaFold writes pLDDT here)
        from Bio.PDB import PDBParser
        import numpy as np

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("prot", pdb_file)
        model = structure[0]

        res_plddt = []
        for chain in model:
            for residue in chain:
                atoms = list(residue.get_atoms())
                if len(atoms) == 0:
                    res_plddt.append(0.0)
                    continue
                # Mean B-factor of atoms in the residue
                bvals = [float(atom.get_bfactor()) for atom in atoms]
                res_plddt.append(float(np.mean(bvals)))

        # Normalize to [0,1]
        plddt = np.array(res_plddt, dtype=np.float32) / 100.0
        plddt = np.clip(plddt, 0.0, 1.0)
        return plddt[:, None]  # [L,1]

    @torch.no_grad()
    def _parse_and_features(self, pdb_file: str):
        traj = md.load(pdb_file)
        L = traj.n_residues

        angles = self._angles(traj)              # [L,6]
        ss8    = self._ss8(traj)                 # [L,8]
        rsa    = self._rsa(traj)                 # [L,1]
        cnt    = self._contact_count(traj)       # [L,1]
        plddt  = self._plddt(pdb_file)           # [L,1]

        feats = np.concatenate([angles, ss8, rsa, cnt, plddt], axis=1).astype(np.float32)  # [L,17]

        ca_atoms = traj.topology.select("name CA")
        if len(ca_atoms) == L:
            coors_nm = traj.xyz[0, ca_atoms, :]
        else:
            xyz = traj.xyz[0]
            res_coords = []
            for res in traj.topology.residues:
                idxs = [a.index for a in res.atoms]
                res_coords.append(xyz[idxs, :].mean(axis=0))
            coors_nm = np.array(res_coords, dtype=np.float32)
        coors_ang = coors_nm * 10.0  # nm -> Å
        return coors_ang.astype(np.float32), feats  # [L,3], [L,17]

    @torch.no_grad()
    def forward(self, pdb_file: str):
        coors_ang, scalars = self._parse_and_features(pdb_file)
        coors = torch.tensor(coors_ang, dtype=torch.float32, device=self.device)   # [N,3]
        scalars = torch.tensor(scalars,  dtype=torch.float32, device=self.device)    # [N,17]

        return scalars, coors  # [N,17], [N,3]

import uuid
class ResiduePipelineWithHFESM:
    def __init__(self, 
                 esm_model_name="facebook/esmfold_v1",
                 cache_dir=None,
                 esm_device='cpu',
                 allow_tf32=True
                 ):
        self.esm = ESMFoldPredictorHF(esm_model_name, cache_dir, esm_device, allow_tf32)
        self.struct_encoder = StructureFeatureExtractorNoDSSP(device=esm_device)
        self.cache_dir = cache_dir

    @torch.no_grad()
    # def __call__(self, seq: str, save_pdb_path: str = None) -> torch.Tensor:
    #     pdb_str = self.esm.infer_pdb_str(seq)
    #     if save_pdb_path is None:
    #         tmpdir = self.cache_dir if self.cache_dir is not None else tempfile.gettempdir()
    #         save_pdb_path = str(Path(tmpdir) / "esmfold_pred_fold15.pdb")
    #     Path(save_pdb_path).write_text(pdb_str)

    #     struct_emb, struct_coords = self.struct_encoder(save_pdb_path)
    #     return struct_emb, struct_coords 
    def __call__(self, seq: str, save_pdb_path: str = None) -> torch.Tensor:
        pdb_str = self.esm.infer_pdb_str(seq)
        
        # Temporary file cleanup logic
        created_temp = False
        
        if save_pdb_path is None:
            tmpdir = self.cache_dir if self.cache_dir is not None else tempfile.gettempdir()
            # use uuid to generate unique filename to prevent concurrency conflicts
            unique_name = f"esmfold_{uuid.uuid4().hex}.pdb"
            save_pdb_path = str(Path(tmpdir) / unique_name)
            created_temp = True

        Path(save_pdb_path).write_text(pdb_str)
        struct_emb, struct_coords = self.struct_encoder(save_pdb_path)
        if created_temp and os.path.exists(save_pdb_path):
            os.remove(save_pdb_path)
            
        return struct_emb, struct_coords

def sanitize_protein_seq(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    s = "".join(seq.split()).upper()
    allowed = set("ACDEFGHIKLMNPQRSTVWYXBZJUO")
    return "".join([c for c in s if c in allowed])

@torch.no_grad()
def batch_embed_to_dicts(
    df: pd.DataFrame,
    seq_col: str,
    pipeline,
    show_progress: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[Tuple[str, str]]]:
    """
    Returns:
      - emb_dict:  {seq -> z(torch.Tensor[L, D])}
      - coord_dict:{seq -> coords(torch.Tensor[L, 3])}
      - failures:  [(seq, err_msg), ...]
    """

    raw_list = df[seq_col].astype(str).tolist()
    seqs = []
    for s in raw_list:
        ss = sanitize_protein_seq(s)
        if ss:
            seqs.append(ss)
    uniq_seqs = sorted(set(seqs)) 

    logger.info(f"Total rows: {len(df)}, valid seqs: {len(seqs)}, unique: {len(uniq_seqs)}")

    emb_dict: Dict[str, torch.Tensor] = {}
    coord_dict: Dict[str, torch.Tensor] = {}
    failures: List[Tuple[str, str]] = []

    iterator = tqdm(uniq_seqs, desc="ESMfold Predicting structure...") if show_progress else uniq_seqs
    for seq in tqdm(iterator):
        if seq in emb_dict:
            continue
        try:
            z_t, c_t = pipeline(seq)      # z: [L, D], coords: [L, 3] (torch.Tensor)
            emb_dict[seq] = z_t.detach().float().cpu()
            coord_dict[seq] = c_t.detach().float().cpu()
        except Exception as e:
            failures.append((seq, repr(e)))
            continue

    logger.info(f"[DONE] OK: {len(emb_dict)}, Failed: {len(failures)}")
    if failures[:3]:
        logger.error("[SAMPLE failures]", failures[:3])
    return emb_dict, coord_dict, failures

class ESMFoldEncoder(nn.Module):
    def __init__(self, model_name="facebook/esmfold_v1", esm_cache_dir="esm_cache", cache_dir="cache"):
        super(ESMFoldEncoder, self).__init__()
        self.model_name = model_name
        self.esm_cache_dir = esm_cache_dir
        self.cache_dir = cache_dir
    
    def save_obj(self, obj, path):
        """Save object to a file (no compression)."""
        torch.save(obj, path)
        
    def load_obj(self, path):
        """Load object from a file (no compression)."""
        return torch.load(path, map_location='cpu', weights_only=False)

    def load_esm_dict(self, device, df_data, chain, re_embed, cache_save):

        def _clean_unique(series: pd.Series) -> list:
            cleaned = []
            for s in series.astype(str).tolist():
                ss = sanitize_protein_seq(s)
                if ss:
                    cleaned.append(ss)
            return sorted(set(cleaned))
        
        def _retry_embed_df(
            df: pd.DataFrame,
            chain: str,
            max_retries: int = 2,
            show_progress: bool = True,
        ):
            """
            Try to embed protein sequences with retries on failures.

            Args:
                df (pd.DataFrame): A DataFrame containing a column `chain` with sequences.
                chain (str): The column name containing the sequences (e.g., "alpha", "beta").
                pipeline: An embedding pipeline, should return (embedding, coords) for a sequence.
                max_retries (int): Maximum number of retries for failed sequences.
                show_progress (bool): Whether to display tqdm progress bars.

            Returns:
                feat_dict (Dict[str, torch.Tensor]): {sequence -> embedding tensor [L, D]}.
                coord_dict (Dict[str, torch.Tensor]): {sequence -> coordinate tensor [L, 3]}.
                failures (List[Tuple[str, str]]): List of (sequence, error_message) that still failed after retries.
            """
            
            pipeline = ResiduePipelineWithHFESM(
                esm_model_name=self.model_name,
                cache_dir=self.esm_cache_dir,
                esm_device=device
            )
        
            # 1. First attempt
            feat_dict, coord_dict, failures = batch_embed_to_dicts(
                df, chain, pipeline, show_progress=show_progress
            )

            # 2. Retry loop for failed sequences
            tries = 0
            while failures and tries < max_retries:
                tries += 1
                retry_seqs = [s for s, _ in failures]
                logger.info(f"[retry {tries}/{max_retries}] {len(retry_seqs)} sequences")
                retry_df = pd.DataFrame({chain: retry_seqs})

                f2, c2, failures = batch_embed_to_dicts(
                    retry_df, chain, pipeline, show_progress=show_progress
                )
                feat_dict.update(f2)
                coord_dict.update(c2)

            return feat_dict, coord_dict, failures

        def update_with_new_seqs(feat_dict, coord_dict, chain):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = base_dir + "/" + self.cache_dir
            os.makedirs(base_dir, exist_ok=True)
            path_feat = os.path.join(base_dir, f"{chain}_feat_dict.pt")
            path_coords = os.path.join(base_dir, f"{chain}_coord_dict.pt")

            all_seqs_clean = _clean_unique(df_data[chain])
            new_seqs = [s for s in all_seqs_clean if s not in feat_dict]
            if not new_seqs:
                logger.info(f"No new {chain} sequences found")
                return feat_dict, coord_dict

            logger.info(f"Found new {chain} sequences, embedding...")
            df_new = pd.DataFrame({chain: new_seqs})
            new_feat_dict, new_coord_dict, failures = _retry_embed_df(df_new, chain, max_retries=100)
            feat_dict.update(new_feat_dict)
            coord_dict.update(new_coord_dict)
            if cache_save:
                self.save_obj(feat_dict, path_feat)
                self.save_obj(coord_dict, path_coords)
                logger.info(f"Updated and saved {path_feat} and {path_coords}")
                
            if failures:
                for seq, err in failures:
                    logger.error(f"[create] failed: {seq} | {err}")
            
            return feat_dict, coord_dict

        def get_or_create_dict(chain):
            base_dir = os.path.dirname(os.path.abspath(__file__)) + "/" + self.cache_dir
            os.makedirs(base_dir, exist_ok=True)
            path_feat = os.path.join(base_dir, f"{chain}_feat_dict.pt")
            path_coords = os.path.join(base_dir, f"{chain}_coord_dict.pt")
            
            if os.path.exists(path_feat) and not re_embed:
                logger.info(f"Loading {path_feat} and {path_coords}")
                feat_dict = self.load_obj(path_feat)
                coord_dict = self.load_obj(path_coords)
            else:
                logger.info(f"{path_feat} and {path_coords} not found or re_embed=True, generating...")
                unique_seqs = _clean_unique(df_data[chain])
                df_uniq = pd.DataFrame({chain: unique_seqs})
                feat_dict, coord_dict, failures = _retry_embed_df(
                    df_uniq, chain, show_progress=True, max_retries=100
                )
            if cache_save:
                self.save_obj(feat_dict, path_feat)
                self.save_obj(coord_dict, path_coords)
                logger.info(f"Saved {path_feat} and {path_coords}")

                if failures:
                    for seq, err in failures:
                        logger.error(f"[create] failed: {seq} | {err}")

            return feat_dict, coord_dict

        self.dict[chain+'_feat'], self.dict[chain+'_coord'] = update_with_new_seqs(*get_or_create_dict(chain), chain)

    def pad_and_stack(self, batch_feats, L_max, batch_coors):
        """
        batch_feats: list of [L_i, D] tensors
        batch_coors: list of [L_i, 3] tensors
        return:
        feats: [B, L_max, D]
        coors: [B, L_max, 3]
        mask : [B, L_max]  (True for real tokens)
        """
        assert len(batch_feats) == len(batch_coors)
        B = len(batch_feats)
        D = batch_feats[0].shape[-1]

        feats_pad = []
        coors_pad = []
        masks = []

        for x, c in zip(batch_feats, batch_coors):
            L = x.shape[0]
            pad_L = L_max - L
            # pad feats/coors with zeros
            feats_pad.append(torch.nn.functional.pad(x, (0, 0, 0, pad_L)))       # [L_max, D]
            coors_pad.append(torch.nn.functional.pad(c, (0, 0, 0, pad_L)))       # [L_max, 3]
            m = torch.zeros(L_max, dtype=torch.bool)
            m[:L] = True
            masks.append(m)

        feats = torch.stack(feats_pad, dim=0)   # [B, L_max, D]
        coors = torch.stack(coors_pad, dim=0)   # [B, L_max, 3]
        mask  = torch.stack(masks, dim=0)       # [B, L_max]
        return feats, coors, mask
    
    def forward(self, df_data, chain, device='cpu', re_embed=False, cache_save=False):
        """
        df_data: pd.DataFrame with a column `chain` containing sequences
        chain: str, e.g. "alpha" or "beta"
        device: str, e.g. 'cpu' or 'cuda:0'
        re_embed: bool, whether to re-embed even if cached files exist
        """
        self.dict = {}
        self.load_esm_dict(device, df_data, chain, re_embed, cache_save)

        batch_feats = []
        batch_coors = []
        for seq in df_data[chain].astype(str).tolist():
            ss = sanitize_protein_seq(seq)
            if ss in self.dict[chain+'_feat'] and ss in self.dict[chain+'_coord']:
                batch_feats.append(self.dict[chain+'_feat'][ss])
                batch_coors.append(self.dict[chain+'_coord'][ss])
            else:
                raise ValueError(f"Sequence not found in embedding dict: {ss}")

        # L_max = max(x.shape[0] for x in batch_feats)

        return batch_feats, batch_coors

class ResidueProjector(nn.Module):
    """Align channel dimensions of different branches to the same D"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    def forward(self, x):  # x: [B,L,Di]
        return self.proj(x)

class ResidueDoubleFusion(nn.Module):
    """
    ResidueDoubleFusion:
    A residue-level two-branch fusion module that combines two modalities (x1, x2)
    using cross-attention followed by gated residual fusion and linear projection.

    Typical usage:
        - x1: physicochemical features
        - x2: ESM embeddings  (or structure features)
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim

        # Cross-attention: allows information flow between two modalities
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Gating mechanism: adaptively weight two modalities per residue
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        # Optional projection after fusion
        self.out_proj = nn.Linear(dim, dim)

        # Layer norms for stable training
        self.norm_x1 = nn.LayerNorm(dim)
        self.norm_x2 = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        """
        Args:
            x1: Tensor [B, L, D] - first modality (e.g., physicochemical)
            x2: Tensor [B, L, D] - second modality (e.g., ESM embeddings)
        Returns:
            fused: Tensor [B, L, D] - fused residue-level representation
        """

        # 1) Normalize both branches
        x1_norm = self.norm_x1(x1)
        x2_norm = self.norm_x2(x2)

        # 2) Cross-attention (x1 queries, x2 keys/values)
        #    This allows x1 to attend to x2 at each residue position
        attn_out, _ = self.cross_attn(
            query=x1_norm,
            key=x2_norm,
            value=x2_norm
        )  # [B, L, D]

        # 3) Gating between original x1 and attention-enhanced x2
        gate_val = self.gate(torch.cat([x1, attn_out], dim=-1))  # [B, L, 1]
        fused = gate_val * x1 + (1 - gate_val) * attn_out

        # 4) Optional projection + normalization
        fused = self.out_proj(fused)
        fused = self.norm_out(fused)

        return fused

class ResidueTripleFusion(nn.Module):
    """
    ResidueTripleFusion:
    A hierarchical three-branch feature fusion module for residue-level representations.
    
    Step 1: Fuse physicochemical features and protein language model embeddings.
    Step 2: Fuse the intermediate representation with structure-based features.
    
    Each fusion step uses ResidueDoubleFusion (cross-attention + gating + linear projection).
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        # Fuse physicochemical + ESM embeddings
        self.fuse_phys_esm = ResidueDoubleFusion(dim, num_heads=num_heads, dropout=dropout)
        # Fuse the fused phys+esm representation with structure embeddings
        self.fuse_f12_struct = ResidueDoubleFusion(dim, num_heads=num_heads, dropout=dropout)

    def forward(self, phys, esm, struct):
        """
        Args:
            phys:   Tensor [B, L, D], physicochemical features (e.g., AAindex-based)
            esm:    Tensor [B, L, D], protein language model embeddings (e.g., ESM2, ProtT5)
            struct: Tensor [B, L, D], structure-derived features (e.g., torsion, RSA)
        
        Returns:
            fused:  Tensor [B, L, D], final fused representation
        """
        # Step 1: Fuse physicochemical and ESM embeddings
        f12 = self.fuse_phys_esm(phys, esm)

        # Step 2: Fuse the intermediate fused representation with structure features
        fused = self.fuse_f12_struct(f12, struct)

        return fused

class BANLayer(nn.Module):
    """
    Bilinear Attention Network Layer with proper 2D masked-softmax.
    v_mask: [B, L_v]  True=valid
    q_mask: [B, L_q]  True=valid
    """
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super().__init__()
        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)

        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat  = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):  # att_map: [B, L_v, L_q]
        logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            logits = self.p_net(logits.unsqueeze(1)).squeeze(1) * self.k
        return logits

    def _masked_softmax_2d(self, logits, v_mask, q_mask):
        """
        logits:  [B, h_out, L_v, L_q]
        v_mask:  [B, L_v]  or None
        q_mask:  [B, L_q]  or None
        return:  probs  [B, h_out, L_v, L_q]  (masked entries=0, 在有效的二维子矩阵内归一化)
        """
        B, H, Lv, Lq = logits.shape
        device = logits.device
        if v_mask is None:
            v_mask = torch.ones(B, Lv, dtype=torch.bool, device=device)
        if q_mask is None:
            q_mask = torch.ones(B, Lq, dtype=torch.bool, device=device)

        mask2d = (v_mask[:, :, None] & q_mask[:, None, :])          # [B, Lv, Lq]
        mask2d = mask2d[:, None, :, :].expand(B, H, Lv, Lq)         # [B, H, Lv, Lq]

        logits = logits.masked_fill(~mask2d, -float('inf'))

        # Perform softmax over the joint Lv*Lq space
        flat = logits.view(B, H, -1)                                # [B, H, Lv*Lq]
        # Handle extreme cases: some samples may have no valid cells, avoid NaN
        flat = torch.where(torch.isinf(flat), torch.full_like(flat, -1e9), flat)
        flat = F.softmax(flat, dim=-1)
        flat = torch.nan_to_num(flat, nan=0.0)                      # Safety fallback
        probs = flat.view(B, H, Lv, Lq)

        # Zero out masked positions (for numerical stability & easier visualization)
        probs = probs * mask2d.float()
        return probs

    def forward(self, v, q, v_mask=None, q_mask=None, softmax=True):
        """
        v: [B, L_v, Dv], q: [B, L_q, Dq]
        """
        B, L_v, _ = v.size()
        _, L_q, _ = q.size()

        v_ = self.v_net(v)   # [B, L_v, h_dim*k]
        q_ = self.q_net(q)   # [B, L_q, h_dim*k]

        if self.h_out <= self.c:
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias   # [B,H,Lv,Lq]
        else:
            v_t = v_.transpose(1, 2).unsqueeze(3)                      # [B, K, Lv, 1]
            q_t = q_.transpose(1, 2).unsqueeze(2)                      # [B, K, 1, Lq]
            d_  = torch.matmul(v_t, q_t)                               # [B, K, Lv, Lq]
            att_maps = self.h_net(d_.permute(0, 2, 3, 1))              # [B, Lv, Lq, H]
            att_maps = att_maps.permute(0, 3, 1, 2)                    # [B, H, Lv, Lq]

        if softmax:
            att_maps = self._masked_softmax_2d(att_maps, v_mask, q_mask)
        else:
            # Even if not softmax, zero out invalid cells to prevent leakage
            if v_mask is not None:
                att_maps = att_maps.masked_fill(~v_mask[:, None, :, None], 0.0)
            if q_mask is not None:
                att_maps = att_maps.masked_fill(~q_mask[:, None, None, :], 0.0)

        # Note: at this point v_ / q_ are still [B, L, K], aligned with att_maps [B,H,Lv,Lq]
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits = logits + self.attention_pooling(v_, q_, att_maps[:, i, :, :])

        logits = self.bn(logits)
        return logits, att_maps

class FCNet(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0.2):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class StackedEGNN(nn.Module):
    def __init__(self, dim, layers, update_coors=False, **egnn_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            EGNN(dim=dim, update_coors=update_coors, **egnn_kwargs)
            for _ in range(layers)
        ])

    def forward(self, feats, coors, mask=None):
        # feats: [B, L_max, D], coors: [B, L_max, 3], mask: [B, L_max] (bool)
        for layer in self.layers:
            feats, coors = layer(feats, coors, mask=mask)
        return feats, coors

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)

        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_weight * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
        
# ===================================== Main Model (Full Version) ===========================================
class PeptideHLABindingPredictor(nn.Module):
    def __init__(
        self,
        phys_dim=20,               # Dimension of Physicochemical features
        pep_dim=256,              # Unified peptide channel dimension
        hla_dim=256,              # Unified HLA channel dimension
        bilinear_dim=256,
        pseudo_seq_pos=None,      # Pocket positions (assumed 0-based and within [0,179])
        device="cuda:0",
        loss_fn='bce',
        alpha=0.5,
        gamma=2.0,
        dropout=0.2,
        pos_weights=None
    ):
        super().__init__()
        self.device = device
        self.pep_dim = pep_dim
        self.hla_dim = hla_dim
        self.bilinear_dim = bilinear_dim
        self.alpha = alpha
        self.gamma = gamma
        self.dropout = dropout
        if loss_fn == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weights]) if pos_weights is not None else None)
        elif loss_fn == 'focal':
            self.loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        self.se3_model = StackedEGNN(
            dim=17, layers=3
        )
        
        self.max_pep_len = 20  
        self.max_hla_len = 180 
        
        self.pep_pos_embed = nn.Parameter(torch.randn(self.max_pep_len, pep_dim))
        self.hla_pos_embed = nn.Parameter(torch.randn(self.max_hla_len, hla_dim))

        # —— Branch projection to unified dimension (per residue) ——
        # peptide branch (Physicochem -> pep_dim, ESM2(1280) -> pep_dim)
        self.proj_pep_phys = ResidueProjector(in_dim=phys_dim, out_dim=pep_dim)  # Your PhysEnc output dim set to pep_dim
        self.proj_pep_esm  = ResidueProjector(in_dim=1280, out_dim=pep_dim)

        # HLA branch (Physicochem -> hla_dim, ESM2(1280) -> hla_dim, Struct(17/or se3_out) -> hla_dim)
        self.proj_hla_phys = ResidueProjector(in_dim=phys_dim, out_dim=hla_dim)  # Your PhysEnc output dim set to hla_dim
        self.proj_hla_esm  = ResidueProjector(in_dim=1280, out_dim=hla_dim)
        self.proj_hla_se3  = ResidueProjector(in_dim=17, out_dim=hla_dim)  # Let se3_model output dim be hla_dim

        # —— Gate fusion (per residue) ——
        self.gate_pep = ResidueDoubleFusion(pep_dim)   # pep_phys × pep_esm
        self.gate_hla = ResidueTripleFusion(hla_dim)  # hla_phys × hla_esm × hla_struct
        
        d_model = self.pep_dim
        n_heads = 8
        
        # 1. For "Peptide queries HLA" (pep_q_hla_kv)
        self.cross_attn_pep_hla = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            dropout=self.dropout, 
            batch_first=True
        )
        self.norm_cross_pep = nn.LayerNorm(d_model)

        # 2. For "HLA queries Peptide" (hla_q_pep_kv)
        self.cross_attn_hla_pep = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            dropout=self.dropout, 
            batch_first=True
        )
        self.norm_cross_hla = nn.LayerNorm(d_model)

        # —— Interaction module (Bilinear attention map) ——
        self.bi_attn = BANLayer(v_dim=pep_dim, q_dim=hla_dim, h_dim=bilinear_dim, h_out=4, k=3)

        # —— Head —— 
        self.head = nn.Sequential(
            nn.Linear(bilinear_dim, bilinear_dim),
            nn.ReLU(),
            nn.Linear(bilinear_dim, 1)
        )

        # —— Pocket positions —— 
        if pseudo_seq_pos is None:
            pseudo_seq_pos = [i-2 for i in [7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171]]
        self.register_buffer("contact_idx", torch.tensor(pseudo_seq_pos, dtype=torch.long))

        # --------------------------------------------
        # Transformer Encoders for peptide & HLA
        # --------------------------------------------
        encoder_layer_pep = TransformerEncoderLayer(
            d_model=pep_dim,        # Input dimension
            nhead=8,               # Number of attention heads (tunable)
            dim_feedforward=pep_dim*4,
            dropout=self.dropout,
            batch_first=True       # Input shape [B,L,D]
        )
        self.pep_encoder = TransformerEncoder(encoder_layer_pep, num_layers=2)

        encoder_layer_hla = TransformerEncoderLayer(
            d_model=hla_dim,
            nhead=8,
            dim_feedforward=hla_dim*4,
            dropout=self.dropout,
            batch_first=True
        )
        self.hla_encoder = TransformerEncoder(encoder_layer_hla, num_layers=1)

    # -------------------------- Utility: pad list of [L,D] to [B,L_max,D] --------------------------
    def _pad_stack(self, tensors, L_max=None):
        Ls = [t.shape[0] for t in tensors]
        if L_max is None: L_max = max(Ls)
        D = tensors[0].shape[-1]
        B = len(tensors)
        out = tensors[0].new_zeros((B, L_max, D))
        mask = torch.zeros(B, L_max, dtype=torch.bool, device=out.device)
        for i, t in enumerate(tensors):
            L = t.shape[0]
            out[i, :L] = t
            mask[i, :L] = True
        return out, mask  # [B,L_max,D], [B,L_max]
    
    def _mask_to_pockets(self, hla_feat):
        """
        From HLA features, keep only pocket positions, return:
        - hla_pocket: [B, n_pocket, D]
        - pocket_mask: [B, n_pocket] (all True)
        """
        B, L, D = hla_feat.shape

        # ensure idx in [0, L-1]
        idx = self.contact_idx.clamp(min=0, max=L-1)
        # gather pocket features
        hla_pocket = hla_feat[:, idx, :]     # [B, n_pocket, D]

        return hla_pocket
    
    def add_positional_encoding(self, x, pos_embed):
        """
        x: [B, L, D]
        pos_embed: [L_max, D]
        """
        B, L, D = x.shape
        # Take the first L positional encodings
        pe = pos_embed[:L, :].unsqueeze(0).expand(B, -1, -1)  # [B, L, D]
        return x + pe

    def forward(self, batch):
        # take batch from DataLoader
        pep_phys = batch['pep_phys'].to(self.device, non_blocking=True)
        pep_esm  = batch['pep_esm'].to(self.device, non_blocking=True)
        hla_phys = batch['hla_phys'].to(self.device, non_blocking=True)
        hla_esm  = batch['hla_esm'].to(self.device, non_blocking=True)
        hla_struct = batch['hla_struct'].to(self.device, non_blocking=True)
        hla_coord  = batch['hla_coord'].to(self.device, non_blocking=True)
        labels = batch['label'].to(self.device)

        # 1) peptide physicochemical + ESM2 → gate fusion
        pep_phys = self.proj_pep_phys(pep_phys)
        pep_esm  = self.proj_pep_esm(pep_esm)
        pep_feat = self.gate_pep(pep_phys, pep_esm)   # [B, Lp, D]
        
        pep_feat = self.add_positional_encoding(pep_feat, self.pep_pos_embed)
        pep_feat = self.pep_encoder(pep_feat, src_key_padding_mask=~batch['pep_mask'].to(self.device, non_blocking=True))
        
        # 2) HLA physicochemical + ESM2 + structure → SE3 → gate fusion
        hla_phys = self.proj_hla_phys(hla_phys)
        hla_esm  = self.proj_hla_esm(hla_esm)
        # hla_struct is [B, 180, 17], first pass through SE3
        hla_se3 = self.se3_model(hla_struct, hla_coord, None)[0]  # [B, 180, 17]
        hla_se3 = self.proj_hla_se3(hla_se3) # →256
        hla_feat = self.gate_hla(hla_phys, hla_esm, hla_se3)
        
        hla_feat = self.add_positional_encoding(hla_feat, self.hla_pos_embed)
        hla_feat = self.hla_encoder(hla_feat)
        
        # cross attention for pep
        pep_feat_cross, _ = self.cross_attn_pep_hla(
            query=pep_feat,
            key=hla_feat,
            value=hla_feat,
            key_padding_mask=None
        )

        # cross attention for hla
        hla_feat_cross, _ = self.cross_attn_hla_pep(
            query=hla_feat,
            key=pep_feat,
            value=pep_feat,
            key_padding_mask=~batch['pep_mask'].to(self.device, non_blocking=True)
        )
        
        pep_feat_updated = self.norm_cross_pep(pep_feat + pep_feat_cross)
        hla_feat_updated = self.norm_cross_hla(hla_feat + hla_feat_cross)

        # 3) mask HLA pocket positions
        hla_pocket = self._mask_to_pockets(hla_feat_updated)

        # 4) bilinear attention
        fused_vec, attn = self.bi_attn(
            pep_feat_updated,
            hla_pocket,
            v_mask=batch['pep_mask'].to(self.device, non_blocking=True),
            q_mask=None
        )
        logits = self.head(fused_vec).squeeze(-1)
        
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        binding_loss = self.loss_fn(logits, labels.float())

        return probs, binding_loss, attn.detach().cpu().numpy().sum(axis=1), fused_vec.detach().cpu().numpy()

    # -------------------------- Encoder reuse interface (for TCR-HLA model) --------------------------
    def _pad_peptide(self, x, max_len):
        """Pad peptide feature tensor [1, L, D] to [1, max_len, D]."""
        B, L, D = x.shape
        if L < max_len:
            pad = x.new_zeros(B, max_len - L, D)
            return torch.cat([x, pad], dim=1)
        else:
            return x[:, :max_len, :]
        
    @torch.no_grad()
    def encode_peptide_hla(self, pep_id, pep_phys, pep_esm, hla_phys, hla_esm, hla_struct, hla_coord, max_pep_len):
        Lp = len(pep_id)
                
        pep_phys = self.proj_pep_phys(pep_phys)
        pep_esm  = self.proj_pep_esm(pep_esm)
        
        pep_phys = self._pad_peptide(pep_phys, max_pep_len)
        pep_esm  = self._pad_peptide(pep_esm,  max_pep_len)
    
        device = pep_phys.device
        pep_mask = torch.zeros(1, max_pep_len, dtype=torch.bool, device=device)
        pep_mask[0, :Lp] = True

        pep_feat = self.gate_pep(pep_phys, pep_esm)
        pep_feat = self.add_positional_encoding(pep_feat, self.pep_pos_embed)
        pep_feat = self.pep_encoder(pep_feat, src_key_padding_mask=~pep_mask)

        # 2) hla encoding
        hla_phys = self.proj_hla_phys(hla_phys)
        hla_esm  = self.proj_hla_esm(hla_esm)
        hla_se3  = self.se3_model(hla_struct, hla_coord, None)[0]
        hla_se3  = self.proj_hla_se3(hla_se3)
        hla_feat = self.gate_hla(hla_phys, hla_esm, hla_se3)
        hla_feat = self.add_positional_encoding(hla_feat, self.hla_pos_embed)
        hla_feat = self.hla_encoder(hla_feat)

        # --- 3a. Peptide (Q) queries HLA (K, V) ---
        pep_feat_cross, _ = self.cross_attn_pep_hla(
            query=pep_feat,
            key=hla_feat,
            value=hla_feat,
            key_padding_mask=None
        )
        pep_feat_updated = self.norm_cross_pep(pep_feat + pep_feat_cross)
        
        # --- 3b. HLA (Q) queries Peptide (K, V) ---
        hla_feat_cross, _ = self.cross_attn_hla_pep(
            query=hla_feat,
            key=pep_feat,
            value=pep_feat,
            key_padding_mask=~pep_mask
        )
        hla_feat_updated = self.norm_cross_hla(hla_feat + hla_feat_cross)

        return pep_feat_updated, hla_feat_updated

    @torch.no_grad()
    def encode_peptide_hla_batch(self, batch_dict: dict, max_pep_len: int):
        """
        (Refactored) 
        Run the pHLA encoder model on a single batch.
        """
        # 1. Get data from batch (assumed padded)
        pep_phys = batch_dict['pep_phys'].to(self.device)
        pep_esm = batch_dict['pep_esm'].to(self.device)
        hla_phys = batch_dict['hla_phys'].to(self.device)
        hla_esm = batch_dict['hla_esm'].to(self.device)
        hla_struct = batch_dict['hla_struct'].to(self.device)
        hla_coord = batch_dict['hla_coord'].to(self.device)
        pep_lens = batch_dict['pep_lens'].to(self.device) # [B]
        
        B = pep_phys.shape[0]
        device = pep_phys.device

        # 2. Peptide encoding (batch)
        pep_phys = self.proj_pep_phys(pep_phys)
        pep_esm  = self.proj_pep_esm(pep_esm)

        # Key: create batch mask
        pep_mask = torch.arange(max_pep_len, device=device).unsqueeze(0).expand(B, -1) < pep_lens.unsqueeze(1)
        
        pep_feat = self.gate_pep(pep_phys, pep_esm)
        pep_feat = self.add_positional_encoding(pep_feat, self.pep_pos_embed)
        pep_feat = self.pep_encoder(pep_feat, src_key_padding_mask=~pep_mask)

        # 3. HLA encoding (model itself is batch)
        hla_phys = self.proj_hla_phys(hla_phys)
        hla_esm  = self.proj_hla_esm(hla_esm)
        hla_se3  = self.se3_model(hla_struct, hla_coord, None)[0]
        hla_se3  = self.proj_hla_se3(hla_se3)
        hla_feat = self.gate_hla(hla_phys, hla_esm, hla_se3)
        hla_feat = self.add_positional_encoding(hla_feat, self.hla_pos_embed)
        hla_feat = self.hla_encoder(hla_feat)

        # 4. Cross-attention (model itself is batch)
        pep_feat_cross, _ = self.cross_attn_pep_hla(
            query=pep_feat, key=hla_feat, value=hla_feat, key_padding_mask=None
        )
        pep_feat_updated = self.norm_cross_pep(pep_feat + pep_feat_cross)
        
        hla_feat_cross, _ = self.cross_attn_hla_pep(
            query=hla_feat, key=pep_feat, value=pep_feat, key_padding_mask=~pep_mask
        )
        hla_feat_updated = self.norm_cross_hla(hla_feat + hla_feat_cross)

        return pep_feat_updated, hla_feat_updated

class TCRPeptideHLABindingPredictor(nn.Module):
    def __init__(
            self, 
            tcr_dim=256, 
            pep_dim=256, 
            hla_dim=256, 
            bilinear_dim=256, 
            loss_fn='bce',
            alpha=0.5,
            gamma=2.0,
            dropout=0.1,
            device='cuda:0',
            pos_weights=None,
            use_struct=True
        ):
        super().__init__()
        
        # TCR α / β position embeddings
        self.max_tcra_len = 500
        self.max_tcrb_len = 500
        self.max_pep_len  = 20
        self.max_hla_len  = 180
        self.alpha = alpha
        self.gamma = gamma
        self.dropout = dropout
        self.use_struct = use_struct

        if loss_fn == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weights]) if pos_weights is not None else None)
        elif loss_fn == 'focal':
            self.loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        self.tcra_pos_embed = nn.Parameter(torch.randn(self.max_tcra_len, tcr_dim))
        self.tcrb_pos_embed = nn.Parameter(torch.randn(self.max_tcrb_len, tcr_dim))
        self.pep_pos_embed  = nn.Parameter(torch.randn(self.max_pep_len, pep_dim))
        self.hla_pos_embed  = nn.Parameter(torch.randn(self.max_hla_len, hla_dim))

        self.device = device
        self.tcr_dim = tcr_dim
        self.pep_dim = pep_dim
        self.hla_dim = hla_dim
        self.bilinear_dim = bilinear_dim
        
        d_model = tcr_dim
        n_heads = 8

        self.cross_attn_tcra_pep = nn.MultiheadAttention(d_model, n_heads, dropout=self.dropout, batch_first=True)
        self.cross_attn_tcra_hla = nn.MultiheadAttention(d_model, n_heads, dropout=self.dropout, batch_first=True)
        self.cross_attn_tcrb_pep = nn.MultiheadAttention(d_model, n_heads, dropout=self.dropout, batch_first=True)
        self.cross_attn_tcrb_hla = nn.MultiheadAttention(d_model, n_heads, dropout=self.dropout, batch_first=True)
        self.norm_tcra_pep = nn.LayerNorm(d_model)
        self.norm_tcra_hla = nn.LayerNorm(d_model)
        self.norm_tcrb_pep = nn.LayerNorm(d_model)
        self.norm_tcrb_hla = nn.LayerNorm(d_model)

        # =======================
        # TCRα / TCRβ encoders
        # =======================
        def make_tcr_encoder():
            proj_phys = ResidueProjector(20, tcr_dim)
            proj_esm = ResidueProjector(1280, tcr_dim)
            if self.use_struct:
                se3 = StackedEGNN(dim=17, layers=1)
                proj_struct = ResidueProjector(17, tcr_dim)
                gate = ResidueTripleFusion(tcr_dim)
            else:
                se3 = None
                proj_struct = None
                gate = ResidueDoubleFusion(tcr_dim)
            encoder_layer = TransformerEncoderLayer(
                d_model=tcr_dim, nhead=8, dim_feedforward=tcr_dim*4, dropout=self.dropout, batch_first=True
            )
            encoder = TransformerEncoder(encoder_layer, num_layers=2)
            return nn.ModuleDict(dict(
                proj_phys=proj_phys, proj_esm=proj_esm, proj_struct=proj_struct,
                se3=se3, gate=gate, encoder=encoder
            ))

        self.tcra_enc = make_tcr_encoder()
        self.tcrb_enc = make_tcr_encoder()

        # =======================
        # Peptide encoder (phys + esm + structure)
        # =======================
        self.proj_pep_phys = ResidueProjector(20, pep_dim)
        self.proj_pep_esm  = ResidueProjector(1280, pep_dim)
        if self.use_struct:
            self.pep_se3 = StackedEGNN(dim=17, layers=1)
            self.proj_pep_struct = ResidueProjector(17, pep_dim)
            self.pep_gate = ResidueTripleFusion(pep_dim)
        else:
            self.pep_se3 = None
            self.proj_pep_struct = None
            self.pep_gate = ResidueDoubleFusion(pep_dim)
            
        pep_encoder_layer = TransformerEncoderLayer(
            d_model=pep_dim, nhead=8, dim_feedforward=pep_dim*4, dropout=self.dropout, batch_first=True
        )
        self.pep_encoder = TransformerEncoder(pep_encoder_layer, num_layers=2)

        # =======================
        # HLA encoder
        # =======================
        self.proj_hla_phys = ResidueProjector(20, hla_dim)
        self.proj_hla_esm = ResidueProjector(1280, hla_dim)
        if self.use_struct:
            self.hla_se3 = StackedEGNN(dim=17, layers=1)
            self.proj_hla_struct = ResidueProjector(17, hla_dim)
            self.hla_gate = ResidueTripleFusion(hla_dim)
        else:
            self.hla_se3 = None
            self.proj_hla_struct = None
            self.hla_gate = ResidueDoubleFusion(hla_dim)
        hla_encoder_layer = TransformerEncoderLayer(
            d_model=hla_dim, nhead=8, dim_feedforward=hla_dim*4, dropout=self.dropout, batch_first=True
        )
        self.hla_encoder = TransformerEncoder(hla_encoder_layer, num_layers=1)

        self.pep_gate_2 = ResidueDoubleFusion(pep_dim)
        self.hla_gate_2 = ResidueDoubleFusion(hla_dim)

        # =======================
        # Bilinear interactions
        # =======================
        self.bi_tcra_pep = BANLayer(tcr_dim, pep_dim, bilinear_dim, h_out=4, k=3)
        self.bi_tcrb_pep = BANLayer(tcr_dim, pep_dim, bilinear_dim, h_out=4, k=3)
        self.bi_tcra_hla = BANLayer(tcr_dim, hla_dim, bilinear_dim, h_out=4, k=3)
        self.bi_tcrb_hla = BANLayer(tcr_dim, hla_dim, bilinear_dim, h_out=4, k=3)

        # =======================
        # Head
        # =======================
        total_fused_dim = bilinear_dim * 4
        self.head = nn.Sequential(
            nn.Linear(total_fused_dim, bilinear_dim),
            nn.ReLU(),
            nn.Linear(bilinear_dim, 1)
        )

    def encode_tcr(self, x_phys, x_esm, x_struct, x_coord, x_mask, enc, pos_embed):
        phys = enc['proj_phys'](x_phys)
        esm  = enc['proj_esm'](x_esm)
        if self.use_struct:
            se3  = enc['se3'](x_struct, x_coord, None)[0]
            se3  = enc['proj_struct'](se3)
            feat = enc['gate'](phys, esm, se3)
        else:
            feat = enc['gate'](phys, esm)
        feat = self.add_positional_encoding(feat, pos_embed)
        feat = enc['encoder'](feat, src_key_padding_mask=~x_mask)
        return feat
    
    def add_positional_encoding(self, x, pos_embed):
        """
        x: [B, L, D]
        pos_embed: [L_max, D]
        """
        B, L, D = x.shape
        pe = pos_embed[:L, :].unsqueeze(0).expand(B, -1, -1)
        return x + pe
    
    def _extract_cdr3_segment(self, tcr_feat, cdr3_start, cdr3_end):
        """
        Extracts CDR3 embeddings and corresponding mask.
        tcr_feat: [B, L, D]
        cdr3_start, cdr3_end: [B]
        Returns:
            out:  [B, max_len, D]
            mask: [B, max_len]  (True = valid)
        """
        B, L, D = tcr_feat.shape
        device = tcr_feat.device

        # Length of CDR3 for each sample
        lens = (cdr3_end - cdr3_start).clamp(min=0)
        max_len = lens.max().item()

        rel_idx = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)  # [B, max_len]
        abs_idx = cdr3_start.unsqueeze(1) + rel_idx  # [B, max_len]

        # mask: True means valid
        mask = rel_idx < lens.unsqueeze(1)  # [B, max_len]

        # Set out-of-range indices to 0 (any valid index works because they will be masked out)
        abs_idx = torch.where(mask, abs_idx, torch.zeros_like(abs_idx))

        # gather
        gather_idx = abs_idx.unsqueeze(-1).expand(-1, -1, D)
        out = torch.gather(tcr_feat, 1, gather_idx)

        # Force zero out positions where mask is False to avoid invalid tokens participating in computation
        out = out * mask.unsqueeze(-1)

        return out, mask

    def forward(self, batch):
        # TCRα / TCRβ
        tcra_feat = self.encode_tcr(
            batch['tcra_phys'].to(self.device, non_blocking=True),
            batch['tcra_esm'].to(self.device, non_blocking=True),
            batch['tcra_struct'].to(self.device, non_blocking=True),
            batch['tcra_coord'].to(self.device, non_blocking=True),
            batch['tcra_mask'].to(self.device, non_blocking=True),
            self.tcra_enc,
            self.tcra_pos_embed
        )
        tcrb_feat = self.encode_tcr(
            batch['tcrb_phys'].to(self.device, non_blocking=True),
            batch['tcrb_esm'].to(self.device, non_blocking=True),
            batch['tcrb_struct'].to(self.device, non_blocking=True),
            batch['tcrb_coord'].to(self.device, non_blocking=True),
            batch['tcrb_mask'].to(self.device, non_blocking=True),
            self.tcrb_enc,
            self.tcrb_pos_embed
        )
        # peptide
        pep_phys = self.proj_pep_phys(batch['pep_phys'].to(self.device, non_blocking=True))
        pep_esm  = self.proj_pep_esm(batch['pep_esm'].to(self.device, non_blocking=True))
        if self.use_struct:
            pep_se3 = self.pep_se3(batch['pep_struct'].to(self.device, non_blocking=True), batch['pep_coord'].to(self.device, non_blocking=True), None)[0]
            pep_se3 = self.proj_pep_struct(pep_se3)
            pep_feat = self.pep_gate(pep_phys, pep_esm, pep_se3)
        else:
            pep_feat = self.pep_gate(pep_phys, pep_esm)
        pep_feat = self.add_positional_encoding(pep_feat, self.pep_pos_embed)
        pep_feat = self.pep_encoder(
            pep_feat,
            src_key_padding_mask=~batch['pep_mask'].to(self.device)
        )
        # hla
        hla_phys = self.proj_hla_phys(batch['hla_phys'].to(self.device, non_blocking=True))
        hla_esm  = self.proj_hla_esm(batch['hla_esm'].to(self.device, non_blocking=True))
        if self.use_struct:
            hla_se3 = self.hla_se3(batch['hla_struct'].to(self.device, non_blocking=True), batch['hla_coord'].to(self.device, non_blocking=True), None)[0]
            hla_se3 = self.proj_hla_struct(hla_se3)
            hla_feat = self.hla_gate(hla_phys, hla_esm, hla_se3)
        else:
            hla_feat = self.hla_gate(hla_phys, hla_esm)
        hla_feat = self.add_positional_encoding(hla_feat, self.hla_pos_embed)
        hla_feat = self.hla_encoder(hla_feat)

        if ('pep_feat_pretrain' in batch) and ('hla_feat_pretrain' in batch):
            pep_pretrain = batch['pep_feat_pretrain'].to(self.device, non_blocking=True)
            hla_pretrain = batch['hla_feat_pretrain'].to(self.device, non_blocking=True)

            # ---- Robust length alignment (clip to minimum length) ----
            Lp = pep_feat.shape[1]
            Lp_pretrain = pep_pretrain.shape[1]
            if Lp != Lp_pretrain:
                Lp_min = min(Lp, Lp_pretrain)
                pep_feat = pep_feat[:, :Lp_min, :]
                pep_pretrain = pep_pretrain[:, :Lp_min, :]

            Lh = hla_feat.shape[1]
            Lh_pretrain = hla_pretrain.shape[1]
            if Lh != Lh_pretrain:
                Lh_min = min(Lh, Lh_pretrain)
                hla_feat = hla_feat[:, :Lh_min, :]
                hla_pretrain = hla_pretrain[:, :Lh_min, :]

            # ---- Peptide gating ----
            pep_feat = self.pep_gate_2(pep_feat, pep_pretrain)
            # ---- HLA gating ----
            hla_feat = self.hla_gate_2(hla_feat, hla_pretrain)

        # TCRα CDR3 segment
        tcra_cdr3, cdr3a_mask = self._extract_cdr3_segment(
            tcra_feat,
            batch['cdr3a_start'].to(self.device, non_blocking=True),
            batch['cdr3a_end'].to(self.device, non_blocking=True)
        )

        # TCRβ CDR3 segment
        tcrb_cdr3, cdr3b_mask = self._extract_cdr3_segment(
            tcrb_feat,
            batch['cdr3b_start'].to(self.device, non_blocking=True),
            batch['cdr3b_end'].to(self.device, non_blocking=True)
        )
        
        # TCRα CDR3 ← Peptide
        tcra_cdr3_cross, _ = self.cross_attn_tcra_pep(
            query=tcra_cdr3,                  # [B, La_cdr3, D]
            key=pep_feat, value=pep_feat,     # [B, Lp, D]
            key_padding_mask=~batch['pep_mask'].to(self.device)
        )
        tcra_cdr3 = self.norm_tcra_pep(tcra_cdr3 + tcra_cdr3_cross)
        # Re-mask padding CDR3 positions to prevent invalid tokens from leaking
        tcra_cdr3 = tcra_cdr3 * cdr3a_mask.unsqueeze(-1)

        # TCRβ CDR3 ← Peptide
        tcrb_cdr3_cross, _ = self.cross_attn_tcrb_pep(
            query=tcrb_cdr3,
            key=pep_feat, value=pep_feat,
            key_padding_mask=~batch['pep_mask'].to(self.device)
        )
        tcrb_cdr3 = self.norm_tcrb_pep(tcrb_cdr3 + tcrb_cdr3_cross)
        tcrb_cdr3 = tcrb_cdr3 * cdr3b_mask.unsqueeze(-1)

        # ------------------ Cross-Attn：TCR full sequences ------------------
        # TCRα full ← HLA
        tcra_hla_cross, _ = self.cross_attn_tcra_hla(
            query=tcra_feat,                  # [B, La, D]
            key=hla_feat, value=hla_feat,     # [B, Lh, D]
            key_padding_mask=None
        )
        tcra_feat = self.norm_tcra_hla(tcra_feat + tcra_hla_cross)
        tcra_feat = tcra_feat * batch['tcra_mask'].to(self.device).unsqueeze(-1)

        # TCRβ full ← HLA
        tcrb_hla_cross, _ = self.cross_attn_tcrb_hla(
            query=tcrb_feat,
            key=hla_feat, value=hla_feat,
            key_padding_mask=None
        )
        tcrb_feat = self.norm_tcrb_hla(tcrb_feat + tcrb_hla_cross)
        tcrb_feat = tcrb_feat * batch['tcrb_mask'].to(self.device).unsqueeze(-1)

        # bilinear fusion
        vec_tcra_pep, attn_tcra_pep = self.bi_tcra_pep(tcra_cdr3, pep_feat, v_mask=cdr3a_mask, q_mask=batch['pep_mask'].to(self.device))
        vec_tcrb_pep, attn_tcrb_pep = self.bi_tcrb_pep(tcrb_cdr3, pep_feat, v_mask=cdr3b_mask, q_mask=batch['pep_mask'].to(self.device))
        vec_tcra_hla, attn_tcra_hla = self.bi_tcra_hla(tcra_feat, hla_feat, v_mask=batch['tcra_mask'].to(self.device), q_mask=None)
        vec_tcrb_hla, attn_tcrb_hla = self.bi_tcrb_hla(tcrb_feat, hla_feat, v_mask=batch['tcrb_mask'].to(self.device), q_mask=None)
        
        attn_tcra_pep_small = attn_tcra_pep.sum(dim=1).float() 
        attn_tcrb_pep_small = attn_tcrb_pep.sum(dim=1).float()
        attn_tcra_hla_small = attn_tcra_hla.sum(dim=1).float()
        attn_tcrb_hla_small = attn_tcrb_hla.sum(dim=1).float()

        attn_dict = {
            'attn_tcra_pep': attn_tcra_pep_small.detach().cpu().numpy(),
            'attn_tcrb_pep': attn_tcrb_pep_small.detach().cpu().numpy(),
            'attn_tcra_hla': attn_tcra_hla_small.detach().cpu().numpy(),
            'attn_tcrb_hla': attn_tcrb_hla_small.detach().cpu().numpy()
        }

        fused = torch.cat([vec_tcra_pep, vec_tcrb_pep, vec_tcra_hla, vec_tcrb_hla], dim=-1)
        logits = self.head(fused).squeeze(-1)
        
        labels = batch['label'].to(self.device)
        loss_binding = self.loss_fn(logits, labels.float())

        probs = torch.sigmoid(logits)
                
        return probs, loss_binding, fused, attn_dict
