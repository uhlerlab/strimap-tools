import torch
# =================================== Dataset / Collate ===========================================
class PepHLA_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, phys_dict, esm2_dict, struct_dict):
        self.df = df
        self.phys_dict = phys_dict
        self.esm2_dict = esm2_dict
        self.struct_dict = struct_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pep = row['peptide']
        hla = row['HLA_full']
        label = torch.tensor(row['label'], dtype=torch.float32)

        pep_phys = self.phys_dict['pep'][pep]
        pep_esm = self.esm2_dict['pep'][pep]

        hla_phys = self.phys_dict['hla'][hla]
        hla_esm = self.esm2_dict['hla'][hla]
        hla_struct, hla_coord = self.struct_dict[hla]

        return {
            'pep_phys': pep_phys,
            'pep_esm': pep_esm,
            'hla_phys': hla_phys,
            'hla_esm': hla_esm,
            'hla_struct': hla_struct,
            'hla_coord': hla_coord,
            'label': label,
            'pep_id': pep,
            'hla_id': hla,
        }
        
def peptide_hla_collate_fn(batch):
    def pad_or_crop(x, original_len, target_len):
        L, D = x.shape
        valid_len = min(original_len, target_len)
        valid_part = x[:valid_len]
        if valid_len < target_len:
            pad_len = target_len - valid_len
            padding = x.new_zeros(pad_len, D)
            return torch.cat([valid_part, padding], dim=0)
        else:
            return valid_part

    out_batch = {}

    pep_lens = [len(item['pep_id']) for item in batch]
    max_pep_len = max(pep_lens)

    for key in batch[0].keys():
        if key == 'label':
            out_batch[key] = torch.stack([item[key] for item in batch])
        elif key.startswith('pep_') and not key.endswith('_id'):
            out_batch[key] = torch.stack([pad_or_crop(item[key], len(item['pep_id']), max_pep_len) for item in batch])
        elif key.endswith('_id'):
            out_batch[key] = [item[key] for item in batch]
        else:
            out_batch[key] = torch.stack([item[key] for item in batch])
        
    def make_mask(lengths, max_len):
        masks = []
        for L in lengths:
            m = torch.zeros(max_len, dtype=torch.bool)
            m[:L] = True
            masks.append(m)
        return torch.stack(masks)

    out_batch['pep_mask'] = make_mask(pep_lens, max_pep_len)
    return out_batch

# =================================== Dataset / Collate ===========================================
class TCRPepHLA_Dataset(torch.utils.data.Dataset):
    """
    Dataset for TCRα + TCRβ + peptide + HLA binding.
    """
    def __init__(self, df, phys_dict, esm2_dict, struct_dict, pep_hla_feat_dict):
        self.df = df
        self.phys_dict = phys_dict
        self.esm2_dict = esm2_dict
        self.struct_dict = struct_dict
        self.pep_hla_feat_dict = pep_hla_feat_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tcra = row['tcra']
        tcrb = row['tcrb']
        pep = row['peptide']
        hla = row['HLA_full']
        label = torch.tensor(row['label'], dtype=torch.float32)

        # ---- TCRα ----
        tcra_phys = self.phys_dict['tcra'][tcra]
        tcra_esm  = self.esm2_dict['tcra'][tcra]
        tcra_struct, tcra_coord = self.struct_dict['tcra'][tcra]
        tcra_cdr3_start = torch.tensor(row['cdr3a_start'], dtype=torch.long)
        tcra_cdr3_end   = torch.tensor(row['cdr3a_end'], dtype=torch.long)

        # ---- TCRβ ----
        tcrb_phys = self.phys_dict['tcrb'][tcrb]
        tcrb_esm  = self.esm2_dict['tcrb'][tcrb]
        tcrb_struct, tcrb_coord = self.struct_dict['tcrb'][tcrb]
        tcrb_cdr3_start = torch.tensor(row['cdr3b_start'], dtype=torch.long)
        tcrb_cdr3_end   = torch.tensor(row['cdr3b_end'], dtype=torch.long)

        # ---- peptide ----
        pep_phys = self.phys_dict['pep'][pep]
        pep_esm  = self.esm2_dict['pep'][pep]
        pep_struct, pep_coord = self.struct_dict['pep'][pep]

        # ---- HLA ----
        hla_phys = self.phys_dict['hla'][hla]
        hla_esm  = self.esm2_dict['hla'][hla]
        hla_struct, hla_coord = self.struct_dict['hla'][hla]
        
        feats = self.pep_hla_feat_dict[(pep, hla)]
        pep_feat_pretrain = feats['pep_feat_pretrain']
        hla_feat_pretrain = feats['hla_feat_pretrain']

        return {
            # TCRα
            'tcra_phys': tcra_phys,
            'tcra_esm': tcra_esm,
            'tcra_struct': tcra_struct,
            'tcra_coord': tcra_coord,
            'cdr3a_start': tcra_cdr3_start,
            'cdr3a_end': tcra_cdr3_end,

            # TCRβ
            'tcrb_phys': tcrb_phys,
            'tcrb_esm': tcrb_esm,
            'tcrb_struct': tcrb_struct,
            'tcrb_coord': tcrb_coord,
            'cdr3b_start': tcrb_cdr3_start,
            'cdr3b_end': tcrb_cdr3_end,

            # peptide
            'pep_phys': pep_phys,
            'pep_esm': pep_esm,
            'pep_struct': pep_struct,
            'pep_coord': pep_coord,

            # HLA
            'hla_phys': hla_phys,
            'hla_esm': hla_esm,
            'hla_struct': hla_struct,
            'hla_coord': hla_coord,

            'tcra_id': tcra,
            'tcrb_id': tcrb,
            'pep_id': pep,
            'hla_id': hla,
            'label': label,

            'pep_feat_pretrain': pep_feat_pretrain,
            'hla_feat_pretrain': hla_feat_pretrain,
        }

# =================================== Collate Function ===========================================
def tcr_pep_hla_collate_fn(batch):
    def pad_or_crop(x, original_len, target_len):
        L, D = x.shape
        valid_len = min(original_len, target_len)
        valid_part = x[:valid_len]
        if valid_len < target_len:
            pad_len = target_len - valid_len
            padding = x.new_zeros(pad_len, D)
            return torch.cat([valid_part, padding], dim=0)
        else:
            return valid_part

    out_batch = {}

    tcra_lens = [len(item['tcra_id']) for item in batch]
    tcrb_lens = [len(item['tcrb_id']) for item in batch]
    pep_lens  = [len(item['pep_id'])  for item in batch]

    max_tcra_len = max(tcra_lens)
    max_tcrb_len = max(tcrb_lens)
    max_pep_len  = max(pep_lens)

    for key in batch[0].keys():
        if key == 'label':
            out_batch[key] = torch.stack([item[key] for item in batch])

        elif key.startswith('tcra_') and not key.endswith('_id'):
            out_batch[key] = torch.stack([pad_or_crop(item[key], len(item['tcra_id']), max_tcra_len) for item in batch])

        elif key.startswith('tcrb_') and not key.endswith('_id'):
            out_batch[key] = torch.stack([pad_or_crop(item[key], len(item['tcrb_id']), max_tcrb_len) for item in batch])

        elif key.startswith('pep_') and not key.endswith('_id'):
            out_batch[key] = torch.stack([pad_or_crop(item[key], len(item['pep_id']), max_pep_len) for item in batch])

        elif key.endswith('_id'):
            out_batch[key] = [item[key] for item in batch]

        else:
            out_batch[key] = torch.stack([item[key] for item in batch])

    def make_mask(lengths, max_len):
        masks = []
        for L in lengths:
            m = torch.zeros(max_len, dtype=torch.bool)
            m[:L] = True
            masks.append(m)
        return torch.stack(masks)

    out_batch['tcra_mask'] = make_mask(tcra_lens, max_tcra_len)
    out_batch['tcrb_mask'] = make_mask(tcrb_lens, max_tcrb_len)
    out_batch['pep_mask']  = make_mask(pep_lens,  max_pep_len)

    return out_batch