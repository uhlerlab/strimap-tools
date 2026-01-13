import csv
import numpy as np
import pandas as pd
import random
import re
from typing import Tuple, List
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def load_train_data(
    df_train_list: List[pd.DataFrame],
    df_val_list: List[pd.DataFrame],
    hla_dict_path: str = 'pMHC/HLA_dict.npy',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and validation datasets only.

    Args:
        hla_dict_path: Path to HLA dictionary
        train_folds: List of training fold indices
        val_folds: List of validation fold indices
        sample_frac: Fraction of data to sample (for quick testing)
        seed: Random seed

    Returns:
        df_train, df_val
    """
    print("Loading training and validation data...")

    # Load HLA dictionary
    HLA_dict = np.load(hla_dict_path, allow_pickle=True).item()
    
    # Process HLA names → full sequence
    for df in df_train_list + df_val_list:
        df['HLA'] = df['HLA'].apply(lambda x: x[4:] if x.startswith('HLA-') else x)
        df['HLA_full'] = df['HLA'].apply(lambda x: HLA_dict[x])

    return df_train_list, df_val_list

def load_test_data(
    df_test: pd.DataFrame,
    hla_dict_path: str = 'pMHC/HLA_dict.npy'
) -> pd.DataFrame:
    """
    Preprocess a given test DataFrame (e.g. independent or external set).

    Args:
        df_test: Test dataframe with at least 'HLA', 'peptide', 'label'
        hla_dict_path: Path to HLA dictionary (to map HLA name to full sequence)

    Returns:
        Processed df_test with 'HLA_full' added
    """
    print("Processing test data...")

    HLA_dict = np.load(hla_dict_path, allow_pickle=True).item()

    df_test = df_test.copy()
    df_test['HLAtmp'] = df_test['HLA'].apply(lambda x: x[4:] if x.startswith('HLA-') else x)
    df_test['HLA_full'] = df_test['HLAtmp'].apply(lambda x: HLA_dict[x])
    df_test = df_test.drop(columns=['HLAtmp'])

    print(f"✓ Test set: {len(df_test)} samples")
    return df_test

# ==============================
# Helper: HLA Normalization
# ==============================
def normalize_hla(x):
    if pd.isna(x) or str(x).strip() == "":
        return None
    x = str(x).strip().upper()

    # Remove leading HLA- or HLA:
    x = re.sub(r"^HLA[-:]", "", x)

    # Convert A0201 -> A*02:01
    if re.match(r"^[A-Z]\d{4}$", x):
        x = f"{x[0]}*{x[1:3]}:{x[3:]}"

    # Convert A*0201 -> A*02:01
    if re.match(r"^[A-Z]\*\d{4}$", x):
        x = f"{x[:3]}{x[3:5]}:{x[5:]}"

    # Ensure final format is HLA-A*02:01
    return "HLA-" + x

def compute_percentile_ranking(df_all, df_pred):
    """
    Optimized version using binary search (np.searchsorted) instead of broadcasting.
    Drastically faster for large datasets (Saturation Mutagenesis).
    """
    df_pred = df_pred.copy().reset_index(drop=True)
    
    # Initialize result column with NaNs
    df_pred["rank_percentile"] = np.nan

    # Get unique HLAs involved in the prediction
    target_hlas = df_pred['HLA'].unique()

    for hla in target_hlas:
        # 1. Get Prediction Scores for this HLA group
        group_mask = df_pred["HLA"] == hla
        scores = df_pred.loc[group_mask, "Prediction"].values
        
        # 2. Get Background Scores for this HLA
        bkg_scores = df_all[df_all["HLA"] == hla]["Prediction"].values
        
        if len(bkg_scores) == 0:
            continue
            
        # 3. CRITICAL OPTIMIZATION: Sort background and use Binary Search
        # This avoids creating a (N_background x N_pred) matrix which crashes/slows down memory
        bkg_scores_sorted = np.sort(bkg_scores)
        
        # Find indices where elements should be inserted to maintain order
        # 'left' means indices = number of background scores strictly less than the prediction
        indices = np.searchsorted(bkg_scores_sorted, scores, side='left')
        
        # Calculate percentile: (count_smaller / total_background) * 100
        # Rank Percentile: 100 - (percent_smaller) => Top X%
        percentile_ranks = 100 - (indices / len(bkg_scores_sorted) * 100)
        
        # Assign back to DataFrame
        df_pred.loc[group_mask, "rank_percentile"] = percentile_ranks

    # ===== Binder Assignment =====
    def assign_binder(row):
        rank = row["rank_percentile"]
        if pd.isna(rank): return " "
        if rank <= 1: return "Strong"
        elif rank <= 5: return "Weak"
        else: return " "

    df_pred["Binder"] = df_pred.apply(assign_binder, axis=1)

    return df_pred
    
def determine_tcr_seq_vj(cdr3,V,J,chain,guess01=False):
    
    def file2dict(filename,key_fields,store_fields,delimiter='\t'):
        """Read file to a dictionary.
        key_fields: fields to be used as keys
        store_fields: fields to be saved as a list
        delimiter: delimiter used in the given file."""
        dictionary={}
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile,delimiter=delimiter)
            for row in reader:
                keys = [row[k] for k in key_fields]
                store= [row[s] for s in store_fields]

                sub_dict = dictionary
                for key in keys[:-1]:
                    if key not in sub_dict:
                        sub_dict[key] = {}
                    sub_dict = sub_dict[key]
                key = keys[-1]
                if key not in sub_dict:
                    sub_dict[key] = []
                sub_dict[key].append(store)
        return dictionary
    
    def get_protseqs_ntseqs(chain='B'):
        """returns sequence dictioaries for genes: protseqsV, protseqsJ, nucseqsV, nucseqsJ"""
        seq_dicts=[]
        for gene,type in zip(['v','j','v','j'],['aa','aa','nt','nt']):
            name = 'library/'+'tr'+chain.lower()+gene+'s_'+type+'.tsv'
            sdict = file2dict(name,key_fields=['Allele'],store_fields=[type+'_seq'])
            for g in sdict:
                sdict[g]=sdict[g][0][0]
            seq_dicts.append(sdict)
        return seq_dicts
    
    protVb,protJb,_,_ = get_protseqs_ntseqs(chain='B')
    protVa,protJa,_,_ = get_protseqs_ntseqs(chain='A')
    def splice_v_cdr3_j(pv: str, pj: str, cdr3: str) -> str:
        """
        pv: V gene protein sequence
        pj: J gene protein sequence
        cdr3: C-starting, F/W-ending CDR3 sequence (protein)
        Returns: The spliced full sequence (V[:lastC] + CDR3 + J suffix)
        """
        pv = (pv or "").strip().upper()
        pj = (pj or "").strip().upper()
        cdr3 = (cdr3 or "").strip().upper()

        # 1) V segment: Take the last 'C' (including the conserved C in V region)
        cpos = pv.rfind('C')
        if cpos == -1:
            raise ValueError("V sequence has no 'C' to anchor CDR3 start.")
        v_prefix = pv[:cpos]  # up to and including C

        # 2) Align CDR3's "end overlap" in J
        #    Start from the full length of cdr3, gradually shorten it, and find the longest suffix that can match in J
        j_suffix = pj  # fallback (in extreme cases)
        for k in range(len(cdr3), 0, -1):
            tail = cdr3[-k:]                 # CDR3's suffix
            m = re.search(re.escape(tail), pj)
            if m:
                j_suffix = pj[m.end():]      # Take the suffix from the matching segment
                break

        return v_prefix + cdr3 + j_suffix
        
    tcr_list = []
    for i in range(len(cdr3)):
        cdr3_ = cdr3[i]
        V_ = V[i]
        J_ = J[i]
        if chain=='A':
            protseqsV = protVa
            protseqsJ = protJa
        else:
            protseqsV = protVb
            protseqsJ = protJb
        if guess01:
            if '*' not in V_:
                V_+='*01'
            if '*' not in J_:
                J_+='*01'
        pv = protseqsV[V_]
        pj = protseqsJ[J_]
        # t = pv[:pv.rfind('C')]+  cdr3_ + pj[re_search(r'[FW]G.[GV]',pj).start()+1:]
        t = splice_v_cdr3_j(pv, pj, cdr3_)
        tcr_list.append(t)
    return tcr_list

def get_mutated_peptide_fast(peptide: str, min_mutations: int = 1, max_mutations: int = 3) -> str:
    """
    Fast peptide mutation function.
    Randomly mutates between min_mutations and max_mutations amino acids in the peptide.
    Ensures that the mutated amino acid is different from the original.
    """
    L = len(peptide)
    # If the peptide is too short to satisfy the minimum number of mutations, force 1 mutation
    if L < min_mutations:
        num_mutations = 1
    else:
        num_mutations = random.randint(min_mutations, min(max_mutations, L))
    
    mutation_indices = random.sample(range(L), num_mutations)
    peptide_list = list(peptide)
    
    for idx in mutation_indices:
        original = peptide_list[idx]
        # Quickly select a non-original AA from the list
        # This approach is slightly faster than list comprehension in tight loops
        mutation = random.choice(AMINO_ACIDS)
        while mutation == original:
            mutation = random.choice(AMINO_ACIDS)
        peptide_list[idx] = mutation
            
    return "".join(peptide_list)

def mutated_peptide_sampling_aligned(
    df_pos: pd.DataFrame, 
    neg_multiplier: int = 1, 
    min_mut: int = 1, 
    max_mut: int = 3,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate mutated peptide negative samples aligned with df_pos.
    Each peptide in df_pos is mutated to create a corresponding negative sample.
    The resulting DataFrame maintains the same structure as df_pos, with mutated peptides and label set to 0.
    """
    random.seed(random_state)
    np.random.seed(random_state)
    
    # 1. Prepare data container
    peptides = df_pos['peptide'].values
    
    # 2. Batch generate mutated peptides
    # Here we generate a list with length = len(df_pos) * neg_multiplier
    mutated_peptides = []
    
    # For deduplication check, establish a set of positive samples
    # (Note: For speed, here we only check if Peptide-TCR pairs completely collide,
    #  but in Triplet Loss scenarios, as long as the Peptide changes, even if the collision probability is very low, it is usually considered a negative sample)
    # If the dataset is very large, deduplication checks can be very slow and can be skipped if necessary
    
    for _ in range(neg_multiplier):
        for pep in peptides:
            mutated_peptides.append(get_mutated_peptide_fast(pep, min_mut, max_mut))
    
    # 3. Construct negative DataFrame
    # Directly copy the structure of df_pos to avoid row-wise operations
    if neg_multiplier == 1:
        df_neg = df_pos.copy()
    else:
        # If multiplier > 1, we need to replicate df_pos multiple times to maintain column information
        df_neg = pd.concat([df_pos] * neg_multiplier, ignore_index=True)
    
    # 4. Overwrite Peptide column
    df_neg['peptide'] = mutated_peptides
    
    # 5. Set Label to 0
    if 'label' in df_neg.columns:
        df_neg['label'] = 0
        
    return df_neg

def mutate_cdr3_in_full_sequence(full_seq, start, end, min_mut=1, max_mut=3):
    """
    In the full-length sequence, only mutate the CDR3 region specified by [start:end].
    Preserve the first and last amino acids of the CDR3 (usually C and F/W), and only mutate the middle part.
    
    Returns:
        tuple: (new_full_seq, new_cdr3_seq)
    """
    # Extract the CDR3 segment
    # Note: Python slicing is left-inclusive, right-exclusive [start:end], exactly corresponding to start=91, len=15, end=106
    original_cdr3 = full_seq[start:end]
    
    # Safety check: if index is wrong or CDR3 is too short, do not mutate
    if len(original_cdr3) <= 4:
        return full_seq, original_cdr3
    
    # To protect conserved anchor residues, only mutate the middle part
    # CDR3 usually starts with C and ends with F/W.
    # mutable_part_indices are indices relative to the cdr3 string
    # Range from 1 to len-2
    mutable_len = len(original_cdr3)
    if mutable_len > 4:
        valid_indices = list(range(1, mutable_len - 1))
    else:
        # If very short (rare), mutate all except the first
        valid_indices = list(range(1, mutable_len))
        
    # Randomly decide the number of mutations
    num_mutations = random.randint(min_mut, min(max_mut, len(valid_indices)))
    mutate_indices = random.sample(valid_indices, num_mutations)
    
    cdr3_list = list(original_cdr3)
    
    for idx in mutate_indices:
        original_aa = cdr3_list[idx]
        mutation = random.choice(AMINO_ACIDS)
        while mutation == original_aa:
            mutation = random.choice(AMINO_ACIDS)
        cdr3_list[idx] = mutation
        
    new_cdr3 = "".join(cdr3_list)
    
    # Concatenate back to the full-length sequence
    new_full_seq = full_seq[:start] + new_cdr3 + full_seq[end:]
    
    return new_full_seq, new_cdr3

def mutated_tcr_sampling_aligned(
    df_pos: pd.DataFrame, 
    neg_multiplier: int = 1, 
    min_mut: int = 1, 
    max_mut: int = 3,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate TCR mutated negative samples strictly aligned with df_pos.
    Mutate the CDR3 region and simultaneously update the full sequence and cdr3 column.
    """
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Extract necessary numpy arrays for faster iteration
    tcra_list = df_pos['tcra'].values
    tcrb_list = df_pos['tcrb'].values
    # Ensure indices are integers
    start_a_list = df_pos['cdr3a_start'].astype(int).values
    end_a_list   = df_pos['cdr3a_end'].astype(int).values
    start_b_list = df_pos['cdr3b_start'].astype(int).values
    end_b_list   = df_pos['cdr3b_end'].astype(int).values
    
    cdr3a_col_list = df_pos['cdr3a'].values
    cdr3b_col_list = df_pos['cdr3b'].values

    # Result containers
    new_tcra_full = []
    new_tcrb_full = []
    new_cdr3a_part = []
    new_cdr3b_part = []
    
    # Loop to generate
    for _ in range(neg_multiplier):
        # Use zip to iterate over all columns simultaneously, much faster than iterrows
        iterator = zip(tcra_list, start_a_list, end_a_list, cdr3a_col_list,
                       tcrb_list, start_b_list, end_b_list, cdr3b_col_list)
        
        for (seq_a, s_a, e_a, c_a, seq_b, s_b, e_b, c_b) in iterator:
            
            # Random strategy: mutate Alpha chain, Beta chain, or both
            # This strategy increases the diversity of negative samples
            r = random.random()
            
            # Initialize with original values
            mut_seq_a, mut_c_a = seq_a, c_a
            mut_seq_b, mut_c_b = seq_b, c_b
            
            if r < 0.33: 
                # Mutate only Alpha CDR3
                mut_seq_a, mut_c_a = mutate_cdr3_in_full_sequence(seq_a, s_a, e_a, min_mut, max_mut)
            elif r < 0.66: 
                # Mutate only Beta CDR3
                mut_seq_b, mut_c_b = mutate_cdr3_in_full_sequence(seq_b, s_b, e_b, min_mut, max_mut)
            else: 
                # Mutate both simultaneously
                mut_seq_a, mut_c_a = mutate_cdr3_in_full_sequence(seq_a, s_a, e_a, min_mut, max_mut)
                mut_seq_b, mut_c_b = mutate_cdr3_in_full_sequence(seq_b, s_b, e_b, min_mut, max_mut)
            
            new_tcra_full.append(mut_seq_a)
            new_cdr3a_part.append(mut_c_a)
            new_tcrb_full.append(mut_seq_b)
            new_cdr3b_part.append(mut_c_b)

    # Construct DataFrame
    if neg_multiplier == 1:
        df_neg = df_pos.copy()
    else:
        df_neg = pd.concat([df_pos] * neg_multiplier, ignore_index=True)
        
    # Update all relevant columns
    df_neg['tcra'] = new_tcra_full
    df_neg['tcrb'] = new_tcrb_full
    df_neg['cdr3a'] = new_cdr3a_part
    df_neg['cdr3b'] = new_cdr3b_part
    
    # Set negative label
    df_neg['label'] = 0
    
    # Retain original sequences for structural reuse (if "lazy" strategy is needed)
    # If not using structural reuse, these two lines can be commented out, but keeping them does no harm
    df_neg['original_tcra'] = list(df_pos['tcra']) * neg_multiplier
    df_neg['original_tcrb'] = list(df_pos['tcrb']) * neg_multiplier
    
    return df_neg

def negative_sampling_phla(df, neg_ratio=5, label_col='label', neg_label=0, random_state=42):
    """
    Create negative samples by shuffling TCRs while keeping peptide–HLA pairs intact.
    Ensures negative samples count = neg_ratio × positive samples count.
    """
    np.random.seed(random_state)
    pos_triplets = set(zip(df['tcra'], df['tcrb'], df['peptide'], df['HLA_full']))
    tcr_cols = ['tcra', 'cdr3a_start', 'cdr3a_end', 'tcrb', 'cdr3b_start', 'cdr3b_end']

    n_pos = len(df)
    target_n_neg = n_pos * neg_ratio
    all_neg = []
    
    i = 0
    while len(all_neg) < target_n_neg:
        shuffled_df = df.copy()
        shuffled_tcr = df[tcr_cols].sample(frac=1, random_state=random_state + i).reset_index(drop=True)
        for col in tcr_cols:
            shuffled_df[col] = shuffled_tcr[col]

        mask_keep = []
        for idx, row in shuffled_df.iterrows():
            triplet = (row['tcra'], row['tcrb'], row['peptide'], row['HLA_full'])
            mask_keep.append(triplet not in pos_triplets)
        shuffled_df = shuffled_df[mask_keep]
        shuffled_df[label_col] = neg_label

        all_neg.append(shuffled_df)
        i += 1

        if len(pd.concat(all_neg)) > target_n_neg * 1.5:
            break

    negative_samples = pd.concat(all_neg, ignore_index=True).drop_duplicates()
    negative_samples = negative_samples.sample(
        n=min(len(negative_samples), target_n_neg), random_state=random_state
    ).reset_index(drop=True)

    return negative_samples


