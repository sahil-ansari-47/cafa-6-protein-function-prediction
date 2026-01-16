import os
import gc
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
# 1. OPTIMIZED OBO PARSER
def parse_obo_parents(go_obo_path):
    print(f"[1/5] Parsing OBO Ontology...")
    term_parents = defaultdict(set)
    roots = set(['GO:0003674', 'GO:0008150', 'GO:0005575'])
    
    with open(go_obo_path, "r") as f:
        cur_id = None
        for line in f:
            line = line.strip()
            if line == "[Term]":
                cur_id = None
            elif line.startswith("id: "):
                cur_id = line.split("id: ")[1].strip()
            elif line.startswith("is_a: "):
                pid = line.split()[1].strip()
                if cur_id: term_parents[cur_id].add(pid)
            elif line.startswith("relationship: part_of "):
                parts = line.split()
                if len(parts) >= 3:
                    pid = parts[2].strip()
                    if cur_id: term_parents[cur_id].add(pid)
    return term_parents, roots

def get_ancestors_map(term_parents):
    print("[1/5] Building Ancestor Map...")
    ancestors = {}
    def get_all_ancestors(term):
        if term in ancestors: return ancestors[term]
        parents = term_parents.get(term, set())
        all_anc = set(parents)
        for p in parents:
            all_anc |= get_all_ancestors(p)
        ancestors[term] = all_anc
        return all_anc
    
    for term in tqdm(list(term_parents.keys())):
        get_all_ancestors(term)
    return ancestors

# 2. LOGIC: PROPAGATION + NORMALIZATION
def process_predictions(df, ancestors_map, roots):
    print("[3/5] Processing Predictions (Propagate + Normalize)...")
    # 1. Convert to Dict Structure for fast access
    # { protein_id: { go_term: score } }
    data_map = defaultdict(dict)
    values = df.values # protein, term, score
    for pid, term, score in tqdm(values, desc="Grouping"):
        data_map[pid][term] = float(score)
    new_rows = []
    # 2. Iterate Per Protein
    for pid, terms_dict in tqdm(data_map.items(), desc="Optimizing"):
        final_scores = terms_dict.copy()
        # --- A. POSITIVE PROPAGATION ---
        # Ensure Parent >= Child
        # Sort terms to process leaves first (optional but helps)
        current_terms = list(terms_dict.keys())
        for term in current_terms:
            s = terms_dict[term]
            if term in ancestors_map:
                for anc in ancestors_map[term]:
                    final_scores[anc] = max(final_scores.get(anc, 0.0), s)
        
        # --- B. FORCE ROOTS ---
        # If we have any prediction, the roots must be 1.0
        if len(final_scores) > 0:
            for r in roots:
                final_scores[r] = 1.0
        
        # --- C. RANK NORMALIZATION (The Boost) ---
        # Find the max score for this protein (excluding roots usually 1.0)
        # We want to boost the best "non-root" prediction to a high confidence
        # to ensure it survives thresholding.
        
        max_val = 0.0
        for t, s in final_scores.items():
            if t not in roots:
                max_val = max(max_val, s)
        
        # If the best prediction is weak (e.g., 0.3), scale everything up
        # Target: Make the max score at least 0.95
        if max_val > 0 and max_val < 0.95:
            scale_factor = 0.95 / max_val
            for t in final_scores:
                if t not in roots: # Don't scale roots > 1.0
                    final_scores[t] = min(1.0, final_scores[t] * scale_factor)
        # Collect
        for go_term, score in final_scores.items():
            # Optimization: Drop extremely low scores to reduce file size
            if score >= 0.001:
                new_rows.append((pid, go_term, score))
                
    return pd.DataFrame(new_rows, columns=['protein_id', 'go_term', 'score'])
# 3. MAIN PIPELINE
OBO_PATH = "/kaggle/input/cafa-6-protein-function-prediction/Train/go-basic.obo"
SUBMISSION_INPUT = '/kaggle/input/cafa6-protein-function-enhanced-nb-v2/submission.tsv'
SUBMISSION_OUTPUT = 'submission.tsv'
# 1. Load Ontology
term_parents, roots = parse_obo_parents(OBO_PATH)
ancestors_map = get_ancestors_map(term_parents)
# 2. Load Submission
print(f"[2/5] Loading submission...")
# If OOM occurs, add nrows=10_000_000 or use chunks
submission = pd.read_csv(SUBMISSION_INPUT, sep='\t', header=None, names=['protein_id', 'go_term', 'score', 'key'])
submission = submission[['protein_id', 'go_term', 'score']]
# 3. Process
final_df = process_predictions(submission, ancestors_map, roots)
# 4. Save
print(f"[4/5] Saving {len(final_df)} rows...")
# Sorting ensures deterministic file for hashing/checks
final_df.sort_values(['protein_id', 'score'], ascending=[True, False], inplace=True)
final_df.to_csv(SUBMISSION_OUTPUT, sep='\t', index=False, header=False)
print(f"[✅] Done. Saved to {SUBMISSION_OUTPUT}")
print(final_df.head())