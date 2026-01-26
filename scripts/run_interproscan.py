import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pickle
from feature_extraction.interproscan import parse_interpro_tsv

TSV_FILE = "output/train_domain_extraction/train_sequences.fasta.tsv"
OUT_PKL = "features/train_interpro_features.pkl"

# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    features = parse_interpro_tsv(TSV_FILE, OUT_PKL)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(features, f)

    print(f"✅ Saved features for {len(features)} proteins → {OUT_PKL}")
    
