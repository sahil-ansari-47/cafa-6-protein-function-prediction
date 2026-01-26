import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feature_extraction.deeptmhmm import extract_deeptmhmm_features
import pickle

FASTA = "Train/train_sequences.fasta"
OUTDIR = "output/train_deeptmhmm"
OUT_PKL = "features/train_deeptmhmm_features.pkl"

print("Running DeepTMHMM...")
# features = extract_deeptmhmm_features(FASTA, OUTDIR)
extract_deeptmhmm_features(FASTA, OUTDIR)

# with open(OUT_PKL, "wb") as f:
#     pickle.dump(features, f)

# print("Saved DeepTMHMM features:", len(features))
