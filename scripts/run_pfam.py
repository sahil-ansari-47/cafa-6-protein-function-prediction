import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from feature_extraction.pfam import extract_pfam_features
import pickle


# if __name__ == "__main__":

FASTA = "Train/train_sequences.fasta"
PFAM_HMM = "Pfam/Pfam-A.hmm"
OUTDIR = "output/pfam"

print("Running Pfam HMMER...")
features, vocab = extract_pfam_features(
    fasta=FASTA,
    pfam_hmm=PFAM_HMM,
    workdir=OUTDIR
)

with open("output/pfam/pfam_features.pkl", "wb") as f:
    pickle.dump(features, f)

with open("output/pfam/pfam_vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("Saved Pfam features:", len(features))
print("Pfam vocab size:", len(vocab))
