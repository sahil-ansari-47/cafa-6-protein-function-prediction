from features import extract_deeptmhmm_features, extract_pfam_features

# DeepTMHMM
tm_feats = extract_deeptmhmm_features(
    fasta="Train/train_sequences.fasta",
    outdir="Features/deeptmhmm"
)

# Pfam (training - builds vocab)
pfam_feats, pfam_vocab = extract_pfam_features(
    fasta="Train/train_sequences.fasta",
    pfam_hmm="Pfam/Pfam-A.hmm",
    workdir="Features/pfam"
)

# Pfam (inference - reuse vocab)
import pickle
with open("pfam_vocab.pkl", "wb") as f:
    pickle.dump(pfam_vocab, f)

# Later, for inference:
with open("pfam_vocab.pkl", "rb") as f:
    pfam_vocab = pickle.load(f)
pfam_feats, _ = extract_pfam_features(
    fasta="Test/testsuperset.fasta",
    pfam_hmm="Pfam/Pfam-A.hmm",
    workdir="Features/pfam_test",
    pfam_vocab=pfam_vocab  # Reuse vocab!
)