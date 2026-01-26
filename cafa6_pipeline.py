import os
import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm
# from cafa6_train import train_model
# from cafa6_parse import parse_go_obo, read_fasta, load_train_terms
import pickle
# -----------------------------
# Feature loaders
# -----------------------------
def load_prott5_embeddings(emb_dir: str, seq_ids: List[str]) -> Dict[str, torch.Tensor]:
    feats = {}
    for sid in tqdm(seq_ids, desc="Loading ProtT5"):
        path = os.path.join(emb_dir, f"{sid}.pt")
        feats[sid] = torch.load(path)  # shape: [D]
    return feats
def load_deeptmhmm_features(path: str) -> Dict[str, np.ndarray]:
    feats = {}
    # Example format: seqid \t num_TM \t has_signal \t ...
    with open(path) as f:
        for line in f:
            sid, *vals = line.strip().split("\t")
            feats[sid] = np.array([float(x) for x in vals], dtype=np.float32)
    return feats
def load_domain_features(path: str) -> Dict[str, np.ndarray]:
    """
    Load InterProScan-derived domain features and return
    fixed-length NumPy feature vectors per protein.

    Returns:
        Dict[str, np.ndarray]  # protein_id -> feature vector
    """

    # ---------------- Load PKL ----------------
    with open(path, "rb") as f:
        data = pickle.load(f)

    # ---------------- Build vocabularies ----------------
    interpro_vocab = {}
    signature_vocab = {}

    for feat in data.values():
        for ipr in feat["interpro"]:
            if ipr not in interpro_vocab:
                interpro_vocab[ipr] = len(interpro_vocab)

        for src, sigs in feat["signatures"].items():
            for sig in sigs:
                key = f"{src}:{sig}"
                if key not in signature_vocab:
                    signature_vocab[key] = len(signature_vocab)

    num_interpro = len(interpro_vocab)
    num_signatures = len(signature_vocab)

    # Numeric feature size (fixed)
    NUMERIC_DIM = 5
    # [num_domains, unique_domains, coverage, min_log_evalue, strong_hit_count]

    total_dim = num_interpro + num_signatures + NUMERIC_DIM

    # ---------------- Build feature vectors ----------------
    features = {}

    for pid, feat in data.items():
        vec = np.zeros(total_dim, dtype=np.float32)

        # ---- InterPro (binary) ----
        for ipr in feat["interpro"]:
            vec[interpro_vocab[ipr]] = 1.0

        offset = num_interpro

        # ---- Signatures (counts) ----
        for src, sigs in feat["signatures"].items():
            for sig, count in sigs.items():
                key = f"{src}:{sig}"
                vec[offset + signature_vocab[key]] = float(count)

        offset += num_signatures

        # ---- Numeric features ----
        vec[offset + 0] = feat["architecture"]["num_domains"]
        vec[offset + 1] = feat["architecture"]["unique_domains"]
        vec[offset + 2] = feat["architecture"]["coverage"]

        min_log_ev = feat["confidence"]["min_log_evalue"]
        vec[offset + 3] = min_log_ev if min_log_ev is not None else 0.0

        vec[offset + 4] = feat["confidence"]["strong_hit_count"]

        features[pid] = vec

    return features
def load_disorder_features(path: str) -> Dict[str, np.ndarray]:
    feats = {}
    # Example: seqid \t frac_disordered \t avg_score \t ...
    with open(path) as f:
        for line in f:
            sid, *vals = line.strip().split("\t")
            feats[sid] = np.array([float(x) for x in vals], dtype=np.float32)
    return feats
def load_mmseqs_features(path: str) -> Dict[str, np.ndarray]:
    feats = {}
    # Example: seqid \t v1,v2,v3,... (GO evidence vector or summary stats)
    with open(path) as f:
        for line in f:
            sid, vec = line.strip().split("\t")
            feats[sid] = np.fromstring(vec, sep=",", dtype=np.float32)
    return feats
# -----------------------------
# Dataset assembly
# -----------------------------
def build_feature_matrix(
    seq_ids: List[str],
    prott5: Dict[str, torch.Tensor],
    tm: Dict[str, np.ndarray],
    pfam: Dict[str, np.ndarray],
    disorder: Dict[str, np.ndarray],
    mmseqs: Dict[str, np.ndarray] | None = None,
):
    X = []
    for sid in tqdm(seq_ids, desc="Building feature matrix"):
        parts = []
        parts.append(prott5[sid].cpu().numpy())
        parts.append(tm.get(sid, np.zeros(4, dtype=np.float32)))
        parts.append(pfam.get(sid, np.zeros(256, dtype=np.float32)))
        parts.append(disorder.get(sid, np.zeros(4, dtype=np.float32)))
        if mmseqs is not None:
            parts.append(mmseqs.get(sid, np.zeros(128, dtype=np.float32)))
        x = np.concatenate(parts)
        X.append(x)
    X = np.stack(X)
    return torch.tensor(X, dtype=torch.float32)
def build_label_matrix(seq_ids: List[str], train_terms, go2idx):
    Y = torch.zeros((len(seq_ids), len(go2idx)), dtype=torch.float32)

    for i, sid in enumerate(seq_ids):
        gos = train_terms.get(sid, [])
        for go in gos:
            if go in go2idx:
                Y[i, go2idx[go]] = 1.0

    return Y
# def train_model(X, Y, epochs=10, batch_size=32, lr=1e-4):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = GOModel(X.shape[1], Y.shape[1]).to(device)
#     opt = torch.optim.AdamW(model.parameters(), lr=lr)
#     loss_fn = torch.nn.BCEWithLogitsLoss()
#     dataset = torch.utils.data.TensorDataset(X, Y)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0.0
#         for xb, yb in loader:
#             xb = xb.to(device)
#             yb = yb.to(device)
#             opt.zero_grad()
#             logits = model(xb)
#             loss = loss_fn(logits, yb)
#             loss.backward()
#             opt.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}: loss = {total_loss/len(loader):.4f}")
#     return model
def main():
    # print("Loading GO...")
    # parents, namespace = parse_go_obo("Train/go-basic.obo")
    # print("Loading train sequences...")
    # train_seqs = read_fasta("Train/train_sequences.fasta")
    # print("Loading train labels...")
    # train_terms = load_train_terms("Train/train_terms.tsv")
    # print("Collecting GO vocabulary...")
    # all_gos = set()
    # for gos in train_terms.values():
    #     all_gos |= gos
    # go2idx = {go: i for i, go in enumerate(sorted(all_gos))}
    # idx2go = {i: go for go, i in go2idx.items()}
    # print("GO terms:", len(go2idx))
    # seq_ids = list(train_seqs.keys())
    # # ---- Load features ----
    # prott5 = load_prott5_embeddings("Embeddings/ProtT5", seq_ids)
    # tm = load_deeptmhmm_features("Features/deeptmhmm.tsv")
    domain = load_domain_features("features/train_interpro_features.pkl")
    # print(domain['sp|Q801X7|AJL1_ANGJA'].shape)
    # disorder = load_disorder_features("Features/disorder.tsv")
    # # mmseqs = load_mmseqs_features("Features/mmseqs.tsv")  # optional
    # # ---- Build matrices ----
    # X = build_feature_matrix(seq_ids, prott5, tm, pfam, disorder, mmseqs=None)
    # Y = build_label_matrix(seq_ids, train_terms, go2idx)
    # print("X shape:", X.shape)
    # print("Y shape:", Y.shape)
    # # ---- Train ----
    # model = train_model(X, Y, epochs=10, batch_size=16, lr=1e-4)
    # # ---- Save ----
    # torch.save({
    #     "model": model.state_dict(),
    #     "go2idx": go2idx,
    # }, "go_model.pt")
    # print("Model saved to go_model.pt")
if __name__ == "__main__":
    main()
