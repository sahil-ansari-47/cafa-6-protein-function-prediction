import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROTT5 = "Rostlab/prot_t5_xl_uniref50"

# =========================
# FASTA
# =========================

def read_fasta(path):
    seqs = {}
    current = None
    for line in open(path):
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            pid = line.split("|")[1] if "|" in line else line[1:].split()[0]
            current = pid
            seqs[pid] = ""
        else:
            seqs[current] += line.strip()
    return seqs

# =========================
# GO PARSER
# =========================

def parse_go_obo(path):
    parents = {}
    namespace = {}
    cur = None
    for line in open(path):
        line = line.strip()
        if line == "[Term]":
            cur = None
        elif line.startswith("id:"):
            cur = line.split("id:")[1].strip()
            parents[cur] = []
        elif cur and line.startswith("namespace:"):
            namespace[cur] = line.split("namespace:")[1].strip()
        elif cur and line.startswith("is_a:"):
            p = line.split("is_a:")[1].split("!")[0].strip()
            parents[cur].append(p)
    return parents, namespace

# =========================
# GO PROPAGATION
# =========================

def propagate(scores, parents):
    out = dict(scores)
    for go, sc in list(scores.items()):
        stack = [go]
        while stack:
            g = stack.pop()
            for p in parents.get(g, []):
                if p not in out or out[p] < sc:
                    out[p] = sc
                    stack.append(p)
    return out

# =========================
# DATASET
# =========================

def load_train_terms(path):
    ann = {}
    for line in open(path):
        pid, go, ont = line.strip().split("\t")
        ann.setdefault(pid, set()).add(go)
    return ann

# =========================
# ProtT5 EMBEDDER
# =========================

class ProtT5Embedder:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(PROTT5, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(PROTT5).to(DEVICE)
        self.model.eval()

    def embed(self, seq):
        seq = " ".join(list(seq))
        tokens = self.tokenizer(seq, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**{k: v.to(DEVICE) for k, v in tokens.items()})
        emb = out.last_hidden_state.mean(dim=1)
        return emb[0].cpu()

# =========================
# MODEL
# =========================
class GOModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, out_dim)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))

# =========================
# MAIN
# =========================

def main():
    print("Loading GO...")
    parents, namespace = parse_go_obo("Train/go-basic.obo")

    print("Loading train sequences...")
    train_seqs = read_fasta("Train/train_sequences.fasta")

    print("Loading train labels...")
    train_terms = load_train_terms("Train/train_terms.tsv")

    print("Collecting GO vocabulary...")
    all_gos = set()
    for gos in train_terms.values():
        all_gos |= gos

    go2idx = {go:i for i,go in enumerate(sorted(all_gos))}
    idx2go = {i:go for go,i in go2idx.items()}

    print("GO terms:", len(go2idx))

    print("Loading ProtT5...")
    embedder = ProtT5Embedder()

    X = []
    Y = []

    print("Embedding training sequences...")
    for pid, seq in tqdm(train_seqs.items()):
        if pid not in train_terms:
            continue
        emb = embedder.embed(seq)
        y = torch.zeros(len(go2idx))
        for go in train_terms[pid]:
            if go in go2idx:
                y[go2idx[go]] = 1.0
        X.append(emb)
        Y.append(y)

    X = torch.stack(X)
    Y = torch.stack(Y)

    print("Train size:", X.shape)

    model = GOModel(X.shape[1], Y.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

    print("Training...")
    for epoch in range(5):
        model.train()
        total = 0
        for i in range(0, len(X), 8):
            xb = X[i:i+8].to(DEVICE)
            yb = Y[i:i+8].to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print("Epoch", epoch, "loss", total)

    torch.save(model.state_dict(), "model.pt")

    print("Predicting test set...")
    test_seqs = read_fasta("test/testsuperset.fasta")

    out = open("submission.tsv", "w")

    model.eval()

    for pid, seq in tqdm(test_seqs.items()):
        emb = embedder.embed(seq).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(emb)[0].cpu().numpy()

        scores = {}
        for i, sc in enumerate(pred):
            if sc > 0.01:
                scores[idx2go[i]] = float(sc)

        # GO propagation
        scores = propagate(scores, parents)

        # write top 100
        for go, sc in sorted(scores.items(), key=lambda x: -x[1])[:100]:
            out.write(f"{pid}\t{go}\t{sc:.4f}\n")

    out.close()
    print("Done. Written submission.tsv")

if __name__ == "__main__":
    main()
