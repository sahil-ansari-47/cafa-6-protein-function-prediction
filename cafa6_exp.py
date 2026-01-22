import torch
import torch.nn as nn
import numpy as np
import json
import subprocess
from transformers import T5Tokenizer, T5EncoderModel
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROTT5 = "Rostlab/prot_t5_xl_uniref50"

# =========================
# FASTA READER
# =========================
def read_fasta(path):
    seqs = {}
    current = None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            pid = line.split("|")[1] if "|" in line else line[1:].split()[0]
            current = pid
            seqs[pid] = ""
        else:
            seqs[current] += line.strip()
    return seqs

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
class SimpleGOModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, out_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# =========================
# GO GRAPH
# =========================
class GOGraph:
    def __init__(self, parents_json):
        self.parents = json.load(open(parents_json))
    def propagate(self, scores):
        out = dict(scores)
        for go, score in scores.items():
            stack = [go]
            while stack:
                g = stack.pop()
                for p in self.parents.get(g, []):
                    if p not in out or out[p] < score:
                        out[p] = score
                        stack.append(p)
        return out

# =========================
# MMSEQS HOMOLOGY (VERY SIMPLE VERSION)
# =========================
def mmseqs_predict(fasta_path):
    """
    This function should:
    - Run mmseqs
    - Find nearest neighbors
    - Transfer their GO terms
    - Return: dict(go -> score)
    """

    # TODO: Replace with real MMseqs pipeline
    return {}  # empty fallback

# =========================
# MAIN PREDICTOR
# =========================
class CAFAPredictor:
    def __init__(self, models, go_index, go_graph):
        self.embedder = ProtT5Embedder()
        self.models = models
        self.go_index = go_index
        self.go_graph = go_graph

    def predict_one(self, pid, seq):
        emb = self.embedder.embed(seq).unsqueeze(0)
        # Ensemble NN
        preds = []
        for model in self.models:
            with torch.no_grad():
                preds.append(model(emb))
        nn_pred = torch.stack(preds).mean(dim=0)[0].cpu().numpy()
        nn_scores = {
            self.go_index[str(i)]: float(nn_pred[i])
            for i in range(len(nn_pred))
        }
        # MMseqs
        fasta = f"/tmp/{pid}.fasta"
        with open(fasta, "w") as f:
            f.write(f">{pid}\n{seq}\n")
        hom_scores = mmseqs_predict(fasta)
        # Blend
        final = dict(nn_scores)
        for go, s in hom_scores.items():
            final[go] = max(final.get(go, 0.0), s)
        # GO propagation
        final = self.go_graph.propagate(final)
        return final

# =========================
# RUN
# =========================
def main():
    # Load data
    seqs = read_fasta("train_sequences.fasta")
    go_index = json.load(open("go_index.json"))  # index -> GO
    go_graph = GOGraph("go_parents.json")
    # Load ensemble models
    models = [
        torch.load("model_fold1.pt", map_location=DEVICE),
        torch.load("model_fold2.pt", map_location=DEVICE),
        torch.load("model_fold3.pt", map_location=DEVICE),
    ]
    for m in models:
        m.eval()
    predictor = CAFAPredictor(models, go_index, go_graph)
    out = open("submission.txt", "w")
    for pid, seq in seqs.items():
        scores = predictor.predict_one(pid, seq)
        # Write top-K predictions
        for go, score in sorted(scores.items(), key=lambda x: -x[1])[:50]:
            if score > 0.01:
                out.write(f"{pid}\t{go}\t{score:.4f}\n")
    out.close()
    print("Done. Written to submission.txt")

if __name__ == "__main__":
    main()
