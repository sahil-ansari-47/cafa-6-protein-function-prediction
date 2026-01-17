import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
print("Using cuda:", torch.cuda.is_available())
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
        self.model = T5EncoderModel.from_pretrained(
            PROTT5,
            use_safetensors=False,
            weights_only=False,
            dtype=torch.float16
        ).to(DEVICE)
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
    # 1. Filter valid sequences first
    valid_pids = [p for p in train_seqs if p in train_terms]
    # 2. Sort by sequence length (Crucial for speed!)
    # Processing similar lengths together reduces wasted padding computations
    valid_pids.sort(key=lambda x: len(train_seqs[x]), reverse=True)
    X_list = []
    Y_list = []
    # 3. Define Batch Size (Tune this!)
    # Start with 2. If you have a 24GB GPU (3090/4090), try 4 or 8. 
    # If it crashes, go down to 1.
    BATCH_SIZE = 2 
    print(f"Embedding {len(valid_pids)} sequences in batches of {BATCH_SIZE}...")
    # 4. Batch Processing Loop
    for i in tqdm(range(0, len(valid_pids), BATCH_SIZE)):
        batch_pids = valid_pids[i : i + BATCH_SIZE]
        batch_seqs = [train_seqs[pid] for pid in batch_pids]
        # Prepare sequences for ProtT5 (add spaces between residues)
        batch_seqs_formatted = [" ".join(list(s)) for s in batch_seqs]
        try:
            # Tokenize the whole batch at once
            # This automatically pads to the longest sequence IN THIS BATCH (not the whole dataset)
            tokens = embedder.tokenizer(
                batch_seqs_formatted, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024 # Limit very long seqs to prevent OOM
            )
            with torch.no_grad():
                # Move entire batch to GPU
                input_ids = tokens['input_ids'].to(DEVICE)
                attention_mask = tokens['attention_mask'].to(DEVICE)
                # Forward pass (this is the heavy part)
                out = embedder.model(input_ids=input_ids, attention_mask=attention_mask)
                # Extract embeddings (mean pooling ignoring padding)
                # We use attention_mask to ensure we don't count padding zeros in the mean
                embeddings = out.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                # Move back to CPU to save RAM
                X_list.append(batch_embeddings.cpu())
                # Prepare Labels for this batch
                for pid in batch_pids:
                    y = torch.zeros(len(go2idx))
                    for go in train_terms[pid]:
                        if go in go2idx:
                            y[go2idx[go]] = 1.0
                    Y_list.append(y)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Skipping batch {i} due to OOM (sequence too long)")
                torch.cuda.empty_cache()
            else:
                raise e
    # Stack results
    X = torch.cat(X_list) # Use cat instead of stack for lists of tensors
    Y = torch.stack(Y_list)
    print("Train size:", X.shape)
    # === CRITICAL: SAVE THIS SO YOU DON'T HAVE TO DO IT AGAIN ===
    print("Saving embeddings to disk...")
    torch.save(X, "train_embeddings.pt")
    torch.save(Y, "train_labels.pt")
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
