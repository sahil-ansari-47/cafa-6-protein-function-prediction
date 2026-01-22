import torch
from torch.utils.data import Dataset

def load_ia(path: str):
    ia = {}
    with open(path) as f:
        for line in f:
            go, val = line.strip().split()
            ia[go] = float(val)
    return ia

def select_go_terms(train_terms, ia_map, min_ia=0.0):
    """
    Drops IA == 0
    Keeps others, but we will weight by IA
    """
    all_gos = set()
    for gos in train_terms.values():
        all_gos |= gos
    selected = []
    for go in all_gos:
        if ia_map.get(go, 0.0) > 0.0:   # drop IA == 0
            selected.append(go)
    selected = sorted(selected)
    go2idx = {go: i for i, go in enumerate(selected)}
    idx2go = {i: go for go, i in go2idx.items()}
    print("Original GO terms:", len(all_gos))
    print("After dropping IA=0:", len(go2idx))
    return go2idx, idx2go

def build_loss_weights(go2idx, ia_map, min_weight=0.1, max_weight=1.0):
    """
    weight(go) = clamp(IA(go), min_weight, max_weight)
    """
    weights = torch.zeros(len(go2idx), dtype=torch.float32)
    for go, idx in go2idx.items():
        ia = ia_map.get(go, 0.0)
        w = max(min_weight, min(max_weight, ia))
        weights[idx] = w
    print("Loss weights stats:")
    print("  min:", weights.min().item())
    print("  max:", weights.max().item())
    print("  mean:", weights.mean().item())
    return weights

class GOSparseDataset(Dataset):
    def __init__(self, seq_ids, features, train_terms, go2idx):
        """
        features: Dict[seq_id] -> torch.Tensor feature vector
        train_terms: Dict[seq_id] -> Set[GO]
        """
        self.seq_ids = seq_ids
        self.features = features
        self.train_terms = train_terms
        self.go2idx = go2idx
        self.num_classes = len(go2idx)
    def __len__(self):
        return len(self.seq_ids)
    def __getitem__(self, i):
        sid = self.seq_ids[i]
        x = self.features[sid]   # torch tensor
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for go in self.train_terms.get(sid, []):
            if go in self.go2idx:
                y[self.go2idx[go]] = 1.0
        return x, y
