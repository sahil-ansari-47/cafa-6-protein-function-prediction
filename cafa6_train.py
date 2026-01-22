import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from cafa6_parse import propagate
from sklearn.metrics import average_precision_score

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
        return self.net(x)
    
class MultiHeadGOModel(nn.Module):
    def __init__(self, in_dim, mf_dim, bp_dim, cc_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.ReLU()
        )
        self.mf = nn.Linear(2048, mf_dim)
        self.bp = nn.Linear(2048, bp_dim)
        self.cc = nn.Linear(2048, cc_dim)

    def forward(self, x):
        h = self.shared(x)
        return {
            "mf": self.mf(h),
            "bp": self.bp(h),
            "cc": self.cc(h),
        }

def compute_metrics(y_true, y_score):
    """
    y_true, y_score: (N, C)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    # Micro AUPR
    micro_aupr = average_precision_score(y_true.ravel(), y_score.ravel())
    # Macro AUPR (ignore classes with no positives)
    per_class = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            per_class.append(average_precision_score(y_true[:, i], y_score[:, i]))
    macro_aupr = float(np.mean(per_class)) if len(per_class) else 0.0
    # Fmax (scan thresholds)
    thresholds = np.linspace(0.05, 0.95, 19)
    fmax = 0.0
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = (y_pred * y_true).sum()
        fp = (y_pred * (1 - y_true)).sum()
        fn = ((1 - y_pred) * y_true).sum()
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        fmax = max(fmax, f1)
    return {
        "micro_aupr": micro_aupr,
        "macro_aupr": macro_aupr,
        "fmax": fmax
    }
def train_model(
    dataset,
    loss_weights,
    epochs=50,
    batch_size=16,
    lr=1e-4,
    val_split=0.1,
    patience=5,
    grad_accum=2,
    use_amp=True
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Split dataset
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    sample_x, _ = dataset[0]
    in_dim = sample_x.shape[0]
    out_dim = dataset.num_classes
    model = GOModel(in_dim, out_dim).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=loss_weights.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_fmax = 0.0
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        # ========== TRAIN ==========
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [TRAIN]", leave=False)
        opt.zero_grad()
        for i, (xb, yb) in enumerate(pbar):
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)
                loss = loss_fn(logits, yb) / grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            total_loss += loss.item() * grad_accum
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")
        avg_train_loss = total_loss / len(train_loader)
        # ========== VALIDATE ==========
        model.eval()
        all_true = []
        all_score = []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [VAL]", leave=False)
            for xb, yb in pbar:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits)
                all_true.append(yb.cpu().numpy())
                all_score.append(probs.cpu().numpy())
        all_true = np.vstack(all_true)
        all_score = np.vstack(all_score)
        metrics = compute_metrics(all_true, all_score)
        print(
            f"Epoch {epoch:03d} | "
            f"TrainLoss {avg_train_loss:.4f} | "
            f"microAUPR {metrics['micro_aupr']:.4f} | "
            f"macroAUPR {metrics['macro_aupr']:.4f} | "
            f"Fmax {metrics['fmax']:.4f}"
        )
        # ========== EARLY STOPPING ==========
        if metrics["fmax"] > best_fmax:
            best_fmax = metrics["fmax"]
            bad_epochs = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("  ✅ New best model saved.")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("  ⛔ Early stopping triggered.")
                break
    print("Training done. Best Fmax:", best_fmax)
    model.load_state_dict(torch.load("best_model.pt"))
    return model

def compute_fmax(y_true, y_score):
    thresholds = np.linspace(0.05, 0.95, 19)
    fmax = 0.0
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = (y_pred * y_true).sum()
        fp = (y_pred * (1 - y_true)).sum()
        fn = ((1 - y_pred) * y_true).sum()
        if tp + fp == 0 or tp + fn == 0:
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r + 1e-8)
        fmax = max(fmax, f1)
    return fmax

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def multi_head_train(model, train_loader, val_loader, loss_fns, epochs=20, lr=1e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device = DEVICE)

    best_f = 0.0
    bad = 0

    for epoch in range(1, epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]")
        for xb, y in pbar:
            xb = xb.to(DEVICE)
            for k in y:
                y[k] = y[k].to(DEVICE)

            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                out = model(xb)
                loss = (
                    loss_fns["mf"](out["mf"], y["mf"]) +
                    loss_fns["bp"](out["bp"], y["bp"]) +
                    loss_fns["cc"](out["cc"], y["cc"])
                )

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ===== VALIDATE =====
        model.eval()
        all_true = {"mf":[], "bp":[], "cc":[]}
        all_score = {"mf":[], "bp":[], "cc":[]}

        with torch.no_grad():
            for xb, y in val_loader:
                xb = xb.to(DEVICE)
                for k in y:
                    y[k] = y[k].to(DEVICE)

                out = model(xb)
                for k in ["mf","bp","cc"]:
                    all_true[k].append(y[k].cpu().numpy())
                    all_score[k].append(torch.sigmoid(out[k]).cpu().numpy())

        fms = []
        for k in ["mf","bp","cc"]:
            yt = np.vstack(all_true[k])
            ys = np.vstack(all_score[k])
            fms.append(compute_fmax(yt, ys))

        fmean = float(np.mean(fms))
        print(f"Epoch {epoch} Fmax MF/BP/CC = {fms}  Mean = {fmean:.4f}")

        if fmean > best_f:
            best_f = fmean
            bad = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("  ✅ saved best model")
        else:
            bad += 1
            if bad >= 5:
                print("  ⛔ early stopping")
                break
    model.load_state_dict(torch.load("best_model.pt"))
    return model