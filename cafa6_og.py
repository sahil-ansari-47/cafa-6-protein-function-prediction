import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import argparse
from cafa6_embed import process_pids, DEVICE
from cafa6_train import GOModel
from cafa6_parse import read_fasta, parse_go_obo, load_train_terms, propagate
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['user', 'friend', 'both'], default='both', help='Which part to process: user (30%), friend (70%), or both')
    args = parser.parse_args()    
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
    print("Loading test sequences...")
    test_seqs = read_fasta("Test/testsuperset.fasta")
    print("Loading ProtT5...")
    # 1. Filter valid sequences first
    valid_pids = [p for p in train_seqs if p in train_terms]
    valid_pids.sort(key=lambda x: len(train_seqs[x]), reverse=True)
    # Split valid_pids: 30% for user, 70% for friend
    split_idx = int(0.4 * len(valid_pids))
    friend_pids = valid_pids[:split_idx]
    user_pids = valid_pids[split_idx:]
    print(f"User will embed {len(user_pids)} sequences, friend will embed {len(friend_pids)} sequences")
    # Split test sequences similarly
    test_pids = list(test_seqs.keys())
    test_pids.sort(key=lambda x: len(test_seqs[x]), reverse=True)
    split_idx = int(0.4 * len(test_pids))
    friend_test_pids = test_pids[:split_idx]
    user_test_pids = test_pids[split_idx:]
    print(f"Friend test: {len(friend_test_pids)}, User test: {len(user_test_pids)}")
    # Create all embeddings if not exist
    if not os.path.exists("user_train_embeddings.pt"):
        process_pids(user_pids, "user_train", train_seqs, train_terms, go2idx)
    if not os.path.exists("friend_train_embeddings.pt"):
        process_pids(friend_pids, "friend_train", train_seqs, train_terms, go2idx)
    if not os.path.exists("friend_test_embeddings.pt"):
        process_pids(friend_test_pids, "friend_test", test_seqs)
    if not os.path.exists("user_test_embeddings.pt"):
        process_pids(user_test_pids, "user_test", test_seqs)
    # Now, for training if role == 'both'
    if args.role == 'both':
        if not os.path.exists("model.pt"):
            print("Loading train embeddings...")
            X_user = torch.load("user_train_embeddings.pt")
            Y_user = torch.load("user_train_labels.pt")
            X_friend = torch.load("friend_train_embeddings.pt")
            Y_friend = torch.load("friend_train_labels.pt")
            X = torch.cat([X_user, X_friend])
            Y = torch.cat([Y_user, Y_friend])
            X = X.float()
            Y = Y.float()
            # Create model
            input_dim = 1024  # ProtT5 XL output dimension
            model = GOModel(input_dim, len(go2idx)).to(DEVICE)
            # Compute pos_weight and train
            pos_counts = Y.sum(dim=0)
            neg_counts = (1 - Y).sum(dim=0)
            pos_weight = neg_counts / (pos_counts + 1e-6)
            scaler = 100.0 / pos_weight.mean()
            pos_weight = pos_weight * scaler
            pos_weight = pos_weight.to(DEVICE)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            batch_size = 32
            print("Training...")
            for epoch in range(5):
                model.train()
                total = 0
                num_batches = 0
                for i in tqdm(range(0, len(X), batch_size), desc=f"Epoch {epoch+1}/5"):
                    xb = X[i:i+batch_size].to(DEVICE)
                    yb = Y[i:i+batch_size].to(DEVICE)
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total += loss.item()
                    num_batches += 1
                avg_loss = total / num_batches
                print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
            torch.save(model.state_dict(), "model.pt")
        else:
            print("Model already exists, skipping training.")
    elif args.role == 'user':
        print("Embedding done for user. Skipping training.")
    elif args.role == 'friend':
        print("Embedding done for friend. Skipping training.")
    
    # Prediction
    print("Predicting test set...")
    # Load model
    input_dim = 1024
    model = GOModel(input_dim, len(go2idx)).to(DEVICE)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    # Load test embeddings
    X_friend_test = torch.load("friend_test_embeddings.pt")
    X_user_test = torch.load("user_test_embeddings.pt")
    X_test = torch.cat([X_friend_test, X_user_test])
    test_pids_ordered = friend_test_pids + user_test_pids
    output_file = "submission.tsv"
    out = open(output_file, "w")
    batch_size = 32
    for i in tqdm(range(0, len(X_test), batch_size), desc="Predicting"):
        xb = X_test[i:i+batch_size].to(DEVICE)
        with torch.no_grad():
            logits = model(xb)
            batch_preds = torch.sigmoid(logits).cpu().numpy()
        for j, pred in enumerate(batch_preds):
            pid = test_pids_ordered[i + j]
            scores = {}
            indices = np.where(pred > 0.01)[0]
            for idx in indices:
                scores[idx2go[idx]] = float(pred[idx])
            # GO propagation
            scores = propagate(scores, parents)
            # write top 100
            for go, sc in sorted(scores.items(), key=lambda x: -x[1])[:100]:
                out.write(f"{pid}\t{go}\t{sc:.4f}\n")
    out.close()
    print(f"Done. Written {output_file}")
    
if __name__ == "__main__":
    main()
