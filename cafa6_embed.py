import torch
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm

print("Using cuda:", torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROTT5 = "Rostlab/prot_t5_xl_uniref50"

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
        emb = out.last_hidden_state.mean(dim=1).float()
        return emb[0].cpu()

embedder = ProtT5Embedder()

def process_pids(pids, prefix, seqs, terms=None, go2idx=None):
    X_list = []
    Y_list = []
    max_tokens_per_batch = 1000
    batch_seqs = []
    batch_pids = []
    current_tokens = 0
    print(f"Embedding {len(pids)} sequences for {prefix} with dynamic batching...")
    for pid in tqdm(pids):
        seq = seqs[pid]
        seq_len = len(seq)  # Approximate tokens
        if current_tokens + seq_len > max_tokens_per_batch and batch_seqs:
            # Process current batch
            try:
                batch_seqs_formatted = [" ".join(list(s)) for s in batch_seqs]
                tokens = embedder.tokenizer(
                    batch_seqs_formatted, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=1024
                )
                with torch.no_grad():
                    input_ids = tokens['input_ids'].to(DEVICE)
                    attention_mask = tokens['attention_mask'].to(DEVICE)
                    out = embedder.model(input_ids=input_ids, attention_mask=attention_mask)
                    embeddings = out.last_hidden_state
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                    X_list.append(batch_embeddings.cpu())
                    if terms is not None and go2idx is not None:
                        for pid in batch_pids:
                            y = torch.zeros(len(go2idx))
                            for go in terms[pid]:
                                if go in go2idx:
                                    y[go2idx[go]] = 1.0
                            Y_list.append(y)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Skipping batch due to OOM")
                    torch.cuda.empty_cache()
                else:
                    raise e
            # Reset batch
            batch_seqs = []
            batch_pids = []
            current_tokens = 0
        batch_seqs.append(seq)
        batch_pids.append(pid)
        current_tokens += seq_len
    # Process remaining batch
    if batch_seqs:
        try:
            batch_seqs_formatted = [" ".join(list(s)) for s in batch_seqs]
            tokens = embedder.tokenizer(
                batch_seqs_formatted, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            )
            with torch.no_grad():
                input_ids = tokens['input_ids'].to(DEVICE)
                attention_mask = tokens['attention_mask'].to(DEVICE)
                out = embedder.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = out.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                X_list.append(batch_embeddings.cpu())
                if terms is not None and go2idx is not None:
                    for pid in batch_pids:
                        y = torch.zeros(len(go2idx))
                        for go in terms[pid]:
                            if go in go2idx:
                                y[go2idx[go]] = 1.0
                        Y_list.append(y)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Skipping final batch due to OOM")
                torch.cuda.empty_cache()
            else:
                raise e
    # Stack results
    X = torch.cat(X_list)
    print(f"{prefix} size:", X.shape)
    # Save
    torch.save(X, f"{prefix}_embeddings.pt")
    if terms is not None and go2idx is not None:
        Y = torch.stack(Y_list)
        torch.save(Y, f"{prefix}_labels.pt")
        return X, Y
    else:
        return X