import torch 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from cafa6_parse import propagate

def predict_one(model, emb, go_maps, parents, priors=None, taxonomy_prior=None, fasta_path=None):
    model.eval()
    with torch.no_grad():
        out = model(emb.to(DEVICE))

    scores = {}

    for head in ["mf","bp","cc"]:
        probs = torch.sigmoid(out[head])[0].cpu().numpy()
        idx2go = go_maps[head]["idx2go"]
        for i, sc in enumerate(probs):
            if sc > 0.001:
                scores[idx2go[i]] = float(sc)

    # MMseqs
    if fasta_path is not None:
        hom = mmseqs_predict(fasta_path)
        for go, sc in hom.items():
            scores[go] = max(scores.get(go, 0.0), sc)

    # Priors
    if priors:
        for go in list(scores.keys()):
            scores[go] *= priors.get(go, 1.0)

    # Propagate
    scores = propagate(scores, parents)

    return scores