"""Pfam (HMMER) feature extraction module."""

import subprocess
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict


def run_hmmscan(fasta, pfam_hmm, out_file, threads=8):
    """Run hmmscan on a FASTA file against Pfam-A database.
    
    Args:
        fasta: Path to input FASTA file
        pfam_hmm: Path to Pfam-A.hmm database
        out_file: Path to output domtblout file
        threads: Number of CPU threads to use
    """
    cmd = [
        "hmmscan",
        "--cpu", str(threads),
        "--domtblout", out_file,
        pfam_hmm,
        fasta,
    ]

    subprocess.run(cmd, check=True)


def build_pfam_vocab(domtblout, max_domains=2000):
    """Build Pfam vocabulary from domtblout file (training only).
    
    Args:
        domtblout: Path to hmmscan domtblout output file
        max_domains: Maximum number of domains to keep in vocabulary
        
    Returns:
        Dictionary mapping pfam_id -> index
    """
    counter = Counter()

    with open(domtblout) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            pfam_id = parts[1]
            counter[pfam_id] += 1

    vocab = {
        pfam: i
        for i, (pfam, _) in enumerate(counter.most_common(max_domains))
    }

    return vocab


def parse_pfam(domtblout, pfam_vocab):
    """Parse domtblout file and produce fixed-length feature vectors.
    
    Args:
        domtblout: Path to hmmscan domtblout output file
        pfam_vocab: Dictionary mapping pfam_id -> index
        
    Returns:
        Dictionary mapping protein_id -> feature vector
        Feature vector: [binary Pfam vector | summary stats]
        Shape: (len(pfam_vocab) + 4,)
    """
    dim = len(pfam_vocab)
    raw = defaultdict(lambda: {
        "vec": np.zeros(dim, dtype=np.float32),
        "lengths": []
    })

    with open(domtblout) as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.split()
            protein = parts[0]
            pfam = parts[1]

            ali_start = int(parts[15])
            ali_end = int(parts[16])
            aln_len = ali_end - ali_start + 1

            if pfam in pfam_vocab:
                raw[protein]["vec"][pfam_vocab[pfam]] = 1.0

            raw[protein]["lengths"].append(aln_len)

    features = {}

    for prot, data in raw.items():
        lengths = data["lengths"]

        summary = np.array([
            len(lengths),
            max(lengths) if lengths else 0,
            sum(lengths),
            len(set(lengths)),
        ], dtype=np.float32)

        features[prot] = np.concatenate([data["vec"], summary])

    return features


def extract_pfam_features(fasta, pfam_hmm, workdir, pfam_vocab=None):
    """End-to-end Pfam feature extraction.
    
    Args:
        fasta: Path to input FASTA file
        pfam_hmm: Path to Pfam-A.hmm database
        workdir: Working directory for intermediate files
        pfam_vocab: Optional pre-built vocabulary (for inference).
                   If None, builds vocabulary from domtblout (training).
        
    Returns:
        Tuple of (features_dict, pfam_vocab)
        - features_dict: Dictionary mapping protein_id -> feature vector
        - pfam_vocab: Vocabulary dictionary (newly built or reused)
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    domtblout = workdir / "pfam.domtblout"
    run_hmmscan(fasta, pfam_hmm, str(domtblout))

    if pfam_vocab is None:
        pfam_vocab = build_pfam_vocab(domtblout)

    features = parse_pfam(domtblout, pfam_vocab)

    return features, pfam_vocab
