"""DeepTMHMM feature extraction module."""

import subprocess
import numpy as np
from pathlib import Path


def run_deeptmhmm(fasta, outdir, threads=4):
    """Run DeepTMHMM on a FASTA file.
    
    Args:
        fasta: Path to input FASTA file
        outdir: Output directory for DeepTMHMM results
        threads: Number of CPU threads to use
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "deeptmhmm.py",
        "--fasta", fasta,
        "--outdir", str(outdir),
        "--cpu", str(threads),
    ]

    subprocess.run(cmd, check=True)


def topology_to_features(topo: str):
    """Convert topology string to fixed-length feature vector.
    
    Args:
        topo: Topology string (e.g., "MMMMIIIOOOSSS")
        
    Returns:
        numpy array of shape (6,) with features:
        [has_signal, num_tm, frac_tm, n_term_in, c_term_in, L]
    """
    L = len(topo)

    has_signal = 1.0 if "S" in topo[:30] else 0.0
    num_tm = topo.count("M")
    frac_tm = num_tm / L if L > 0 else 0.0

    n_term_in = 1.0 if topo[0] == "I" else 0.0
    c_term_in = 1.0 if topo[-1] == "I" else 0.0

    return np.array(
        [has_signal, num_tm, frac_tm, n_term_in, c_term_in, L],
        dtype=np.float32,
    )


def parse_deeptmhmm(outdir):
    """Parse DeepTMHMM output directory and extract features.
    
    Args:
        outdir: Directory containing DeepTMHMM .topology files
        
    Returns:
        Dictionary mapping protein_id -> feature vector (shape 6,)
    """
    outdir = Path(outdir)
    features = {}

    for topo_file in outdir.glob("*.topology"):
        protein_id = topo_file.stem
        topo = topo_file.read_text().strip()

        features[protein_id] = topology_to_features(topo)

    return features


def extract_deeptmhmm_features(fasta, outdir):
    """End-to-end DeepTMHMM feature extraction.
    
    Args:
        fasta: Path to input FASTA file
        outdir: Output directory for DeepTMHMM results
        
    Returns:
        Dictionary mapping protein_id -> feature vector (shape 6,)
    """
    run_deeptmhmm(fasta, outdir)
    return parse_deeptmhmm(outdir)
