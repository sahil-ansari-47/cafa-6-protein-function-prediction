"""
DeepTMHMM feature extraction module (Docker-based).
"""

import subprocess
import numpy as np
from pathlib import Path


def run_deeptmhmm(fasta, outdir, threads=4):
    """
    Run DeepTMHMM via Docker on a FASTA file.

    Args:
        fasta: Path to input FASTA file
        outdir: Output directory for DeepTMHMM results
        threads: Number of CPU threads to use
    """
    fasta = Path(fasta).resolve()
    outdir = Path(outdir).resolve()
    # outdir.mkdir(parents=True, exist_ok=True)

    # We mount the project root so paths are simple
    project_root = fasta.parents[1]  # project/Train/train_sequences.fasta

    cmd = [
    "docker", "run", "--rm",
    "--gpus", "all",
     "--shm-size", "2gb",           # CRITICAL: Prevents cuBLAS memory crashes
    # "-e", "NVIDIA_VISIBLE_DEVICES=all",
    # "-e", "CUDA_LAUNCH_BLOCKING=1",
    "-v", f"{project_root}:/data",
    "interpro/deeptmhmm:1.0",
    "python3", "/data/DeepTMHMM-Academic-License-v1.0/predict.py",
    "--fasta", f"/data/Train/train_sequences.fasta",
    "--output-dir", f"/data/output/train_deeptmhmm"
    # "--fasta", f"/data/{fasta.relative_to(project_root)}",
    # "--output-dir", f"/data/{outdir.relative_to(project_root)}"
]

    subprocess.run(cmd, check=True)


def topology_to_features(topo: str):
    """
    Convert topology string to fixed-length feature vector.
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
    """
    Parse DeepTMHMM output directory and extract features.

    Expects files:
      *.topology
    """
    outdir = Path(outdir)
    features = {}

    for topo_file in outdir.glob("*.topology"):
        protein_id = topo_file.stem
        topo = topo_file.read_text().strip()

        features[protein_id] = topology_to_features(topo)

    return features


def extract_deeptmhmm_features(fasta, outdir):
    """
    End-to-end DeepTMHMM feature extraction.
    """
    run_deeptmhmm(fasta, outdir)
    # return parse_deeptmhmm(outdir)
