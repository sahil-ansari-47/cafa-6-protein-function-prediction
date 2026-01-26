# Use a modern PyTorch base that supports RTX 30/40 series GPUs
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install DeepTMHMM dependencies
# We install the specific versions required for the ESM-1b transformer
RUN pip install --no-cache-dir \
    biopython \
    fair-esm \
    scikit-learn \
    pandas \
    matplotlib

# Set the working directory
WORKDIR /app

# The entrypoint is still python3
ENTRYPOINT ["python3"]
