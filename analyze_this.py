from cafa6_parse import load_train_terms
from collections import defaultdict

def analyze_ia(path: str):
    print("Loading train labels...")
    train_terms = load_train_terms("Train/train_terms.tsv")
    print("Collecting GO vocabulary...")
    all_gos = set()
    for gos in train_terms.values():
        all_gos |= gos
    go2idx = {go:i for i,go in enumerate(sorted(all_gos))}
    print("GO terms:", len(go2idx))
    zero = 0
    small = 0     # 0 < IA < 0.5
    large = 0     # IA >= 0.5
    total = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            go_id, ia_str = line.split()
            if(go_id not in go2idx):
                continue
            ia = float(ia_str)

            total += 1

            if ia == 0.0:
                zero += 1
            elif ia < 0.5:
                small += 1
            else:
                large += 1

    print("========== IA STATS ==========")
    print(f"Total GO terms:        {total}")
    print(f"IA == 0:               {zero}")
    print(f"0 < IA < 0.5:          {small}")
    print(f"IA >= 0.5:             {large}")
    print("==============================")

    
    # Optional: percentages
    # if total > 0:
    #     print("\nPercentages:")
    #     print(f"IA == 0:      {100 * zero / total:.2f}%")
    #     print(f"0 < IA < 0.5: {100 * small / total:.2f}%")
    #     print(f"IA >= 0.5:    {100 * large / total:.2f}%")

def count_taxa(fasta_path: str, top_k: int = 100):
    taxa_counts = defaultdict(int)
    total = 0

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Example: >A0A0C5B5G6 9606
                header = line[1:]
                parts = header.split()

                if len(parts) < 2:
                    raise ValueError(f"Bad header format: {line}")

                taxon_id = parts[1]

                taxa_counts[taxon_id] += 1
                total += 1

    print("=========== TAXA DISTRIBUTION ===========")
    print(f"Total sequences: {total}")
    print(f"Unique taxa: {len(taxa_counts)}")
    print()

    # Sort by count descending
    sorted_taxa = sorted(taxa_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"Top {top_k} taxa by sequence count:\n")

    for taxon, cnt in sorted_taxa[:top_k]:
        pct = 100 * cnt / total
        print(f"Taxon {taxon}: {cnt} ({pct:.2f}%)")

    print("========================================")

    return taxa_counts


if __name__ == "__main__":
    analyze_ia("IA.tsv")   # <-- change path if needed
    count_taxa("Test/testsuperset.fasta", 100)  # <-- change path
