import math
import pickle
from collections import defaultdict



def nested_int_dict():
    return defaultdict(int)


def init_feature_dict():
    return {
        "protein_length": 0,

        # Tier 1
        "interpro": defaultdict(int),

        # Tier 2
        "signatures": defaultdict(nested_int_dict),

        # Architecture
        "architecture": {
            "num_domains": 0,
            "unique_domains": 0,
            "coverage": 0.0
        },

        # Confidence
        "confidence": {
            "min_log_evalue": None,
            "strong_hit_count": 0
        },

        # Internal
        "_covered_positions": set()
    }


def parse_interpro_tsv(tsv_file: str, out_pkl: str):
    
    proteins = defaultdict(init_feature_dict)

    # ---------------- TSV parsing ----------------
    with open(tsv_file, "r") as f:
        for line in f:
            if not line.strip():
                continue

            cols = line.rstrip("\n").split("\t")

            protein_id = cols[0]
            protein_len = int(cols[2])
            analysis = cols[3]
            signature_id = cols[4]
            start = int(cols[6])
            end = int(cols[7])
            evalue = cols[8]
            interpro_id = cols[11]

            entry = proteins[protein_id]
            entry["protein_length"] = protein_len

            # Architecture
            entry["architecture"]["num_domains"] += 1
            entry["_covered_positions"].update(range(start, end + 1))

            # InterPro features
            if interpro_id != "-":
                entry["interpro"][interpro_id] = 1

            # Signature features
            if signature_id != "-":
                entry["signatures"][analysis][signature_id] += 1

            # Confidence
            if evalue != "-":
                try:
                    val = float(evalue)
                    log_eval = -math.log10(val)

                    if entry["confidence"]["min_log_evalue"] is None:
                        entry["confidence"]["min_log_evalue"] = log_eval
                    else:
                        entry["confidence"]["min_log_evalue"] = min(
                            entry["confidence"]["min_log_evalue"], log_eval
                        )

                    if val <= 1e-5:
                        entry["confidence"]["strong_hit_count"] += 1

                except ValueError:
                    pass

    # ------------- Post-processing -------------
    final_proteins = {}

    for pid, feat in proteins.items():
        prot_len = feat["protein_length"]
        covered = len(feat["_covered_positions"])

        feat["architecture"]["coverage"] = (
            covered / prot_len if prot_len > 0 else 0.0
        )

        feat["architecture"]["unique_domains"] = sum(
            len(v) for v in feat["signatures"].values()
        )

        # REMOVE internal field
        del feat["_covered_positions"]

        # 🔥 CONVERT defaultdict → dict (pickle-safe)
        feat["interpro"] = dict(feat["interpro"])
        feat["signatures"] = {
            k: dict(v) for k, v in feat["signatures"].items()
        }

        final_proteins[pid] = feat

    
    return final_proteins