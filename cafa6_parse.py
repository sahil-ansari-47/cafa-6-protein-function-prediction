def read_fasta(path):
    seqs = {}
    current = None
    for line in open(path):
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            pid = line.split("|")[1] if "|" in line else line[1:].split()[0]
            current = pid
            seqs[pid] = ""
        else:
            seqs[current] += line.strip()
    return seqs
# =========================
# GO PARSER
# =========================
def parse_go_obo(path):
    parents = {}
    namespace = {}
    cur = None
    for line in open(path):
        line = line.strip()
        if line == "[Term]":
            cur = None
        elif line.startswith("id:"):
            cur = line.split("id:")[1].strip()
            parents[cur] = []
        elif cur and line.startswith("namespace:"):
            namespace[cur] = line.split("namespace:")[1].strip()
        elif cur and line.startswith("is_a:"):
            p = line.split("is_a:")[1].split("!")[0].strip()
            parents[cur].append(p)
    return parents, namespace
# =========================
# GO PROPAGATION
# =========================
def propagate(scores, parents):
    out = dict(scores)
    for go, sc in list(scores.items()):
        stack = [go]
        while stack:
            g = stack.pop()
            for p in parents.get(g, []):
                if p not in out or out[p] < sc:
                    out[p] = sc
                    stack.append(p)
    return out
def load_train_terms(path):
    ann = {}
    for line in open(path):
        pid, go, ont = line.strip().split("\t")
        ann.setdefault(pid, set()).add(go)
    return ann