from typing import Dict, Set

class GODag:
    def __init__(self, obo_path: str):
        """
        Load GO DAG from go-basic.obo.
        Args:
            obo_path: Path to go-basic.obo
        """
        self.parents: Dict[str, Set[str]] = {}
        self.namespaces: Dict[str, str] = {}

        self._load_obo(obo_path)

    def _load_obo(self, obo_path: str):
        """Parse OBO file and populate parents + namespaces."""
        current_term = None
        with open(obo_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue           
                if line == "[Term]":
                    if current_term and not current_term.get("obsolete"):
                        if "id" in current_term and "namespace" in current_term:
                            tid = current_term["id"]
                            self.parents[tid] = current_term["parents"]
                            self.namespaces[tid] = current_term["namespace"]
                    current_term = {"parents": set(), "obsolete": False}
                elif line == "[Typedef]":
                    if current_term and not current_term.get("obsolete"):
                        if "id" in current_term and "namespace" in current_term:
                            tid = current_term["id"]
                            self.parents[tid] = current_term["parents"]
                            self.namespaces[tid] = current_term["namespace"]
                    current_term = None
                elif current_term is not None:
                    if line.startswith("id:"):
                        current_term["id"] = line[3:].strip()
                    elif line.startswith("namespace:"):
                        current_term["namespace"] = line[10:].strip()
                    elif line.startswith("is_a:"):
                        parent = line[5:].split("!")[0].strip()
                        current_term["parents"].add(parent)
                    elif line.startswith("is_obsolete: true"):
                        current_term["obsolete"] = True
        if current_term and not current_term.get("obsolete"):
            if "id" in current_term and "namespace" in current_term:
                tid = current_term["id"]
                self.parents[tid] = current_term["parents"]
                self.namespaces[tid] = current_term["namespace"]

    def get_parents(self, go_id: str) -> Set[str]:
        """Return direct parents of a GO term."""
        return self.parents.get(go_id, set())

    def get_ancestors(self, go_id: str) -> Set[str]:
        """Return all ancestors (excluding self)."""
        if go_id in self._ancestor_cache:
            return self._ancestor_cache[go_id]
        ancestors = set()
        queue = [go_id]
        visited = {go_id}
        while queue:
            node = queue.pop(0)
            for parent in self.parents.get(node, set()):
                if parent not in visited:
                    visited.add(parent)
                    ancestors.add(parent)
                    queue.append(parent)
        self._ancestor_cache[go_id] = ancestors
        return ancestors

    def get_namespace(self, go_id: str) -> str:
        """Return namespace: molecular_function / biological_process / cellular_component."""
        return self.namespaces.get(go_id, "")

if __name__ == "__main__":
    """
    @Sandarva: This is a test of the DAG class to ensure it is working.
    """
    dag = GODag("Train/go-basic.obo")
    print("Total terms:", len(dag.parents))
    from collections import Counter
    ns_counts = Counter(dag.namespaces.values())
    print("Namespaces:", ns_counts)
    # Spot-check
    term = "GO:0019787"  # MF root
    print(term, "parents:", dag.get_parents(term))