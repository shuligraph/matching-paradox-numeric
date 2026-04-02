import random
import math
import csv
from typing import Dict, List, Tuple, Optional
import networkx as nx
from itertools import combinations

# ============================================================
# Graph builders: K5 and K5 - {1,2}
# ============================================================

def make_K5() -> nx.Graph:
    """Build the complete graph on 5 nodes labeled 1..5."""
    G0 = nx.complete_graph(5)  # nodes 0..4
    return nx.relabel_nodes(G0, {i: i + 1 for i in G0.nodes()})

def make_K5_minus_12() -> nx.Graph:
    """Build K5 with edge (1,2) removed."""
    G = make_K5()
    if G.has_edge(1, 2):
        G.remove_edge(1, 2)
    return G


# ============================================================
# Independent sets utilities (aligned with user's style)
# ============================================================

def _adj_with_self_if_any(G: nx.Graph) -> Dict[int, set]:
    """
    Adjacency dict; include self if there is a self-loop (not present here).
    This matches the user's convention.
    """
    return {
        n: set(G.neighbors(n)) | ({n} if G.has_edge(n, n) else set())
        for n in G.nodes
    }

def is_independent_set(adj: Dict[int, set], nodes: set) -> bool:
    """Check independence: no edge between any pair in nodes."""
    return all((v not in adj[u]) for u, v in combinations(nodes, 2))

def all_independent_sets(adj: Dict[int, set]) -> List[set]:
    """Enumerate all non-empty independent sets."""
    V = list(adj.keys())
    out: List[set] = []
    for r in range(1, len(V) + 1):
        for S in combinations(V, r):
            S = set(S)
            if is_independent_set(adj, S):
                out.append(S)
    return out

def neighborhood(adj: Dict[int, set], nodes: set) -> set:
    """
    Union of neighbors; by user's convention we do not subtract nodes themselves.
    There are no self-loops in our graphs, so this equals the usual open neighborhood.
    """
    return set().union(*(adj[n] for n in nodes))

def is_stable(adj: Dict[int, set], mu: Dict[int, float], indep_sets: List[set]) -> bool:
    """
    NCOND: sum_{i in I} mu_i < sum_{j in Gamma(I)} mu_j for every independent set I.
    """
    for I in indep_sets:
        mu_I = sum(mu[i] for i in I)
        mu_EI = sum(mu[j] for j in neighborhood(adj, I))
        if mu_I >= mu_EI:
            return False
    return True

def stable_on_both_graphs(mu: Dict[int, float]) -> bool:
    """Check NCOND on K5 and K5-{1,2}."""
    for G in (make_K5(), make_K5_minus_12()):
        adj = _adj_with_self_if_any(G)
        indep = all_independent_sets(adj)
        if not is_stable(adj, mu, indep):
            return False
    return True


# ============================================================
# Normalization
# ============================================================

def normalize_mu(mu: Dict[int, float]) -> Dict[int, float]:
    """Return a copy of mu normalized so that sum(mu) == 1 exactly."""
    s = sum(mu.values())
    if s <= 0:
        raise ValueError("Sum of mu must be positive.")
    return {k: v / s for k, v in mu.items()}


# ============================================================
# Closed-form expected holding costs (Sections 3.1 and 3.2)
# ============================================================

def expected_cost_K5(mu: Dict[int, float], c: Dict[int, float]) -> float:
    """
    E[K5, mu, c] for the complete graph.
    pi0 = (1 + sum_i mu_i / (mu(Gamma_i) - mu_i))^{-1}
    mu(Gamma_i) = total - mu_i, denom = total - 2*mu_i
    E = pi0 * sum_i c_i * (mu(Gamma_i)*mu_i) / (denom^2)
    """
    mu = normalize_mu(mu)  # enforce sum(mu) = 1
    nodes = sorted(mu.keys())
    total = sum(mu[i] for i in nodes)  # equals 1.0

    # Kn stability: total - 2*mu_i > 0 for all i
    if any((total - 2 * mu[i]) <= 0 for i in nodes):
        return float('inf')

    denom_terms = [mu[i] / (total - 2 * mu[i]) for i in nodes]
    pi0 = 1.0 / (1.0 + sum(denom_terms))

    s = 0.0
    for i in nodes:
        denom = total - 2 * mu[i]
        s += c[i] * (total - mu[i]) * mu[i] / (denom * denom)
    return pi0 * s


def expected_cost_K5_minus_12(mu: Dict[int, float], c: Dict[int, float]) -> float:
    """
    E[K5 - {1,2}, mu, c] with revised empty-state probability:

    pi0^{-1} = 1
               + mu1/(S - mu1) + mu2/(S - mu2)
               + sum_{i=3..5} mu_i/(total - 2*mu_i)
               + [mu1*mu2/(S - mu1 - mu2)] * [ 1/(S - mu1) + 1/(S - mu2) ]

    E = pi0 * ( sum_{i=3..5} c_i*(total - mu_i)*mu_i/(total - 2*mu_i)^2
                + S*(c1*mu1 + c2*mu2)/(S - mu1 - mu2)^2 )
    where total = sum_i mu_i (we normalize to 1) and S = sum_{i=3..5} mu_i.
    """
    mu = normalize_mu(mu)  # enforce sum(mu) = 1
    nodes = sorted(mu.keys())
    total = sum(mu[i] for i in nodes)  # equals 1.0
    S = sum(mu[i] for i in nodes if i not in (1, 2))

    # Denominator positivity implied by NCOND for {1}, {2}, and {1,2}:
    if not (mu[1] < S and mu[2] < S and (mu[1] + mu[2] < S)):
        return float('inf')

    if any((total - 2 * mu[i]) <= 0 for i in nodes if i not in (1, 2)):
        return float('inf')

    # Revised pi0
    part_singletons = (
        mu[1] / (S - mu[1]) +
        mu[2] / (S - mu[2]) +
        sum(mu[i] / (total - 2 * mu[i]) for i in nodes if i not in (1, 2))
    )
    pair_term = (mu[1] * mu[2]) / (S - mu[1] - mu[2]) * (1.0 / (S - mu[1]) + 1.0 / (S - mu[2]))
    pi0 = 1.0 / (1.0 + part_singletons + pair_term)

    # Expected cost
    term_rest = sum(
        c[i] * (total - mu[i]) * mu[i] / ((total - 2 * mu[i]) ** 2)
        for i in nodes if i not in (1, 2)
    )
    denom12 = (S - mu[1] - mu[2])
    if denom12 <= 0:
        return float('inf')
    term_12 = S * (c[1] * mu[1] + c[2] * mu[2]) / (denom12 ** 2)

    return pi0 * (term_rest + term_12)


# ============================================================
# Samplers for mu and c (sum(mu)=1 normalization is applied later)
# ============================================================

def sample_mu_dirichlet(n: int, alpha: float = 1.0) -> Dict[int, float]:
    """Dirichlet sampling; will be normalized to sum 1."""
    xs = [random.gammavariate(alpha, 1.0) for _ in range(n)]
    s = sum(xs)
    vals = [x / s for x in xs]
    return {i + 1: vals[i] for i in range(n)}

def sample_mu_heavy_traffic(n: int, center_idx: int, rho: float = 0.49) -> Dict[int, float]:
    """
    Put ~rho on one class (to mimic heavy traffic near bottleneck) and randomize the rest.
    Will be normalized to sum 1.
    """
    assert 0 < rho < 0.5
    vals = [0.0] * n
    vals[center_idx - 1] = rho
    remainder = 1.0 - vals[center_idx - 1]
    others = [i for i in range(1, n + 1) if i != center_idx]
    xs = [random.gammavariate(1.0, 1.0) for _ in others]
    s = sum(xs)
    for j, i in enumerate(others):
        vals[i - 1] = remainder * xs[j] / s
    return {i + 1: vals[i] for i in range(n)}

def sample_costs(n: int, low: float = 0.5, high: float = 3.0, mode: str = "uniform") -> Dict[int, float]:
    """Sample holding costs."""
    if mode == "uniform":
        vals = [random.uniform(low, high) for _ in range(n)]
    elif mode == "lognormal":
        vals = [math.exp(random.gauss(0.0, 0.6)) for _ in range(n)]
        vals = [min(high, max(low, v)) for v in vals]
    else:
        raise ValueError("unknown cost sampling mode")
    return {i + 1: vals[i] for i in range(n)}


# ============================================================
# Monte Carlo search
# ============================================================

def try_one(mu: Dict[int, float], c: Dict[int, float]) -> Tuple[bool, float, float, float, Dict[int, float]]:
    """
    Return (is_paradox, E_full, E_minus, gap, mu_normalized)
    with sum(mu_normalized) == 1 exactly.
    """
    mu_norm = normalize_mu(mu)

    # NCOND on both graphs
    if not stable_on_both_graphs(mu_norm):
        return (False, float('inf'), float('inf'), float('-inf'), mu_norm)

    E_full = expected_cost_K5(mu_norm, c)
    E_minus = expected_cost_K5_minus_12(mu_norm, c)

    if not (math.isfinite(E_full) and math.isfinite(E_minus)):
        return (False, E_full, E_minus, float('-inf'), mu_norm)

    gap = E_full - E_minus
    return (gap > 0.0, E_full, E_minus, gap, mu_norm)

def monte_carlo_search(
    n_trials: int = 100000,
    mode_mu: str = "mixed",     # "dirichlet" | "heavy" | "mixed"
    mode_c: str = "uniform",
    seed: int = 123456,
    keep_top_k: Optional[int] = None,     # None => keep ALL paradoxes
    save_csv: Optional[str] = "paradox_k5.csv",
    eps_gap: float = 1e-12
):
    """
    Run Monte Carlo and collect paradox samples (E[K5] > E[K5-{1,2}]).
    Returns the kept list of (gap, mu, c, E_full, E_minus).
    """
    random.seed(seed)
    solutions: List[Tuple[float, Dict[int, float], Dict[int, float], float, float]] = []

    for _ in range(n_trials):
        # sample mu
        if mode_mu == "dirichlet":
            mu = sample_mu_dirichlet(5, alpha=1.0)
        elif mode_mu == "heavy":
            center = random.choice([3, 4, 5])
            mu = sample_mu_heavy_traffic(5, center_idx=center, rho=random.uniform(0.44, 0.495))
        elif mode_mu == "mixed":
            if random.random() < 0.6:
                center = random.choice([3, 4, 5])
                mu = sample_mu_heavy_traffic(5, center_idx=center, rho=random.uniform(0.44, 0.495))
            else:
                mu = sample_mu_dirichlet(5, alpha=1.0)
        else:
            raise ValueError("unknown mode_mu")

        # sample costs
        c = sample_costs(5, low=0.5, high=3.0, mode=mode_c)

        ok, E_full, E_minus, gap, mu_norm = try_one(mu, c)
        if ok and gap > eps_gap:
            solutions.append((gap, mu_norm, c, E_full, E_minus))

    # sort by gap descending
    solutions.sort(key=lambda x: x[0], reverse=True)

    # keep all or top-k
    kept = solutions if keep_top_k is None else solutions[:keep_top_k]

    # optional CSV save
    if save_csv is not None:
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "gap", "E_K5", "E_K5_minus_12",
                "mu1", "mu2", "mu3", "mu4", "mu5",
                "c1", "c2", "c3", "c4", "c5"
            ])
            for gap, mu_n, c, E_full, E_minus in kept:
                row = [gap, E_full, E_minus] + [mu_n[i] for i in range(1, 6)] + [c[i] for i in range(1, 6)]
                writer.writerow(row)

    return kept


# ============================================================
# Pretty printing
# ============================================================

def fmt_vec(d: Dict[int, float]) -> str:
    keys = sorted(d.keys())
    return "[" + ", ".join(f"{k}:{d[k]:.6f}" for k in keys) + "]"

def dump_solutions(solutions: List[Tuple[float, Dict[int, float], Dict[int, float], float, float]],
                   print_top_k: Optional[int] = None):
    """
    Print either all solutions or the top-k by gap.
    """
    if not solutions:
        print("No paradox found.")
        return
    items = solutions if print_top_k is None else solutions[:print_top_k]
    for idx, (gap, mu, c, E_full, E_minus) in enumerate(items, 1):
        print(f"\n#{idx}  GAP = E[K5]-E[K5-12] = {gap:.6f}")
        print(f"  mu = {fmt_vec(mu)}  (sum={sum(mu.values()):.6f})")
        print(f"  c  = {fmt_vec(c)}")
        print(f"  E[K5]     = {E_full:.6f}")
        print(f"  E[K5-12]  = {E_minus:.6f}")
    print(f"\nTotal paradox samples kept: {len(solutions)}")


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":
    # Run a mixed search; keep ALL paradoxes found; also save to CSV.
    solutions = monte_carlo_search(
        n_trials=100000,         # increase for more samples
        mode_mu="mixed",         # "dirichlet" | "heavy" | "mixed"
        mode_c="uniform",        # "uniform" | "lognormal"
        seed=20250825,
        keep_top_k=None,         # None => keep ALL paradoxes
        save_csv="paradox_k5.csv",
        eps_gap=1e-12
    )
    # Print all or only top-k to the console. Here we print top 50 and still save all to CSV.
    dump_solutions(solutions, print_top_k=50)
