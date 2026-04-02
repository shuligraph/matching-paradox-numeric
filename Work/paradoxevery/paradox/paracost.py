import random
import math
import csv
import networkx as nx
from itertools import combinations
from typing import Dict, List, Tuple, Optional

# =========================
# -- Graphs: K5 and K5-{1,2}
# =========================
def make_K5():
    G = nx.complete_graph(5)
    mapping = {i: i+1 for i in G.nodes()}
    return nx.relabel_nodes(G, mapping)

def make_K5_minus_12():
    G = make_K5()
    if G.has_edge(1, 2):
        G.remove_edge(1, 2)
    return G

# =========================
# -- Independent Set Utilities (your style)
# =========================
def _adj_with_self_if_any(G: nx.Graph) -> Dict[int, set]:
    # include self only if self-loop exists (your convention)
    return {n: set(G.neighbors(n)) | ({n} if G.has_edge(n, n) else set())
            for n in G.nodes}

def is_independent_set(adj, nodes):
    return all(v not in adj[u] for u, v in combinations(nodes, 2))

def all_independent_sets(adj):
    nodes = list(adj.keys())
    result = []
    for r in range(1, len(nodes) + 1):
        for subset in combinations(nodes, r):
            if is_independent_set(adj, subset):
                result.append(set(subset))
    return result

def neighborhood(adj, nodes):
    # union of neighbors; do NOT subtract nodes (same as your original)
    return set().union(*(adj[n] for n in nodes))

def is_stable(adj, mu, indep_sets):
    # NCOND: sum_mu(I) < sum_mu(Gamma(I)) for every independent set I
    for I in indep_sets:
        sum_mu_I = sum(mu[i] for i in I)
        sum_mu_EI = sum(mu[j] for j in neighborhood(adj, I))
        if sum_mu_I >= sum_mu_EI:
            return False
    return True

def stable_both_graphs(mu: Dict[int, float]) -> bool:
    G_full = make_K5()
    G_minus = make_K5_minus_12()
    for G in (G_full, G_minus):
        adj = _adj_with_self_if_any(G)
        indep = all_independent_sets(adj)
        if not is_stable(adj, mu, indep):
            return False
    return True

# =========================
# -- Closed-form Expected Holding Cost (from Section 3.1/3.2)
# =========================
def expected_cost_K5(mu: Dict[int, float], c: Dict[int, float]) -> float:
    """
    E[K_5, mu, c]
    pi0 = (1 + sum_i mu_i/(mu(Gamma_i)-mu_i))^{-1}
    mu(Gamma_i) = total - mu_i ; denom = total - 2*mu_i
    E = pi0 * sum_i c_i * (mu(Gamma_i)*mu_i)/(denom^2)
    """
    nodes = sorted(mu.keys())
    total = sum(mu[i] for i in nodes)

    # Kn stability: mu_i < total - mu_i  <=>  total - 2*mu_i > 0
    if any((total - 2*mu[i]) <= 0 for i in nodes):
        return float('inf')

    denom_terms = [mu[i] / (total - 2*mu[i]) for i in nodes]
    pi0 = 1.0 / (1.0 + sum(denom_terms))

    s = 0.0
    for i in nodes:
        denom = total - 2*mu[i]
        s += c[i] * (total - mu[i]) * mu[i] / (denom * denom)
    return pi0 * s

def expected_cost_K5_minus_12(mu: Dict[int, float], c: Dict[int, float]) -> float:
    """
    E[K_5 - {1,2}, mu, c]
    Let S = sum_{j>=3} mu_j.
    pi0 = ( 1 + mu1/(S-mu1) + mu2/(S-mu2)
              + sum_{i=3..5} mu_i/(total-2*mu_i)
              + mu1*mu2/((S-mu1)(S-mu2)) )^{-1}
    E = pi0 * [ sum_{i=3..5} c_i*(total-mu_i)*mu_i/(total-2*mu_i)^2
                + S*(c1*mu1 + c2*mu2)/(S - mu1 - mu2)^2 ]
    """
    nodes = sorted(mu.keys())
    total = sum(mu[i] for i in nodes)
    S = sum(mu[i] for i in nodes if i not in (1, 2))

    # necessary denominators > 0 (these come from NCOND on {1}, {2}, {1,2})
    if not (mu[1] < S and mu[2] < S and (mu[1] + mu[2] < S)):
        return float('inf')

    if any((total - 2*mu[i]) <= 0 for i in nodes if i not in (1, 2)):
        return float('inf')

    # pi0
    part_singletons = (
        mu[1] / (S - mu[1]) +
        mu[2] / (S - mu[2]) +
        sum(mu[i] / (total - 2*mu[i]) for i in nodes if i not in (1, 2))
    )
    part_pair12 = (mu[1] * mu[2]) / ((S - mu[1]) * (S - mu[2]))
    pi0 = 1.0 / (1.0 + part_singletons + part_pair12)

    # cost sum
    term_rest = sum(
        c[i] * (total - mu[i]) * mu[i] / ((total - 2*mu[i]) ** 2)
        for i in nodes if i not in (1, 2)
    )
    denom12 = (S - mu[1] - mu[2])
    if denom12 <= 0:
        return float('inf')
    term_12 = S * (c[1] * mu[1] + c[2] * mu[2]) / (denom12 ** 2)

    return pi0 * (term_rest + term_12)

# =========================
# -- Samplers (mu, c)
# =========================
def sample_mu_dirichlet(n: int, alpha: float = 1.0, total: float = 1.0) -> Dict[int, float]:
    xs = [random.gammavariate(alpha, 1.0) for _ in range(n)]
    s = sum(xs)
    vals = [total * x / s for x in xs]
    return {i+1: vals[i] for i in range(n)}

def sample_mu_heavy_traffic(n: int, center_idx: int, rho: float = 0.49, total: float = 1.0) -> Dict[int, float]:
    assert 0 < rho < 0.5
    vals = [0.0]*n
    vals[center_idx-1] = rho * total
    rest = total - vals[center_idx-1]
    others = [i for i in range(1, n+1) if i != center_idx]
    xs = [random.gammavariate(1.0, 1.0) for _ in others]
    s = sum(xs)
    for j, i in enumerate(others):
        vals[i-1] = rest * xs[j] / s
    return {i+1: vals[i] for i in range(n)}

def sample_costs(n: int, low: float = 0.5, high: float = 3.0, mode: str = "uniform") -> Dict[int, float]:
    if mode == "uniform":
        vals = [random.uniform(low, high) for _ in range(n)]
    elif mode == "lognormal":
        vals = [math.exp(random.gauss(0.0, 0.6)) for _ in range(n)]
        vals = [min(high, max(low, v)) for v in vals]
    else:
        raise ValueError("unknown cost sampling mode")
    return {i+1: vals[i] for i in range(n)}

# =========================
# -- Core test
# =========================
def try_one(mu: Dict[int, float], c: Dict[int, float]) -> Tuple[bool, float, float, float]:
    # stability on BOTH graphs using your NCOND
    if not stable_both_graphs(mu):
        return (False, float('inf'), float('inf'), float('-inf'))

    E_full = expected_cost_K5(mu, c)
    E_minus = expected_cost_K5_minus_12(mu, c)

    if not (math.isfinite(E_full) and math.isfinite(E_minus)):
        return (False, E_full, E_minus, float('-inf'))

    gap = E_full - E_minus
    return (gap > 0.0, E_full, E_minus, gap)

# =========================
# -- Monte Carlo search (returns ALL paradox samples)
# =========================
def monte_carlo_search(
    n_trials: int = 50000,
    mode_mu: str = "mixed",   # "dirichlet" | "heavy" | "mixed"
    mode_c: str = "uniform",
    seed: int = 12345,
    keep_top_k: Optional[int] = None,     # None => keep ALL
    save_csv: Optional[str] = None,       # e.g., "paradox_k5.csv"
    eps_gap: float = 1e-12
):
    random.seed(seed)
    solutions = []  # (gap, mu, c, E_full, E_minus)

    for _ in range(n_trials):
        # sample mu
        if mode_mu == "dirichlet":
            mu = sample_mu_dirichlet(5, alpha=1.0, total=1.0)
        elif mode_mu == "heavy":
            center = random.choice([3, 4, 5])
            mu = sample_mu_heavy_traffic(5, center_idx=center,
                                         rho=random.uniform(0.44, 0.495), total=1.0)
        elif mode_mu == "mixed":
            if random.random() < 0.6:
                center = random.choice([3, 4, 5])
                mu = sample_mu_heavy_traffic(5, center_idx=center,
                                             rho=random.uniform(0.44, 0.495), total=1.0)
            else:
                mu = sample_mu_dirichlet(5, alpha=1.0, total=1.0)
        else:
            raise ValueError("unknown mode_mu")

        c = sample_costs(5, low=0.5, high=3.0, mode=mode_c)

        ok, E_full, E_minus, gap = try_one(mu, c)
        if ok and gap > eps_gap:
            solutions.append( (gap, mu, c, E_full, E_minus) )

    # sort by gap descending
    solutions.sort(key=lambda x: x[0], reverse=True)

    # keep all or top-k
    kept = solutions if keep_top_k is None else solutions[:keep_top_k]

    # optional save
    if save_csv is not None:
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gap", "E_K5", "E_K5_minus_12",
                             "mu1","mu2","mu3","mu4","mu5",
                             "c1","c2","c3","c4","c5"])
            for gap, mu, c, E_full, E_minus in kept:
                row = [gap, E_full, E_minus] + \
                      [mu[i] for i in range(1,6)] + \
                      [c[i] for i in range(1,6)]
                writer.writerow(row)

    return kept

# =========================
# -- Pretty print
# =========================
def fmt_vec(d: Dict[int, float]) -> str:
    keys = sorted(d.keys())
    return "[" + ", ".join(f"{k}:{d[k]:.6f}" for k in keys) + "]"

def dump_solutions(solutions):
    if not solutions:
        print("No paradox found.")
        return
    for idx, (gap, mu, c, E_full, E_minus) in enumerate(solutions, 1):
        print(f"\n#{idx}  GAP = E[K5]-E[K5-12] = {gap:.6f}")
        print(f"  mu = {fmt_vec(mu)}")
        print(f"  c  = {fmt_vec(c)}")
        print(f"  E[K5]     = {E_full:.6f}")
        print(f"  E[K5-12]  = {E_minus:.6f}")

# =========================
# -- Runner
# =========================
if __name__ == "__main__":
    sols = monte_carlo_search(
        n_trials=100000,          # 多试几次，找“尽可能多”的悖论样本
        mode_mu="mixed",          # "dirichlet" | "heavy" | "mixed"
        mode_c="uniform",         # or "lognormal"
        seed=20250824,
        keep_top_k=None,          # None => 保留全部
        save_csv="paradox_k5.csv" # 保存CSV，后续可筛选/画图
    )
    dump_solutions(sols)
    print(f"\nTotal paradox samples: {len(sols)}")
