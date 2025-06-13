import networkx as nx
import numpy as np
from itertools import combinations, permutations
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ---------- Core Utilities ----------

def is_independent_set(G, nodes):
    return all(not G.has_edge(u, v) for u, v in combinations(nodes, 2))

def all_independent_sets(G):
    nodes = list(G.nodes)
    result = []
    for r in range(1, len(nodes)+1):
        for subset in combinations(nodes, r):
            if is_independent_set(G, subset):
                result.append(set(subset))
    return result

def neighborhood(G, nodes):
    return set().union(*(G.neighbors(n) for n in nodes))

def get_V2(G):
    return {node for node in G.nodes if not G.has_edge(node, node)}

def is_stable(G, mu):
    indep_sets = all_independent_sets(G)
    for I in indep_sets:
        sum_mu_I = sum(mu[i] for i in I)
        sum_mu_NI = sum(mu[j] for j in neighborhood(G, I))
        if sum_mu_I >= sum_mu_NI - 1e-8:
            return False
    return True

# ---------- Expectation Computation ----------

def compute_T_Io(G, ordered, mu, V2):
    prod = 1.0
    S = []
    for i in ordered:
        S.append(i)
        sum_neighbors = sum(mu[j] for j in neighborhood(G, S))
        sum_S_V2 = sum(mu[j] for j in S if j in V2)
        denom = sum_neighbors - sum_S_V2
        if denom <= 0:
            return 0
        prod *= mu[i] / denom
    return prod

def compute_E_Io(G, ordered, mu, V2):
    prod = 1.0
    summ = 0.0
    S = []
    for i in ordered:
        S.append(i)
        sum_neighbors = sum(mu[j] for j in neighborhood(G, S))
        sum_S_V2 = sum(mu[j] for j in S if j in V2)
        denom = sum_neighbors - sum_S_V2
        if denom <= 0:
            return 0
        prod *= mu[i] / denom
        summ += sum_neighbors / denom
    return prod * summ

def compute_T_I(G, I, mu, V2):
    return sum(compute_T_Io(G, list(p), mu, V2) for p in permutations(I))

def compute_E_I(G, I, mu, V2):
    return sum(compute_E_Io(G, list(p), mu, V2) for p in permutations(I))

def compute_expectation(G, mu):
    V2 = get_V2(G)
    indep_sets = all_independent_sets(G)
    T = sum(compute_T_I(G, I, mu, V2) for I in indep_sets)
    E = sum(compute_E_I(G, I, mu, V2) for I in indep_sets)
    denom = 1 + T
    return E / denom if denom > 1e-12 else 0

# ---------- Parallel Sampling Core ----------

def sample_and_check(args):
    G, G_mod, nodes = args

    mu_vals = np.random.dirichlet(np.ones(len(nodes)))
    mu = dict(zip(nodes, mu_vals))

    if not is_stable(G, mu):
        return None

    eq_base = compute_expectation(G, mu)
    eq_mod = compute_expectation(G_mod, mu)

    if eq_mod > eq_base:
        return (mu, eq_base, eq_mod)

    return None

# ---------- Parallel Search for Paradoxical μ ----------

def find_paradox_mu(G, added_edge, trials=10000):
    G_mod = G.copy()
    G_mod.add_edge(*added_edge)
    nodes = list(G.nodes)

    print(f"Searching for paradox over {trials} samples using {cpu_count()} cores...")

    with Pool(processes=cpu_count()) as pool:
        args = [(G, G_mod, nodes)] * trials
        for result in tqdm(pool.imap_unordered(sample_and_check, args, chunksize=100), total=trials):
            if result is not None:
                mu, eq_base, eq_mod = result
                print(f"\n Paradox found:")
                print(f"  E[Q] original = {eq_base:.6f}")
                print(f"  E[Q] after adding edge = {eq_mod:.6f}")
                for k in sorted(mu):
                    print(f"  mu[{k}] = {mu[k]:.6f}")
                pool.terminate()
                return mu

    print("No paradox found in given trials.")
    return None

# ---------- Example Run ----------

if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([
        (1,2), (2,3), (3,4),
        (4,5), (5,6), (6,7), (7,1)  # self-loops
    ])
    added_edge = (1, 5)

    find_paradox_mu(G, added_edge, trials=50000)
