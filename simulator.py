import networkx as nx
from itertools import combinations, permutations

# -- Independent Set Utilities --

def is_independent_set(adj, nodes):
    # Check if no edge exists between any pair in the subset
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
    # Union of neighbors of all nodes in the set
    return set().union(*(adj[n] for n in nodes)) - set(nodes)

def get_V2(adj):
    # Nodes without self-loops
    return {node for node, nbrs in adj.items() if node not in nbrs}

# -- Stability Check --

def is_stable(adj, mu, indep_sets):
    for I in indep_sets:
        sum_mu_I = sum(mu[i] for i in I)
        sum_mu_EI = sum(mu[j] for j in neighborhood(adj, I))
        if sum_mu_I >= sum_mu_EI:
            print(f"Not stable: sum_mu({I}) = {sum_mu_I:.4f} >= neighbors' sum = {sum_mu_EI:.4f}")
            return False
    return True

# -- Expectation Computation --

def compute_T_Io(adj, ordered, mu, V2):
    prod = 1.0
    S = []
    for i in ordered:
        S.append(i)
        sum_neighbors = sum(mu[j] for j in neighborhood(adj, S))
        sum_S_V2 = sum(mu[j] for j in S if j in V2)
        denom = sum_neighbors - sum_S_V2
        if denom <= 0:
            return 0
        prod *= mu[i] / denom
    return prod

def compute_E_Io(adj, ordered, mu, V2):
    prod = 1.0
    summ = 0.0
    S = []
    for i in ordered:
        S.append(i)
        sum_neighbors = sum(mu[j] for j in neighborhood(adj, S))
        sum_S_V2 = sum(mu[j] for j in S if j in V2)
        denom = sum_neighbors - sum_S_V2
        if denom <= 0:
            return 0
        prod *= mu[i] / denom
        summ += sum_neighbors / denom
    return prod * summ

def compute_T_I(adj, I, mu, V2):
    return sum(compute_T_Io(adj, list(p), mu, V2) for p in permutations(I))

def compute_E_I(adj, I, mu, V2):
    return sum(compute_E_Io(adj, list(p), mu, V2) for p in permutations(I))

def compute_expectation(adj, mu, indep_sets):
    V2 = get_V2(adj)
    T = sum(compute_T_I(adj, I, mu, V2) for I in indep_sets)
    E = sum(compute_E_I(adj, I, mu, V2) for I in indep_sets)
    denom = 1 + T
    return E / denom if denom > 1e-12 else 0

def find_bottleneck_set(adj, mu, indep_sets):
    min_delta = float('inf')
    bottleneck_sets = []

    for I in indep_sets:
        if not I:
            continue
        sum_mu_I = sum(mu[i] for i in I)
        sum_mu_EI = sum(mu[j] for j in neighborhood(adj, I))
        delta = sum_mu_EI - sum_mu_I

        if delta < min_delta - 1e-9:
            min_delta = delta
            bottleneck_sets = [I]
        elif abs(delta - min_delta) < 1e-9:
            bottleneck_sets.append(I)

    bottleneck_set = max(bottleneck_sets, key=len)
    return bottleneck_set, min_delta

# -- Runner --

def run(G, mu, label="Graph"):
    print(f"\n--- {label} ---")

    # Precompute adjacency dictionary
    adj = {n: set(G.neighbors(n)) | {n} if G.has_edge(n, n) else set(G.neighbors(n)) for n in G.nodes}
    indep_sets = all_independent_sets(adj)

    if not is_stable(adj, mu, indep_sets):
        print("Not stable. Skipping computation.")
        return

    bottleneck_set, min_delta = find_bottleneck_set(adj, mu, indep_sets)
    eq = compute_expectation(adj, mu, indep_sets)

    print(f"  Bottleneck set: {bottleneck_set}")
    print(f"  Bottleneck: {min_delta:.6f}")
    print(f"  E[Q] = {eq:.6f}")

    return bottleneck_set, min_delta, eq

# -- Example --

if __name__ == "__main__":
    G1 = nx.Graph()
    G1.add_edges_from([
        (1,2), (1,3), (1,4),
        (3,4), (2,5),
        (2,2), (3,3), (4,4), (5,5)  # self-loops
    ])

    mu = {1: 0.008145 , 2: 0.724085, 3: 0.0761 , 4: 0.06936 , 5: 0.12231}

    G1_mod1 = G1.copy(); G1_mod1.add_edge(1,1)
    

    run(G1, mu, label="G1")
    run(G1_mod1, mu, label="G1_mod1")
    
