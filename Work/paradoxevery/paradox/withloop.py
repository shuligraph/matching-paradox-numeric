import networkx as nx
from itertools import combinations, permutations
import sympy # 引入 sympy 库

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
    return set().union(*(adj[n] for n in nodes))

def get_V2(adj):
    # Nodes without self-loops
    return {node for node, nbrs in adj.items() if node not in nbrs}

def compute_T_Io(adj, ordered, mu, V2):
    prod = 1.0
    S = []
    for i in ordered:
        S.append(i)
        sum_neighbors = sum(mu[j] for j in neighborhood(adj, S))
        sum_S_V2 = sum(mu[j] for j in S if j in V2)
        denom = sum_neighbors - sum_S_V2
        # 移除数值判断，让 SymPy 处理符号分母
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
        # 移除数值判断，让 SymPy 处理符号分母
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
    print("E", E, "T", T) # E 和 T 将是符号表达式
    denom = 1 + T
    # 直接返回符号表达式，不进行数值判断
    return E / denom


def run(G, mu, label="Graph"):
    print(f"\n--- {label} ---")
    adj = {n: set(G.neighbors(n)) | {n} if G.has_edge(n, n) else set(G.neighbors(n)) for n in G.nodes}
    indep_sets = all_independent_sets(adj)
    eq = compute_expectation(adj, mu, indep_sets)
    # 在这里对结果进行简化
    eq_simplified = sympy.simplify(eq)
    print(f"  E[Q] = {eq_simplified}") # 打印简化后的符号表达式
    return eq_simplified # 返回简化后的表达式

# -- Example --
if __name__ == "__main__":
    G1 = nx.Graph()
    G1.add_edges_from([
        (1, 2), (2, 3), (3, 4), (4,5), (5,1), (1,1), (2,2), (3,3), (4,4), (5,5)
    ])
    # 使用 sympy.symbols 定义 a, b, c, d 为符号变量
    a, b, c, d, e = sympy.symbols('a b c d e')
    mu = {1: a, 2: b, 3: c, 4: d, 5: e}
    run(G1, mu, label="G1")
