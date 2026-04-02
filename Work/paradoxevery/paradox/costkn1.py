import numpy as np

def E_minus_12(vec):
    """
    vec: list/array，前 n 个是 μ_i，后 n 个是 c_i；要求 n >= 2
    返回：E[K_n - {1,2}, μ, c]
    """
    n = len(vec) // 2
    if n < 2 or len(vec) != 2 * n:
        raise ValueError("输入向量应为长度 2n，且 n >= 2。")

    mu = np.array(vec[:n], dtype=float)
    c  = np.array(vec[n:], dtype=float)

    # 构造 μ(Γ_i)
    mu_gamma = np.empty(n, dtype=float)
    mu_gamma[0] = mu_gamma[1] = 1.0 - mu[0] - mu[1]           # i=1,2
    if n > 2:
        mu_gamma[2:] = 1.0 - mu[2:]                           # i>=3

    # 各项公共分母 μ(Γ_i) - μ_i
    denom = mu_gamma - mu
    if np.any(denom == 0.0):
        raise ZeroDivisionError("存在分母 0：μ(Γ_i) - μ_i = 0。")
    
     # 额外分母 μ(Γ_1) - μ_1 - μ_2
    d12 = mu_gamma[0] - mu[0] - mu[1]
    if d12 == 0.0:
        raise ZeroDivisionError("出现分母 0：μ(Γ_1) - μ_1 - μ_2 = 0。")

    # π_∅
    pi_inv = 1.0 + np.sum(mu / denom) + (mu[0] * mu[1]) / (denom[0] * d12) + (mu[0] * mu[1]) / (denom[1] * d12)
    if pi_inv == 0.0:
        raise ZeroDivisionError("π_∅ 的倒数为 0。")
    pi_empty = 1.0 / pi_inv

    # 主和式：i=3..n
    term_rest = 0.0
    if n > 2:
        term_rest = np.sum(c[2:] * mu_gamma[2:] * mu[2:] / (denom[2:] ** 2))

    # 合并项：用 i=1 的 μ(Γ_1)（=μ(Γ_2)）
    d12 = mu_gamma[0] - mu[0] - mu[1]  # 即 (1 - μ1 - μ2) - μ1 - μ2 = 1 - 2μ1 - 2μ2
    if d12 == 0.0:
        raise ZeroDivisionError("出现分母 0：μ(Γ_1) - μ_1 - μ_2 = 0。")
    term_12 = mu_gamma[0] * (c[0] * mu[0] + c[1] * mu[1]) / (d12 ** 2)

    # 最终 E
    E = pi_empty * (term_rest + term_12)
    return E

# 示例
vec = [0.2123089053707668,0.2742714546697317,0.006999635670957356,0.009689658354828117,0.4967303459337161,1,1,1,1,1] # 前5个 μ，后5个 c（n=5）
print(E_minus_12(vec))
