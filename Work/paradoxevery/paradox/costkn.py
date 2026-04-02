import numpy as np

def compute_E(vec):
    """
    vec: list or array, 前 n 个是 μ_i, 后 n 个是 c_i
    返回: E[K_n, μ, c]
    """
    n = len(vec) // 2
    mu = np.array(vec[:n], dtype=float)
    c  = np.array(vec[n:], dtype=float)

    mu_gamma = 1 - mu  # 根据定义 μ(Γ_i) = 1 - μ_i
    denom = mu_gamma - mu

    if np.any(denom == 0):
        raise ZeroDivisionError("某项出现分母为 0：μ(Γ_i) - μ_i = 0。")

    # π_∅
    pi_empty = 1.0 / (1.0 + np.sum(mu / denom))

    # E
    E = pi_empty * np.sum(c * mu_gamma * mu / (denom ** 2))
    return E

# 示例
vec = [0.2123089053707668,0.2742714546697317,0.006999635670957356,0.009689658354828117,0.4967303459337161,1,1,1,1,1]  # 前5个是 μ，后5个是 c
print(compute_E(vec))
