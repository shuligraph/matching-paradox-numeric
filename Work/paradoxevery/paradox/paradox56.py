import math
import numpy as np

def compute_N(epsilon, delta, alphas):
    m = len(alphas)
    Ni_list = []
    for alpha in alphas:
        if alpha < delta:
            Ni = 1
            while math.pi**2 / (6 * math.factorial(Ni - 1)) >= epsilon / m:
                Ni += 1
            Ni_list.append(Ni)
        else:
            p = math.ceil(alpha / delta)
            if p * delta == alpha:
                p += 1
            Ni = p
            r = alpha / (p * delta)
            term1 = (alpha / delta) ** p / math.factorial(p)
            while term1 * (r ** (Ni - p)) / (1 - r) >= epsilon / m:
                  Ni += 1
            Ni_list.append(Ni)
    return max(Ni_list)

def compute_pi_bound(N, epsilon, delta, alphas, lower=False):
    S = 0.0
    for alpha in alphas:
        beta = 1 - alpha
        k_values = np.arange(1, N + 1)
        prod = np.cumprod(alpha / (beta + k_values * delta))
        S += np.sum(prod)
    return 1.0 / (S + (0 if lower else epsilon) + 1)

def compute_EK_bound(pi_bound, N, delta, alphas, upper=False):
    m = len(alphas)
    EK = 0.0
    for alpha in alphas:
        beta = 1 - alpha
        k_values = np.arange(1, N + 1)
        prod = np.cumprod(alpha / (beta + k_values * delta))
        EK += np.sum(k_values * prod) * pi_bound
    return EK + (epsilon / m if upper else 0)

# Example parameters
epsilon = 1e-6
delta = 0.0071
alphas1 = [0.009, 0.009, 0.298, 0.224, 0.237, 0.223]
alphas2 = [alphas1[0] + alphas1[1], alphas1[2], alphas1[3], alphas1[4], alphas1[5]]

# Compute N
N = max(compute_N(epsilon, delta, alphas1), compute_N(epsilon, delta, alphas2))

# Compute empty bounds
pi_lower1 = compute_pi_bound(N, epsilon, delta, alphas2, lower=True)
pi_upper2 = compute_pi_bound(N, epsilon, delta, alphas2, lower=False)

# Compensation for errors
def compute_er(pi_bou, N, delta, alphas):
    S = 1.0
    for alpha in alphas:
        beta = 1 - alpha
        k_values = np.arange(1, N + 1)
        prod = np.cumprod(alpha / (beta + k_values * delta))
        S += np.sum(prod)
    er=(1 - pi_bou * S)
    return er

# Compute E_K
E_K6_lower = ( 1 +  compute_er(pi_lower1, N, delta, alphas1) ) **2 *  compute_EK_bound(pi_lower1, N, delta, alphas1,upper=False)
E_K5_upper = ( 1 +  compute_er(pi_upper2, N, delta, alphas2) ) **2 *  compute_EK_bound(pi_upper2, N, delta, alphas2, upper=True)

print("Lower bound of E[K_4] =", E_K6_lower)
print("Upper bound of E[K_3] =", E_K5_upper)
print(N)