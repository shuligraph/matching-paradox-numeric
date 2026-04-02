import numpy as np
from math import log, ceil
from numba import njit

@njit
def factorial(m):
    result = 1
    for i in range(2, m + 1):
        result *= i
    return result

@njit
def estimate_N_case1(epsilon, n):
    target = log(n / epsilon)
    Ni = 2
    while Ni * log(Ni) - Ni < target:
        Ni += 1
    return Ni

@njit
def compute_N_numba(epsilon, delta, alphas):
    n = len(alphas)
    if n == 0:
        return 1
    max_N = 0
    for alpha in alphas:
        beta = 1.0 - alpha
        if alpha < delta:
            Ni = estimate_N_case1(epsilon, n)
            while Ni > 1 and 1 / factorial(Ni - 1) < epsilon / n:
                Ni -= 1
            Ni += 1
        else:
            P = int(np.ceil(beta / delta))
            r = alpha / (P * delta)
            if r >= 1:
                raise ValueError("alpha_i / (P * delta) must be < 1 for convergence.")
            A = (alpha / (beta + delta)) ** P
            C = (epsilon / n) * (1 - r)
            logA = P * log(alpha / (beta + delta))
            logC = log(C)
            logR = log(r)
            Ni_est = P + (logC - logA) / logR
            Ni = ceil(Ni_est)
            while Ni > P:
                tail = A * (r ** (Ni - P)) / (1 - r)
                if tail < epsilon / n:
                    Ni -= 1
                else:
                    break
            Ni += 1
        max_N = max(max_N, Ni)
    return max_N

@njit
def compute_pi_bounds(alphas, delta, epsilon, N):
    n = len(alphas)
    sum_trunc = 0.0
    for i in range(n):
        alpha = alphas[i]
        beta = 1.0 - alpha
        inner_sum = 0.0
        for m in range(1, N + 1):
            prod = 1.0
            for k in range(1, m + 1):
                prod *= alpha / (beta + k * delta)
            inner_sum += prod
        sum_trunc += inner_sum
    lower_bound = 1.0 / (sum_trunc + epsilon + 1.0)
    upper_bound = 1.0 / (sum_trunc + 1.0)
    return lower_bound, upper_bound

@njit
def compute_expectation_bounds(alphas, delta, epsilon, N, pi_lower, pi_upper):
    n = len(alphas)
    expect_lower = 0.0
    expect_upper = 0.0
    for i in range(n):
        alpha = alphas[i]
        beta = 1.0 - alpha
        sum_lower = 0.0
        sum_upper = 0.0
        for m in range(1, N + 1):
            prod = 1.0
            for k in range(1, m + 1):
                prod *= alpha / (beta + k * delta)
            sum_lower += m * prod
            sum_upper += m * prod
        expect_lower += pi_lower * sum_lower
        expect_upper += pi_upper * sum_upper + epsilon / n
    return expect_lower, expect_upper



# Example usage
epsilon = 0.001
delta = 1.000000e-06
b=0.06359026378676357+0.06359026378676357
c=b+0.21134413059978402+0.20759193059345998+0.4538834112332289
alphas = [b,0.21134413059978402,0.20759193059345998,0.4538834112332289]

# Step 1: Compute N
N = compute_N_numba(epsilon, delta, alphas)

# Step 2: Compute bounds for pi_0
pi_lower, pi_upper = compute_pi_bounds(alphas, delta, epsilon, N)

# Step 3: Compute bounds for expected total number of items
expect_lower, expect_upper = compute_expectation_bounds(alphas, delta, epsilon, N, pi_lower, pi_upper)

# Output results
print("Computed N =", N)
print("Lower bound for pi_0:", pi_lower)
print("Upper bound for pi_0:", pi_upper)
print("Lower bound for E:", expect_lower)
print("Upper bound for E:", expect_upper)
print(c)
