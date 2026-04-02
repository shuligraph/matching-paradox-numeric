import numpy as np
from math import log, ceil
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm
import csv

NUM_ALPHAS = 5
EPSILON = 0.001
DELTA_VALUES = np.linspace(1e-6, 1e-5, 10)


@njit
def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

@njit
def estimate_N_case1(epsilon, m):
    target = log(m / epsilon)
    Ni = 2
    while Ni * log(Ni) - Ni < target:
        Ni += 1
    return Ni

@njit
def compute_N_numba(epsilon, delta, alphas):
    m = len(alphas)
    if m == 0:
        return 1
    max_N = 0
    for alpha in alphas:
        beta = 1.0 - alpha
        if alpha < delta:
            Ni = estimate_N_case1(epsilon, m)
            while Ni > 1 and 1 / factorial(Ni - 1) < epsilon / m:
                Ni -= 1
            Ni += 1
        else:
            P = int(np.ceil(beta / delta))
            r = alpha / (P * delta)
            if r >= 1:
                return -1
            A = (alpha / (beta + delta)) ** P
            C = (epsilon / m) * (1 - r)
            logA = P * log(alpha / (beta + delta))
            logC = log(C)
            logR = log(r)
            Ni_est = P + (logC - logA) / logR
            Ni = ceil(Ni_est)
            while Ni > P:
                tail = A * (r ** (Ni - P)) / (1 - r)
                if tail < epsilon / m:
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

def generate_valid_alphas(num):
    while True:
        values = np.random.uniform(0.01, 0.49, size=num)
        values[1] = values[0]  # ensure alpha1 == alpha2
        values /= values.sum()
        if np.all(values < 0.5) and np.all(values > 0):
            return values.tolist()

def run_trial_once(trial_id):
    results = []
    alphas = generate_valid_alphas(NUM_ALPHAS)
    for delta in DELTA_VALUES:
        try:
            alphas_np = np.array(alphas)
            N_full = compute_N_numba(EPSILON, delta, alphas_np)
            if N_full == -1:
                continue
            piL, _ = compute_pi_bounds(alphas_np, delta, EPSILON, N_full)
            E_lower, _ = compute_expectation_bounds(alphas_np, delta, EPSILON, N_full, piL, piL)

            # merge first two alphas
            merged = [alphas[0] + alphas[1]] + alphas[2:]
            merged = np.array(merged)
            merged /= merged.sum()

            N_merge = compute_N_numba(EPSILON, delta, merged)
            if N_merge == -1:
                continue
            _, piU = compute_pi_bounds(merged, delta, EPSILON, N_merge)
            _, E_upper = compute_expectation_bounds(merged, delta, EPSILON, N_merge, piU, piU)

            if E_lower > E_upper:
                results.append({
                    "alphas": alphas,
                    "delta": delta,
                    "E_lower": E_lower,
                    "E_upper": E_upper
                })
        except Exception:
            continue
    return results

def run_parallel_trials(n_trials=1000, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_trial_once)(i) for i in tqdm(range(n_trials), desc="Running")
    )
    return [r for group in results for r in group]

def save_results(results, csv_file="results.csv"):
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["delta"] + [f"alpha{i+1}" for i in range(NUM_ALPHAS)] + ["E_lower", "E_upper"])
        for r in results:
            row = [r["delta"]] + r["alphas"] + [r["E_lower"], r["E_upper"]]
            writer.writerow(row)

if __name__ == "__main__":
    results = run_parallel_trials(n_trials=1000, n_jobs=-1)
    print(f"\n✅ Found {len(results)} successful alpha sets\n")
    for i, r in enumerate(results):
        print(f"[{i+1}] delta={r['delta']:.2e}, E_lower={r['E_lower']:.4f} > E_upper={r['E_upper']:.4f}")
    save_results(results)
    print("\n📄 Results saved to results.csv")