import pandas as pd
import numpy as np
from math import log, ceil
from numba import njit

# ------------------------ NUMBA 核心计算函数 ------------------------

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
            while Ni > 1 and (alpha ** Ni) / (delta ** Ni * factorial(Ni - 1)) < epsilon / n:
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
        for m in range(1, N + 2):
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
        for m in range(1, N + 2):
            prod = 1.0
            for k in range(1, m + 1):
                prod *= alpha / (beta + k * delta)
            sum_lower += m * prod
            sum_upper += m * prod
        expect_lower += pi_lower * sum_lower
        expect_upper += pi_upper * sum_upper + epsilon / (n * delta)
    return expect_lower, expect_upper

def compute_expectation(alphas, delta, epsilon=1e-5):
    alphas = np.array(alphas)
    N = compute_N_numba(epsilon, delta, alphas)
    pi_l, pi_u = compute_pi_bounds(alphas, delta, epsilon, N)
    exp_l, exp_u = compute_expectation_bounds(alphas, delta, epsilon, N, pi_l, pi_u)
    return exp_l, exp_u

# ------------------------ 主筛选函数 ------------------------

def filter_valid_rows(csv_path, output_path):
    df = pd.read_csv(csv_path)
    valid_rows = []

    for idx, row in df.iterrows():
        try:
            delta = round(row[0], 2)

            alpha1 = row[1]
            alpha2 = row[2]
            alpha3 = row[3]
            alpha4 = row[4]
            alpha5 = row[5]

            # 大图 alpha 向量
            alpha_large = [alpha1, alpha2, alpha3, alpha4, alpha5]
            # 小图 alpha 向量（合并 alpha1 + alpha2）
            alpha_small = [alpha1 + alpha2, alpha3, alpha4, alpha5]

            # 重新计算新期望值
            E_large_lower, _ = compute_expectation(alpha_large, delta)
            _, E_small_upper = compute_expectation(alpha_small, delta)

            # 判断是否满足条件
            if E_large_lower > E_small_upper:
                # 将这两个新期望值写回这一行
                row.iloc[-2] = E_large_lower  # 更新倒数第二列（E_lower）
                row.iloc[-1] = E_small_upper  # 更新最后一列（E_upper）
                valid_rows.append(row)

        except Exception as e:
            print(f"Row {idx} 出错: {e}")
            continue

    result_df = pd.DataFrame(valid_rows)
    result_df.to_csv(output_path, index=False)
    print(f"✅ 已筛选并更新期望值，结果保存至：{output_path}")

# ------------------------ 使用方式 ------------------------
filter_valid_rows("results.csv", "filtered_output.csv")
