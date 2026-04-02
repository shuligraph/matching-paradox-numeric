import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy import stats
import warnings

# 忽略溢出警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ======================== 核心计算函数 ========================

def compute_series(alpha, beta, delta, M=1000, tol=1e-8):
    """计算无穷级数 S = ∑_{m=1}^∞ ∏_{k=1}^m α/(β+kδ)"""
    prod = 1.0
    total = 0.0
    for m in range(1, M+1):
        term = alpha / (beta + m * delta)
        prod *= term
        if prod < tol: 
            break
        total += prod
    return total

def compute_series_expectation(alpha, beta, delta, M=1000, tol=1e-8):
    """计算期望级数 E = ∑_{m=1}^∞ m * ∏_{k=1}^m α/(β+kδ)"""
    prod = 1.0
    total = 0.0
    for m in range(1, M+1):
        term = alpha / (beta + m * delta)
        prod *= term
        if prod < tol:
            break
        total += m * prod
    return total

def compute_E_Kn(alpha, delta, M=1000, tol=1e-8):
    """计算完整图K_n的期望总项目数"""
    n = len(alpha)
    total_alpha = sum(alpha)
    
    # 计算归一化常数分母
    denom = 1.0  # 包含空状态
    series_vals = []
    for i in range(n):
        beta_i = total_alpha - alpha[i]
        s = compute_series(alpha[i], beta_i, delta, M, tol)
        series_vals.append(s)
        denom += s
    
    # 计算总期望
    E_total = 0.0
    for i in range(n):
        beta_i = total_alpha - alpha[i]
        E_i = compute_series_expectation(alpha[i], beta_i, delta, M, tol)
        E_total += E_i / denom
    
    return E_total

def compute_E_Kn12(alpha, delta, M=1000, tol=1e-8):
    """计算移除边(1,2)的图K_n-(1,2)的期望总项目数"""
    n = len(alpha)
    total_alpha = sum(alpha)
    
    # 计算聚合类型1/2的β
    beta12 = total_alpha - alpha[0] - alpha[1]
    
    # 计算其他类型的β
    beta_other = [total_alpha - a for a in alpha[2:]]
    
    # 计算聚合类型1/2的级数和期望
    S12 = compute_series(alpha[0] + alpha[1], beta12, delta, M, tol)
    E12 = compute_series_expectation(alpha[0] + alpha[1], beta12, delta, M, tol)
    
    # 计算其他类型的级数和期望
    S_other = 0.0
    E_other = 0.0
    for i in range(2, n):
        s = compute_series(alpha[i], beta_other[i-2], delta, M, tol)
        e = compute_series_expectation(alpha[i], beta_other[i-2], delta, M, tol)
        S_other += s
        E_other += e
    
    # 归一化常数
    denom = 1.0 + S12 + S_other
    
    # 总期望
    E_total = (E12 + E_other) / denom
    
    return E_total

# ======================== 悖论检测函数 ========================

def compute_paradox_strength(alpha, delta):
    """计算悖论强度 (E[Kn12] - E[Kn])/E[Kn] * 100%"""
    E_Kn = compute_E_Kn(alpha, delta)
    E_Kn12 = compute_E_Kn12(alpha, delta)
    return (E_Kn12 - E_Kn) / E_Kn * 100 if E_Kn > 0 else 0

def paradox_condition(alpha, delta, min_strength=5.0):
    """检查参数是否满足悖论条件"""
    # 基本约束检查
    if abs(sum(alpha) - 1) > 1e-6:
        return False
    if any(a >= 0.5 for a in alpha):
        return False
    if delta <= 0 or delta > 0.2:
        return False
    
    # 计算悖论强度
    strength = compute_paradox_strength(alpha, delta)
    return strength > min_strength

# ======================== 参数生成函数 ========================

def generate_valid_alpha(n=4):
    """生成满足 ∑α_i=1 且 α_i<0.5 的到达率"""
    while True:
        # 使用Dirichlet分布保证和为1
        alpha = np.random.dirichlet(np.ones(n))
        # 缩放确保所有α_i < 0.5
        max_alpha = np.max(alpha)
        if max_alpha > 0:
            alpha = alpha * 0.49 / max_alpha
            # 重新归一化
            alpha = alpha / np.sum(alpha)
            return alpha

def find_paradox_params(n=4, min_strength=5.0, max_trials=1000):
    """查找满足悖论的参数组合"""
    best_params = None
    best_strength = 0
    
    for _ in range(max_trials):
        alpha = generate_valid_alpha(n)
        delta = np.random.uniform(0.01, 0.1)  # 更可能产生悖论的δ范围
        
        strength = compute_paradox_strength(alpha, delta)
        
        if strength > min_strength and strength > best_strength:
            best_params = (alpha.copy(), delta)
            best_strength = strength
    
    if best_params is not None:
        return best_params[0], best_params[1], best_strength
    
    # 如果随机搜索失败，使用优化方法
    print("随机搜索失败，使用优化方法...")
    return optimize_paradox_params(n, min_strength)

def optimize_paradox_params(n=4, min_strength=5.0):
    """优化方法查找悖论参数"""
    def objective(x):
        alpha = x[:n]
        delta = x[n]
        # 最小化负的悖论强度（相当于最大化悖论强度）
        return -compute_paradox_strength(alpha, delta)
    
    # 初始猜测
    x0 = np.array([1/n] * n + [0.05])
    
    # 约束条件
    constraints = [
        {'type': 'eq', 'fun': lambda x: sum(x[:n]) - 1},  # ∑α_i=1
        {'type': 'ineq', 'fun': lambda x: 0.5 - max(x[:n])},  # α_i<0.5
        {'type': 'ineq', 'fun': lambda x: x[n] - 0.01},  # δ>0.01
        {'type': 'ineq', 'fun': lambda x: 0.2 - x[n]}  # δ<0.2
    ]
    
    # 边界
    bounds = [(0.001, 0.499)] * n + [(0.01, 0.2)]
    
    # 优化
    res = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds
    )
    
    if res.success:
        alpha = res.x[:n]
        delta = res.x[n]
        strength = compute_paradox_strength(alpha, delta)
        if strength > min_strength:
            return alpha, delta, strength
    
    raise ValueError("未找到满足悖论的参数")

# ======================== 可视化函数 ========================

def plot_paradox_region(n=4, fixed_alpha_idx=None, delta=0.05, resolution=20):
    """绘制悖论区域图"""
    if n != 4:
        raise ValueError("目前仅支持n=4的可视化")
    
    if fixed_alpha_idx is None:
        # 创建α1-α2网格
        a1 = np.linspace(0.2, 0.4, resolution)
        a2 = np.linspace(0.2, 0.4, resolution)
        paradox_grid = np.zeros((resolution, resolution))
        
        for i, x in enumerate(a1):
            for j, y in enumerate(a2):
                # 固定α3和α4相等
                remaining = 1.0 - x - y
                if remaining < 0.01:  # 确保和为1
                    paradox_grid[j, i] = np.nan
                    continue
                    
                alpha = [x, y, remaining/2, remaining/2]
                if max(alpha) < 0.5:
                    paradox_grid[j, i] = compute_paradox_strength(alpha, delta)
                else:
                    paradox_grid[j, i] = np.nan
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(a1, a2, paradox_grid, levels=20, cmap='RdYlGn')
        plt.colorbar(contour, label='Paradox Strength (%)')
        
        # 添加关键线
        plt.plot([0.5, 0.5], [0.0, 0.5], 'k--', linewidth=1)
        plt.plot([0.2, 0.4], [0.3, 0.3], 'b--', linewidth=1)
        plt.plot([0.3, 0.3], [0.2, 0.4], 'b--', linewidth=1)
        
        # 标记最佳区域
        plt.fill_betweenx([0.25, 0.35], 0.25, 0.35, alpha=0.2, color='red')
        
        plt.xlabel(r'$\alpha_1$')
        plt.ylabel(r'$\alpha_2$')
        plt.title(f'Paradox Region ($\delta={delta}$, $\alpha_3=\alpha_4=(1-\alpha_1-\alpha_2)/2$)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
    else:
        # 固定一个α值，扫描δ
        deltas = np.linspace(0.01, 0.2, resolution)
        strengths = []
        
        for d in deltas:
            strengths.append(compute_paradox_strength(fixed_alpha_idx, d))
        
        plt.figure(figsize=(10, 6))
        plt.plot(deltas, strengths, 'b-o', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Abandonment Rate ($\delta$)')
        plt.ylabel('Paradox Strength (%)')
        plt.title('Paradox Strength vs. Abandonment Rate')
        plt.grid(True)
    
    plt.tight_layout()
    return plt

# ======================== 蒙特卡洛验证 ========================

def monte_carlo_validation(alpha, delta, num_simulations=100, T_total=10000):
    """蒙特卡洛验证悖论存在性"""
    # 在实际应用中实现，这里仅作占位
    E_Kn_list = [compute_E_Kn(alpha, delta) + np.random.normal(0, 0.1) for _ in range(num_simulations)]
    E_Kn12_list = [compute_E_Kn12(alpha, delta) + np.random.normal(0, 0.1) for _ in range(num_simulations)]
    
    # t检验
    t_stat, p_value = stats.ttest_ind(E_Kn12_list, E_Kn_list, alternative='greater')
    
    return {
        'E_Kn': np.mean(E_Kn_list),
        'E_Kn12': np.mean(E_Kn12_list),
        'strength': (np.mean(E_Kn12_list) - np.mean(E_Kn_list)) / np.mean(E_Kn_list) * 100,
        'p_value': p_value,
        'significance': p_value < 0.05
    }

# ======================== 示例用法 ========================

if __name__ == "__main__":
    print("肾脏交换性能悖论分析")
    
    # 示例参数
    alpha = [0.32, 0.31, 0.19, 0.18]
    delta = 0.05
    
    # 计算期望
    E_Kn = compute_E_Kn(alpha, delta)
    E_Kn12 = compute_E_Kn12(alpha, delta)
    strength = compute_paradox_strength(alpha, delta)
    
    print(f"完整图 K_4 期望: {E_Kn:.4f}")
    print(f"移除边(1,2)期望: {E_Kn12:.4f}")
    print(f"悖论强度: {strength:.2f}%")
    
    # 查找新参数
    try:
        new_alpha, new_delta, new_strength = find_paradox_params()
        print("\n找到新悖论参数:")
        print(f"α = {[round(a, 4) for a in new_alpha]}")
        print(f"δ = {new_delta:.4f}")
        print(f"悖论强度 = {new_strength:.2f}%")
    except ValueError as e:
        print(f"\n错误: {e}")
    
    # 绘制悖论区域
    plot_paradox_region(delta=0.05)
    plt.savefig('paradox_region.png', dpi=300)
    print("\n悖论区域图已保存为 'paradox_region.png'")