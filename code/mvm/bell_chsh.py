"""G.2 Bell--CHSH 非局域关联与 Tsirelson 界。

在全局离散演化步数下，构造包含长程闭环瞬发同调重构的联合条件响应分布：

    P(a,b | x,y) = (1/4) [1 + a b · E(x,y)],   E(x,y) = -cos(θ_x - φ_y)

取标准最优测量角度 θ = (0, π/2)，φ = (π/4, -π/4)。
在 CHSH 算子 S = |E(0,0) + E(0,1) + E(1,0) - E(1,1)| 下精确达到
Tsirelson 上界 S = 2√2，同时单侧边缘恒为 1/2（无信号）。
"""

from __future__ import annotations

import numpy as np


# 泡利矩阵
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def correlation(theta: float, phi: float) -> float:
    """关联函数 E(x,y) = -cos(θ_x - φ_y)。"""
    return float(-np.cos(theta - phi))


def chsh_value(
    theta: tuple[float, float] = (0.0, np.pi / 2),
    phi: tuple[float, float] = (np.pi / 4, -np.pi / 4),
) -> tuple[float, np.ndarray]:
    """计算 CHSH 关联参数 S 与关联矩阵 E[a,b]。"""
    E = np.array(
        [[correlation(theta[a], phi[b]) for b in range(2)] for a in range(2)]
    )
    # 标准 CHSH 算子：S = |E00 + E01 + E10 - E11|
    S = abs(E[0, 0] + E[0, 1] + E[1, 0] - E[1, 1])
    return float(S), E


def chsh_max_over_signs(E: np.ndarray) -> float:
    """对八种 ± 符号配置取最大 |S|，避免符号约定依赖。"""
    best = 0.0
    for s in [
        (1, 1, 1, -1),
        (1, 1, -1, 1),
        (1, -1, 1, 1),
        (-1, 1, 1, 1),
        (1, 1, -1, -1),
        (1, -1, 1, -1),
        (1, -1, -1, 1),
        (-1, 1, 1, 1),
    ]:
        val = abs(
            s[0] * E[0, 0] + s[1] * E[0, 1] + s[2] * E[1, 0] + s[3] * E[1, 1]
        )
        best = max(best, val)
    return best


def sample_joint(
    theta: tuple[float, float],
    phi: tuple[float, float],
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """从 P(a,b|x,y) 采样并统计 CHSH 与无信号边缘。

    Returns
    -------
    counts : (2,2,2,2) array, counts[a,b,x,y]
    S_emp : 经验 CHSH 值
    margin_err : 单侧边缘偏离 1/2 的最大绝对值
    """
    a_vals = np.array([1, -1])
    counts = np.zeros((2, 2, 2, 2), dtype=int)
    for x in range(2):
        for y in range(2):
            E = correlation(theta[x], phi[y])
            for _ in range(n_samples):
                # P(a,b) = (1/4)(1 + a b E)
                # 先抽 b ~ 1/2，再抽 a | b：P(a|b) = (1 + a b E)/2
                b = 1 if rng.random() < 0.5 else -1
                p_a = 0.5 * (1 + b * E)
                a = 1 if rng.random() < p_a else -1
                ai = 0 if a == 1 else 1
                bi = 0 if b == 1 else 1
                counts[ai, bi, x, y] += 1
    # 经验关联
    E_emp = np.zeros((2, 2))
    for x in range(2):
        for y in range(2):
            n_xy = counts[:, :, x, y].sum()
            ab = np.array([[1, -1], [-1, 1]])
            E_emp[x, y] = float(np.sum(ab * counts[:, :, x, y]) / n_xy)
    S_emp = abs(E_emp[0, 0] + E_emp[0, 1] + E_emp[1, 0] - E_emp[1, 1])
    # 无信号：P(a|x) 不依赖 y，P(b|y) 不依赖 x
    margin_err = 0.0
    for x in range(2):
        p_a_y = np.zeros(2)
        for y in range(2):
            p_a_y[y] = counts[0, :, x, y].sum() / counts[:, :, x, y].sum()
        margin_err = max(margin_err, abs(p_a_y[0] - p_a_y[1]))
    for y in range(2):
        p_b_x = np.zeros(2)
        for x in range(2):
            p_b_x[x] = counts[:, 0, x, y].sum() / counts[:, :, x, y].sum()
        margin_err = max(margin_err, abs(p_b_x[0] - p_b_x[1]))
    return counts, float(S_emp), float(margin_err)
