"""G.1 有限图上的光子传播。

离散扭转相位波动方程的确定性蛙跳（Leapfrog）整数步更新：

    u_{t+1} = 2 u_t - u_{t-1} + c^2 Δ_G u_t

其中 Δ_G 为图上离散拉普拉斯算子。对于一维闭环链（最近邻），
Δ u[i] = u[i+1] - 2 u[i] + u[i-1]；在临界稳定参数 c^2 = 1 下，
离散色散关系精确为 ω(k) = 2 arcsin(sin(k/2)) = k（第一布里渊区），
群速度恒为 1，故脉冲无畸变传播、波前速度严格等于 1。
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1D 闭环链蛙跳更新
# ---------------------------------------------------------------------------
def leapfrog_1d_ring(
    u0: np.ndarray, u1: np.ndarray, n_steps: int, c2: float = 1.0
) -> np.ndarray:
    """一维周期闭环链上的蛙跳更新。

    Parameters
    ----------
    u0, u1 : (N,) array
        初始两步构型。
    n_steps : int
        总更新步数（返回 n_steps+1 个时刻）。
    c2 : float
        c^2 系数，临界稳定值为 1。

    Returns
    -------
    history : (n_steps+1, N) array
        每一步的场构型。
    """
    N = u0.shape[0]
    u_prev = u0.astype(float).copy()
    u_curr = u1.astype(float).copy()
    history = np.empty((n_steps + 1, N))
    history[0] = u_prev
    if n_steps >= 1:
        history[1] = u_curr
    for t in range(1, n_steps):
        lap = np.roll(u_curr, -1) - 2.0 * u_curr + np.roll(u_curr, 1)
        u_next = 2.0 * u_curr - u_prev + c2 * lap
        u_prev, u_curr = u_curr, u_next
        history[t + 1] = u_curr
    return history


# ---------------------------------------------------------------------------
# 2D 环面蛙跳更新
# ---------------------------------------------------------------------------
def leapfrog_2d_torus(
    u0: np.ndarray, u1: np.ndarray, n_steps: int, c2: float = 0.5
) -> np.ndarray:
    """二维环面格点上的蛙跳更新（5 点拉普拉斯）。

    二维临界稳定 c^2 = 1/2。
    """
    u_prev = u0.astype(float).copy()
    u_curr = u1.astype(float).copy()
    ny, nx = u_curr.shape
    history = np.empty((n_steps + 1, ny, nx))
    history[0] = u_prev
    if n_steps >= 1:
        history[1] = u_curr
    for t in range(1, n_steps):
        lap = (
            np.roll(u_curr, -1, axis=0)
            + np.roll(u_curr, 1, axis=0)
            + np.roll(u_curr, -1, axis=1)
            + np.roll(u_curr, 1, axis=1)
            - 4.0 * u_curr
        )
        u_next = 2.0 * u_curr - u_prev + c2 * lap
        u_prev, u_curr = u_curr, u_next
        history[t + 1] = u_curr
    return history


# ---------------------------------------------------------------------------
# 解析色散关系
# ---------------------------------------------------------------------------
def dispersion_1d(k: np.ndarray, c: float = 1.0) -> np.ndarray:
    """一维离散色散 ω(k) = 2 arcsin(c sin(k/2))。"""
    return 2.0 * np.arcsin(np.clip(c * np.sin(k / 2.0), -1.0, 1.0))


# ---------------------------------------------------------------------------
# 波前速度测量
# ---------------------------------------------------------------------------
def measure_wavefront_speed(
    history: np.ndarray, threshold: float = 0.5
) -> tuple[float, float]:
    """通过脉冲质心随时间的位移线性拟合测得波速。

    使用直接实空间质心，并对跨周期边界的跳变做解绕。

    Returns
    -------
    c_meas, r2
        拟合速度与决定系数。
    """
    n_steps, N = history.shape
    times = np.arange(n_steps)
    x = np.arange(N, dtype=float)
    # 直接质心 <x> = Σ x·u / Σ u
    total = history.sum(axis=1)
    total = np.where(total == 0, 1.0, total)
    centroid = (history @ x) / total
    # 解绕：检测大于 N/2 的跳变
    diff = np.diff(centroid)
    jumps = np.round(diff / N) * N
    centroid[1:] -= np.cumsum(jumps)
    # 线性拟合 centroid = c * t + b
    A = np.vstack([times, np.ones_like(times)]).T
    (c_meas, b_meas), *_ = np.linalg.lstsq(A, centroid, rcond=None)
    fit = c_meas * times + b_meas
    ss_res = float(np.sum((centroid - fit) ** 2))
    ss_tot = float(np.sum((centroid - centroid.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(c_meas), float(r2)


# ---------------------------------------------------------------------------
# 色散关系数值提取
# ---------------------------------------------------------------------------
def extract_dispersion_1d(
    history: np.ndarray, n_modes: int | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """从模的时间序列提取各离散动量模的频率，与解析色散比较。

    利用离散波方程本征模满足的关系：对纯模 z(t)，有
        cos ω = (z(t+1) + z(t-1)) / (2 z(t))
    在所有 t 上取平均得到 cos ω 的稳健估计，再 ω = arccos。
    该方法不受 FFT 频率分辨率限制，可达接近机器精度。

    Returns
    -------
    k, omega_num, max_rel_err
        离散动量、数值频率、与解析色散的最大相对误差。
    """
    n_steps, N = history.shape
    if n_modes is None:
        n_modes = N
    k = 2.0 * np.pi * np.arange(n_modes) / N
    omega_num = np.full(n_modes, np.nan)
    amps = np.zeros(n_modes)
    for m in range(n_modes):
        z = history @ np.exp(-1j * k[m] * np.arange(N))
        amps[m] = float(np.max(np.abs(z)))
    global_max = amps.max()
    for m in range(n_modes):
        if amps[m] < 1e-5 * global_max:
            continue
        z = history @ np.exp(-1j * k[m] * np.arange(N))
        zr = np.real(z)
        z_mid = zr[1:-1]
        d2 = zr[2:] - 2.0 * zr[1:-1] + zr[:-2]
        denom = float(np.sum(z_mid**2))
        if denom < 1e-24:
            continue
        # 瑞利商估计 cos ω：⟨z, Δ²z⟩ = -2(1-cos ω) ⟨z,z⟩
        cos_omega = 1.0 + float(np.sum(z_mid * d2)) / (2.0 * denom)
        cos_omega = float(np.clip(cos_omega, -1.0, 1.0))
        omega_num[m] = float(np.arccos(cos_omega))
    omega_th = dispersion_1d(k)
    valid = ~np.isnan(omega_num)
    rel_err = np.abs(omega_num[valid] - omega_th[valid]) / np.maximum(
        np.abs(omega_th[valid]), 1e-12
    )
    return k, omega_num, float(np.max(rel_err)) if valid.any() else 0.0
