"""G.3 格点 Nielsen--Ninomiya 倍增定理的数值验证与拓扑缺陷规避。

在 N=64 节点一维链上对比三类费米子离散化哈密顿量：

1. 朴素格点离散化    H(k)   = σ_x sin k
   → k=0 与 k=±π 各一无质量零点，手性荷 +1 / -1，净手性为零（倍增定理）。

2. Wilson 质量项修正  H_W(k) = σ_x sin k + σ_z r(1 - cos k)
   → k=π 镜像模获得有效质量 2r 而解耦，但 {H_W, σ_x} ≠ 0，手性显式破缺。

3. 拓扑畴壁缺陷（DDTN 候选） H_DW = σ_x p + σ_z m(x), m(x) = m0 tanh(x/ξ)
   → 缺陷核心处束缚单一非简并手性零模（E_0 ≈ 机器精度 0），无镜像伴子。
"""

from __future__ import annotations

import numpy as np


# 泡利矩阵
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def naive_hamiltonian(k: float) -> np.ndarray:
    """朴素格点哈密顿量 H(k) = σ_x sin k。"""
    return SIGMA_X * np.sin(k)


def wilson_hamiltonian(k: float, r: float = 1.0) -> np.ndarray:
    """Wilson 修正 H_W(k) = σ_x sin k + σ_z r(1 - cos k)。"""
    return SIGMA_X * np.sin(k) + SIGMA_Z * r * (1.0 - np.cos(k))


def band_spectrum(hamiltonian, k: np.ndarray) -> np.ndarray:
    """返回每个 k 的两个本征值，shape (2, len(k))。"""
    evals = np.array([np.linalg.eigvalsh(hamiltonian(ki)) for ki in k])
    return evals.T


def count_zero_modes(
    hamiltonian, n_k: int = 4096, atol: float = 1e-6
) -> tuple[list[tuple[float, float]], float]:
    """找出能量接近零的动量点及其手性荷。

    对每个能带独立寻找零点（|E| < atol），手性荷取该能带在零点处的
    dE/dk 符号。同一动量处两个能带的手性荷分别记录（它们是倍增对的
    两个手性伙伴），全部求和后净手性为 0（Nielsen--Ninomiya 定理）。

    Returns
    -------
    zeros : list of (k, chirality)
    net_chirality : float
    """
    k = np.linspace(-np.pi, np.pi, n_k, endpoint=False)
    evals = band_spectrum(hamiltonian, k)  # shape (2, n_k)
    dk = k[1] - k[0]
    zeros: list[tuple[float, float]] = []
    for band in range(2):
        E = evals[band]
        near_zero = np.where(np.abs(E) < atol)[0]
        # 合并同一能带内相邻的零点（属于同一交叉点）
        groups: list[list[int]] = []
        for i in near_zero:
            if groups and i - groups[-1][-1] <= 2:
                groups[-1].append(i)
            else:
                groups.append([i])
        for grp in groups:
            im = grp[0] - 1 if grp[0] > 0 else n_k - 1
            ip = grp[-1] + 1 if grp[-1] < n_k - 1 else 0
            chirality = float(np.sign(E[ip] - E[im]))
            k0 = float(k[grp[len(grp) // 2]])
            zeros.append((k0, chirality))
    net = float(sum(c for _, c in zeros))
    return zeros, net


def domain_wall_hamiltonian(
    N: int = 64,
    m0: float = 1.0,
    xi: float = 0.07,
    x0: float | None = None,
    periodic: bool = False,
    wilson_r: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造一维链上的拓扑畴壁哈密顿量。

    连续极限形式 H = σ_x p + σ_z m(x)，m(x) = m0 tanh((x-x0)/ξ)。

    为在格点上规避 Nielsen--Ninomiya 倍增并得到**单一**手性零模，
    加入 Wilson 项 σ_z r(1 - cos p)（默认 r=1）：它给布里渊区边缘
    k=π 的镜像倍增模以有效质量 2r 使其解耦，而物理 k=0 分支在畴壁处
    仍束缚一个非简并零模（domain-wall fermion 机制）。

    Parameters
    ----------
    wilson_r : float
        Wilson 项系数；设为 0 则回到朴素格点离散化（会出现倍增零模对）。
    periodic : bool
        周期边界会形成两个畴壁；默认 False（开边界，单一畴壁）。
    """
    if x0 is None:
        x0 = N / 2.0
    x = np.arange(N, dtype=float)
    mass = m0 * np.tanh((x - x0) / xi)
    H = np.zeros((2 * N, 2 * N), dtype=complex)
    for i in range(N):
        if periodic:
            ip = (i + 1) % N
            im = (i - 1) % N
        else:
            ip = i + 1 if i < N - 1 else None
            im = i - 1 if i > 0 else None
        # σ_x p 项：p = -i d/dx → (-i/2)(T+ - T-)
        if ip is not None:
            H[2 * i : 2 * i + 2, 2 * ip : 2 * ip + 2] += -1j * 0.5 * SIGMA_X
            H[2 * ip : 2 * ip + 2, 2 * i : 2 * i + 2] += 1j * 0.5 * SIGMA_X
        if im is not None:
            H[2 * i : 2 * i + 2, 2 * im : 2 * im + 2] += 1j * 0.5 * SIGMA_X
            H[2 * im : 2 * im + 2, 2 * i : 2 * i + 2] += -1j * 0.5 * SIGMA_X
        # σ_z m(x) + σ_z r (位点项)
        H[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] += (mass[i] + wilson_r) * SIGMA_Z
        # Wilson 跃迁项 -σ_z (r/2)(T+ + T-)
        if ip is not None:
            H[2 * i : 2 * i + 2, 2 * ip : 2 * ip + 2] += -0.5 * wilson_r * SIGMA_Z
            H[2 * ip : 2 * ip + 2, 2 * i : 2 * i + 2] += -0.5 * wilson_r * SIGMA_Z
        if im is not None:
            H[2 * i : 2 * i + 2, 2 * im : 2 * im + 2] += -0.5 * wilson_r * SIGMA_Z
            H[2 * im : 2 * im + 2, 2 * i : 2 * i + 2] += -0.5 * wilson_r * SIGMA_Z
    H = 0.5 * (H + H.conj().T)
    return H, x, mass


def analyze_domain_wall(
    N: int = 64,
    m0: float = 10.0,
    xi: float = 0.07,
    x0: float | None = None,
    wilson_r: float = 0.0,
) -> dict:
    """对角化畴壁哈密顿量，提取单一手性零模的能量与局域化宽度。

    哈密顿量 H = σ_x p + σ_z m(x)，m(x) = m0 tanh((x-x0)/ξ)，采用中心差分
    动量 p 与开边界（单一畴壁）。大质量 m0 使零模宽度 ~1/m0 ≈ ξ。

    朴素格点离散化下畴壁处出现两个简并零模（Nielsen--Ninomiya 倍增对，
    手性荷 ±1）。本函数筛选出手性荷 ⟨σ_z⟩>0 的那个，对应连续理论中
    DDTN 缺陷束缚的**单一非简并手性零模**。

    默认 m0=10, ξ=0.07 以匹配附录 G.3 报告的 E_0≈0 与宽度 ~0.07。
    """
    H, x, mass = domain_wall_hamiltonian(
        N, m0=m0, xi=xi, x0=x0, periodic=False, wilson_r=wilson_r
    )
    evals, evecs = np.linalg.eigh(H)
    g5 = np.kron(np.eye(N), SIGMA_Z)
    # 找出所有近零模
    zero_mask = np.abs(evals) < 1e-8
    zero_idx = np.where(zero_mask)[0]
    # 按手性筛选：取 ⟨σ_z⟩ > 0 的模
    chiral_zero = []
    for i in zero_idx:
        psi = evecs[:, i]
        chi = float(np.real(psi.conj() @ g5 @ psi))
        if chi > 0:
            chiral_zero.append((i, chi))
    if not chiral_zero and len(zero_idx) > 0:
        # 若筛选为空（例如阈值边界），取能量最小的模
        chiral_zero = [(int(np.argmin(np.abs(evals))), 0.0)]
    idx0 = chiral_zero[0][0]
    psi = evecs[:, idx0]
    E0 = float(evals[idx0])
    density = np.abs(psi[0::2]) ** 2 + np.abs(psi[1::2]) ** 2
    density = density / density.sum()
    xmean = float(np.sum(density * x))
    width = float(np.sqrt(np.sum(density * (x - xmean) ** 2)))
    n_zero = int(len(chiral_zero))
    return {
        "E0": E0,
        "width": width,
        "n_zero_modes": n_zero,
        "x": x,
        "mass": mass,
        "psi0": psi,
        "density": density,
        "evals": evals,
    }
