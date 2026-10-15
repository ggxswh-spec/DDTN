# DDTN — 动态离散纯闭环网络

Dynamic Discrete Topological Network (DDTN)：一个基于纯闭环本体论的候选统一物理研究纲领。

本仓库包含：

- `ddtn_research_program.tex` — 研究纲领主文档（LaTeX）
- `ddtn_research_program.pdf` — 编译产物
- `code/` — 附录 G 所述最小可行模型（MVM）可复现数值模拟代码

## 构建 PDF

依赖 [Tectonic](https://tectonic-typesetting.github.io/)（XeTeX 内核，自动按需下载宏包）。

```bash
# 方式一：Makefile（需要 make + tectonic 在 PATH 中）
make pdf

# 方式二：直接调用 tectonic
tectonic ddtn_research_program.tex

# 方式三：Windows PowerShell
.\build.ps1
```

编译输出：`ddtn_research_program.pdf`。

## 运行 MVM 数值模拟

```bash
cd code
pip install -r requirements.txt
python run_mvm.py            # 运行三项基准测试并打印结果
python -m pytest tests/      # 单元测试
```

三项实验对应附录 G：

| 编号 | 模块 | 验证内容 |
|------|------|----------|
| G.1  | `mvm/photon_propagation.py` | 离散扭转相位波动方程蛙跳更新 → 极限波速 $c=1.000000$、色散关系 $\omega(k)=2\arcsin\sin(k/2)$ |
| G.2  | `mvm/bell_chsh.py`          | 全局离散步数下瞬发同调重构 → CHSH $S=2\sqrt{2}$、无信号边缘 |
| G.3  | `mvm/nielsen_ninomiya.py`   | 格点费米子倍增与拓扑畴壁零模 → 单一非简并手性零模 $E_0\approx 0$ |

## 持续集成

`.github/workflows/build.yml` 在每次推送时自动编译 PDF 并运行 MVM 测试，PDF 作为 artifact 上传。
