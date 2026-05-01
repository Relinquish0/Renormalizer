import logging
import json
import os

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.utils import Quantity, EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod, log
from renormalizer.utils.constant import cm2au
from renormalizer.transport import ChargeDiffusionDynamics, InitElectron
from renormalizer.multiset import MultisetChargeDiffusionDynamics

import numpy as np
import pandas as pd
from renormalizer.utils.log import package_logger as logger

# ── 参数设定 ──────────────────────────────────  
N      = 75    # 格点数（电子数量）  
omega_0 = 1.0  # 声子频率，单位 a.u.  
J      = 1.0   # 跳跃积分，单位 a.u.  
g      = 1.5   # 电子-振动耦合强度  
nu_max = 16    # 与 pyttn 中的 nu_max 保持一致
phonon_dim = nu_max + 1  # pyttn.boson_mode 使用局域 Hilbert-space dimension
  
# ── 第一步：构建声子模式 ──────────────────────  
# 重组能 λ = g² * ω₀；Renormalizer.simple_phonon 的第二个参数是位移 d
lam = g**2 * omega_0   # = 2.25 a.u.  
displacement = np.sqrt(2 * lam) / omega_0
  
ph = Phonon.simple_phonon(
    Quantity(omega_0),       # ω₀ = 1 a.u.  
    Quantity(displacement),  # d gives g = sqrt(λ / ω₀) = 1.5
    phonon_dim               # local dimension = nu_max + 1 = 17
)  
  
# ── 第二步：构建分子（每个格点一个电子 + 一个声子模式）──  
mol = Mol(Quantity(0), [ph])   # 局域激发能为 0  
  
# ── 第三步：组装 HolsteinModel ─────────────────  
# Quantity(J) 触发均匀最近邻跳跃矩阵的自动构造  
model = HolsteinModel(  
    [mol] * N,         # 75 个相同格点  
    Quantity(J),       # J = 1 a.u.，自动生成三对角跳跃矩阵  
    scheme=2,          # 基组排列方案（e₀, ph₀, e₁, ph₁, …）  
    periodic=True      # 与 pyttn 的 (site + 1) % nsites 周期边界保持一致
)  
  
max_bonddim = int(os.environ.get("MAX_BONDDIM", "8"))
evolve_dt = float(os.environ.get("EVOLVE_DT", "0.1"))
n_snapshots = int(os.environ.get("N_SNAPSHOTS", "500"))
if_startup_substeps = os.environ.get("IF_STARTUP_SUBSTEPS", "1").lower() not in {"0", "false", "no"}
startup_substeps_n = int(os.environ.get("STARTUP_SUBSTEPS_N", "10"))
output_xlsx = os.environ.get("OUTPUT_XLSX", f"Holstein75_{max_bonddim}bd_multiset.xlsx")
dynamics_job = MultisetChargeDiffusionDynamics(
    model=model,
    max_bonddim=max_bonddim,
    stop_at_edge=False,
    if_startup_substeps=if_startup_substeps,
    startup_substeps_n=startup_substeps_n,
)

from renormalizer.mps.backend import USE_GPU, xp  

logger.info(f"GPU enabled: {USE_GPU}")  
logger.info(f"Backend: {'CuPy' if USE_GPU else 'NumPy'}")
logger.info("maximum bond dimension:%d, evolve time step:%s", max_bonddim, evolve_dt)
logger.info("number of stored snapshots:%d", n_snapshots)
logger.info("startup substeps enabled:%s, count:%d", if_startup_substeps, startup_substeps_n)
logger.info("output xlsx: %s", output_xlsx)

logger.info("0th population: %s", dynamics_job.e_occupations_array[0])
dynamics_job.evolve(evolve_dt=evolve_dt, nsteps=n_snapshots - 1)
populations = np.array(dynamics_job.e_occupations_array)
pd.DataFrame(populations).to_excel(output_xlsx, index=False, header=False)
