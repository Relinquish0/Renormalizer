import logging
import json

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.utils import Quantity, EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod, log
from renormalizer.utils.constant import cm2au
from renormalizer.transport import ChargeDiffusionDynamics, InitElectron

import numpy as np
  
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
  
evolve_dt = 0.1
evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=evolve_dt)
evolve_config.if_startup_substeps = True
evolve_config.startup_substeps_n = 10
compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=64)
ct = ChargeDiffusionDynamics(model, evolve_config=evolve_config, compress_config=compress_config, init_electron=InitElectron.fc)
ct.dump_dir = "./"
ct.job_name = 'Holstein75_64bd_subspace'
ct.stop_at_edge = False
ct.evolve(evolve_dt=evolve_dt, evolve_time=50)
