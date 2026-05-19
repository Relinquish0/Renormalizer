import logging
import json

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.utils import Quantity, EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod, log
from renormalizer.utils.constant import cm2au
from renormalizer.transport import ChargeDiffusionDynamics, InitElectron

import numpy as np
  
# ── 参数设定 ──────────────────────────────────  
N      = 5    # 格点数（电子数量）  
omega_0 = 1.0  # 声子频率，单位 a.u.  
J      = 1.0   # 跳跃积分，单位 a.u.  
g      = 1   # 电子-振动耦合强度  
  
# ── 第一步：构建声子模式 ──────────────────────  
# 重组能 λ = g² * ω₀（lam=True 时第二个参数为 λ）  
lam = g**2 * omega_0   # = 2.25 a.u.  
  
displacement = g * np.sqrt(2.0 / omega_0)  # = 1.5 * sqrt(2) ≈ 2.121  
  
ph = Phonon.simple_phonon(  
    Quantity(omega_0),       # ω₀ = 1 a.u.    
    Quantity(displacement),  # 位移 d ≈ 2.121 a.u.    
    5                        # n_phys_dim = 4  
)
  
# ── 第二步：构建分子（每个格点一个电子 + 一个声子模式）──  
mol = Mol(Quantity(0), [ph])   # 局域激发能为 0  
  
# ── 第三步：组装 HolsteinModel ─────────────────  
# Quantity(J) 触发均匀最近邻跳跃矩阵的自动构造  
model = HolsteinModel(  
    [mol] * N,         # 75 个相同格点  
    Quantity(J),       # J = 1 a.u.，自动生成三对角跳跃矩阵  
    scheme=2,          # 基组排列方案（e₀, ph₀, e₁, ph₁, …）  
    periodic=False     # 开放边界条件（若需周期边界改为 True）  
)  
  
evolve_dt = 0.1
evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=evolve_dt)
compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=16)
ct = ChargeDiffusionDynamics(model, evolve_config=evolve_config, compress_config=compress_config, init_electron=InitElectron.fc)
ct.dump_dir = "./"
ct.job_name = 'Holstein5'
ct.stop_at_edge = False
ct.evolve(evolve_dt=evolve_dt, evolve_time=10)
