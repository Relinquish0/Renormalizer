import logging
import json

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.utils import Quantity, EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod, log
from renormalizer.utils.constant import cm2au
from renormalizer.transport import ChargeDiffusionDynamics, InitElectron
from renormalizer.utils.log import package_logger as logger
from renormalizer.model.multiset_model import MultisetModel

import numpy as np
import pandas as pd 
from datetime import datetime

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
  
max_bonddim = 16
evolve_dt = 0.1
multisetmodel = MultisetModel(model, max_bonddim=max_bonddim)

from renormalizer.mps.backend import USE_GPU, xp  

logger.info(f"GPU enabled: {USE_GPU}")  
logger.info(f"Backend: {'CuPy' if USE_GPU else 'NumPy'}")
logger.info("maximum bond dimension:%d, evolve time step:%d", max_bonddim, evolve_dt)

populations = []
for i in range(100):

    population = multisetmodel.popultation()
    logger.info("%dth population: %s", i, population)
    populations.append(population)
    multisetmodel.evolve(evolve_dt=evolve_dt)
pd.DataFrame(populations).to_excel(datetime.now().strftime("%Y-%m-%d-%H%M_Holstein_5") + str(max_bonddim) +'bd_' + str(evolve_dt) + "t.xlsx",
                            index=False, header=False)    