# multiset 总结

本文档分三部分：

1. `renormalizer/multiset/` 核心代码逐文件说明，按“函数/方法做什么、内部各段逻辑做什么、和谁关联”组织。
2. `multiset_202605/` 中各个物理模型、调用了 multiset 哪一层、完成了什么任务。
3. 在不改变算法的前提下，面向后续 `push` 的代码简化建议。

---

## 1. `renormalizer/multiset/` 逐文件说明

### 1.1 `__init__.py`

这个文件本身没有算法逻辑，只有导出逻辑。

- 导出 `MultisetMps`、`ElectronicAncillaMultisetMps`、`MsEvolveMethod`。
  作用：把“多分量态表示层”暴露给外部脚本。
  关联：真实实现都在 [`multiset_mps.py`](./multiset_mps.py)。
- 导出 `MultisetMpo`。
  作用：把“多分量算符表示层”暴露给外部脚本。
  关联：真实实现都在 [`multiset_mpo.py`](./multiset_mpo.py)。
- 导出 `MultisetModel`。
  作用：把“多分量哈密顿量拆分 + TDVP 演化引擎”暴露给外部脚本。
  关联：真实实现都在 [`multiset_model.py`](./multiset_model.py)。
- 导出 `MultisetTdJob`、`MultisetChargeDiffusionDynamics`、`MultisetSpectraZeroT`、`MultisetSpectraFiniteT`。
  作用：把“任务驱动层”暴露给外部。
  关联：真实实现分别在 [`multiset_tdjob.py`](./multiset_tdjob.py) 和 [`multiset_spectra.py`](./multiset_spectra.py)。

### 1.2 `multiset_mps.py`

这个文件定义“多电子分量态”本身。核心思想是：把一个单电子多站点问题写成若干个局域声子态 `mps_alpha` 的集合，每个电子位置对应一个分量。

#### 顶层函数

##### `_state_inner_product(bra, ket)`

- 第一段：调用 `bra.conj().dot(ket)` 计算底层张量网络重叠。
- 第二段：再乘上 `bra.coeff` 与 `ket.coeff`，补回 `Mps/MpDm` 外挂系数。
- 关联：
  与 `MultisetMps.ms_normalize`、`rho_el`、`e_occupations_multiset`、`ElectronicAncillaMultisetMps.total_norm/rho_el`、`MultisetModel.Hamiltonian/Inner_product`、`multiset_spectra.py` 中的重叠计算都直接相关。

##### `_state_expectation(bra, ket, mpo)`

- 第一段：调用 `ket.expectation(mpo=mpo, self_conj=bra.conj())` 算 `<bra|O|ket>` 的张量网络部分。
- 第二段：同样补上 `coeff`。
- 关联：
  被 `MultisetModel.Hamiltonian` 直接使用，是多分量能量和矩阵元的基础。

#### `MsEvolveMethod`

- 只有一个枚举值：`ms_evolve_tdvp_ps`。
  作用：给 `EvolveConfig` 一个 multiset 专用入口。
  关联：
  `MultisetModel.evolve_state` 用它分派到 `_ms_evolve_tdvp_ps`。

#### `class MultisetMps`

##### `__init__(...)`

- 第一段：检查 `init_mp` 和 `msmps` 只能二选一。
- 第二段：保存 `MsModel`、`N_electron`、`temperature`、`init_model`、`method` 等元数据。
- 第三段：把原始输入保存在 `_init_mp` / `_input_msmps`，然后调用 `_ConstructMsMps()` 真正建态。
- 关联：
  被 `MultisetModel.reset_mps`、`MultisetChargeDiffusionDynamics._init_msmps`、`MultisetSpectraZeroT._init_dipole`、`MultisetSpectraFiniteT._build_multiset_state` 调用。

##### `_ConstructMsMps(self)`

- 第一段：如果给了 `msmps`，检查长度是否等于 `N_electron`。
- 第二段：把每个输入分量深拷贝到 `self.msmps`，避免外部对象被原地修改。
- 第三段：如果给的是 `init_mp`，则复制 `N_electron` 份，构成每个电子分量的初态模板。
- 关联：
  是 `__init__` 的实际建态器；所有用“单个局域态广播到全部分量”的入口都会经过这里。

##### `copy(self)`

- 第一段：用 `__new__` 绕开 `__init__`。
- 第二段：复制元数据。
- 第三段：逐个深拷贝 `msmps`。
- 关联：
  被 `MultisetModel.evolve_state`、谱计算初始化等场景用于无副作用复制。

##### `to_complex(self)`

- 结构与 `copy` 类似，但把每个分量调用 `.to_complex()`。
- 关联：
  `MultisetModel._ms_evolve_tdvp_ps` 在实态进入复时间演化前会调用它。

##### `total_mps(self)`

- 调用 `_sum` 把所有分量相加成一个总 `Mps`。
- 关联：
  当前文件里主要提供辅助视角；更核心的物理量一般仍按分量逐项计算。

##### `ms_normalize(self, kind)`

- 第一段：逐个分量求 `_state_inner_product`，累加总范数。
- 第二段：开方得到总幅度。
- 第三段：仅支持 `"mps_only"`，即把每个分量整体缩放同一个因子。
- 关联：
  `MultisetModel.evolve_state`、`MultisetChargeDiffusionDynamics._fc_excitation/init_mps` 会调用。
  注意：当前实现只保留了一种归一化模式，和普通 `Mps.normalize` 的接口风格不同。

##### `rho_el(self)`

- 双循环 `(alpha, beta)` 计算所有电子分量之间的重叠矩阵。
- 关联：
  `MultisetChargeDiffusionDynamics.process_mps`、`MultisetModel.rho_el/decoherence_metrics` 使用它。

##### `e_occupations_multiset`

- 对角元版本的 `rho_el`。
- 对每个 `alpha` 只算 `mps_alpha` 自身范数，得到电子占据。
- 关联：
  `MultisetChargeDiffusionDynamics.process_mps/stop_evolve_criteria`、`MultisetModel.population` 使用。

##### `ph_occupations_multiset`

- 第一段：逐个分量取 `mps.ph_occupations`。
- 第二段：把所有分量的声子占据逐项相加。
- 第三段：如果虚部全为 0，返回实数组；否则保留复数。
- 关联：
  `MultisetChargeDiffusionDynamics.process_mps` 记录声子观测量。

##### `dump(self, fname)`

- 第一段：把目标文件名标准化成 `.npz`。
- 第二段：把每个分量单独 dump 到 `root_set{idx}.npz`。
- 第三段：总索引文件写入 `N_electron`、温度、方法、每个子态路径和类型。
- 关联：
  `MultisetTdJob._dump_state` 调用它做 checkpoint。

##### `load(cls, ...)`

- 第一段：读主 `.npz`。
- 第二段：如果发现 `electronic_ancilla=True`，转发给 `ElectronicAncillaMultisetMps.load`。
- 第三段：恢复元数据。
- 第四段：逐个读取子态文件，按 `MpDm`/`Mps` 分流加载。
- 关联：
  是 multiset checkpoint 的反序列化入口。

#### `class ElectronicAncillaMultisetMps(MultisetMps)`

这个类把“物理电子 index”与“电子 ancilla index”展平到一维 `msmps` 列表里，用来做电子自由度的 purification。

##### `__init__(...)`

- 第一段：仍然先检查 `init_mp` 和 `msmps` 的互斥性。
- 第二段：写入 `N_electron`、`n_anc`、`electronic_ancilla=True` 等附加元数据。
- 第三段：把总状态数设置成 `N_electron * n_anc`。
- 第四段：要么复制传入 `msmps`，要么复制 `init_mp` 生成全部分量。
- 关联：
  主要被 `MultisetSpectraFiniteT._build_electronic_ancilla_state` 构建。

##### `_check_index(self, alpha, ancilla=0)`

- 检查物理电子下标和 ancilla 下标是否合法。
- 返回展平后索引 `alpha * n_anc + ancilla`。
- 关联：
  `get_electron_ancilla_state`/`set_electron_ancilla_state`/`ancilla_set` 全依赖它。

##### `get_electron_ancilla_state(self, alpha, ancilla=0)` / `set_electron_ancilla_state(self, alpha, ancilla, mps)`

- 只是对展平存储的安全访问器。
- 关联：
  `MultisetModel.evolve_state`、`multiset_spectra.py` 中 ancilla 分支大量使用。

##### `ancilla_set(self, ancilla)`

- 收集固定 ancilla 下的全部物理电子分量。
- 关联：
  `ancilla_set_state` 用它抽出一个普通 `MultisetMps` 视图。

##### `ancilla_set_state(self, ancilla)`

- 用 `ancilla_set(ancilla)` 返回的分量重新构造一个普通 `MultisetMps`。
- 关联：
  `MultisetModel.evolve_state` 对 ancilla 态是“逐个 ancilla 块调用普通 multiset TDVP”的，这里是关键桥梁。

##### `copy(self)` / `to_complex(self)`

- 与父类同理，但会保留 `n_anc`。
- 关联：
  被 `MultisetModel.evolve_state`、谱初始化调用。

##### `total_norm(self)`

- 把所有 `(alpha, ancilla)` 分量的范数加起来。
- 关联：
  `ms_normalize`、`MultisetModel._compute_multiset_norm/Hamiltonian/Inner_product` 使用。

##### `ms_normalize(self, kind)`

- 与普通 `MultisetMps` 一样，只支持 `"mps_only"`。
- 只是求范数时跨越全部 ancilla 分量。

##### `rho_el(self)`

- 外层只保留物理电子指标 `(alpha, beta)`。
- 内层对 ancilla 求和，得到物理电子约化密度矩阵。
- 关联：
  `MultisetSpectraFiniteT` 的 ancilla 谱相关量最终都通过这里回到物理电子空间。

##### `e_occupations_multiset`

- 固定 `alpha`，把所有 ancilla 的对角重叠加总。

##### `ph_occupations_multiset`

- 和父类相同，只是累加的是全部 ancilla 分量。

##### `dump(self, fname)`

- 和父类类似，但额外写入 `n_anc` 和 `electronic_ancilla=True`。

##### `load(cls, ...)`

- 读回每个子态后，重新构造 `ElectronicAncillaMultisetMps`。
- 关联：
  `MultisetMps.load` 会把 ancilla 场景转发到这里。

### 1.3 `multiset_mpo.py`

这个文件定义“多分量算符”。`MultisetMpo` 是从 `MsModel[alpha][beta]` 出发的块矩阵 MPO。

#### `class MultisetMpo`

##### `__init__(self, msmodel, N_electron)`

- 保存 `MsModel` 和电子维数。
- 初始化 `msmpo[alpha][beta]` 的二维表。
- 调用 `_ConstructMsMpo` 逐块生成真实 MPO。
- 关联：
  `MultisetModel.__init__` 会构造主哈密顿量的 `MultisetMpo`。

##### `_ConstructMsMpo(self)`

- 双循环扫描每个块模型 `MsModel[i][j]`。
- 如果该块没有 `ham_terms`，保存空列表。
- 否则用 `Mpo(model=..., terms=None)` 生成整块哈密顿量 MPO。
- 关联：
  是 `MultisetModel.SplitHamTerm/ConstructMsModel` 的后继步骤。

##### `total_mpo(self)`

- 第一段：每一行 `alpha` 把所有列块求和。
- 第二段：再把所有行结果求和。
- 关联：
  提供“把块 MPO 合并成一个总 MPO”的辅助接口。


### 1.4 `multiset_model.py`

这个文件是 multiset 的核心。它把普通 `Model` 拆成块模型，再用块结构 TDVP 实现时间演化。

#### `class MultisetModel`

##### `__init__(...)`

- 第一段：保存原始 `model`、温度、初始化方法。
- 第二段：准备 `evolve_config`，默认 method 是 `MsEvolveMethod.ms_evolve_tdvp_ps`。
- 第三段：准备 `compress_config`，默认是固定 bond dimension。
- 第四段：从原始模型提取 `N_electron`、电子自由度索引映射、纯声子基底 `basis_set`。
- 第五段：初始化 `MsModel` 与 `MsOp` 的二维块表。
- 第六段：依次调用 `SplitHamTerm`、`ConstructInitModel`、`ConstructMsModel` 完成模型拆分。
- 第七段：构造 `MsMpo`，建立一组缓存容器：
  - `_active_pairs_index` / `_active_pair_mpos`：非零块列表。
  - `_active_pairs_index_by_alpha`：按目标分量分桶。
  - `_site_group_templates`：按站点和局域张量形状预分组，给 batched matvec 用。
  - `_qr_qn_plan_cache`：量子数 QR 的计划缓存。
  - `_environ_cache`：环境张量缓存。
- 第八段：调用 `_active_mpo_select_grouping()` 填充这些缓存。
- 关联：
  是 `MultisetChargeDiffusionDynamics` 和两个谱类的底层演化引擎。

##### `SplitHamTerm(self)`

- 遍历原始 `model.ham_terms`。
- 先通过 `_get_electron_transition` 判断该项是否含电子跃迁。
- 若没有电子跃迁：说明是纯声子项或纯对角电子项，把去电子后的算符复制到每个对角块 `MsOp[alpha][alpha]`。
- 若有跃迁 `(alpha,beta)`：只放进对应块。
- 关联：
  依赖 `_get_electron_transition` 和 `_reset_all_MsOp`，输出给 `ConstructMsModel`。

##### `_get_electron_transition(self, op)`

- 第一段：扫描 `op.dofs` 和 `split_symbol`，抽出所有电子自由度上的局域算符。
- 第二段：无电子算符则返回 `None`。
- 第三段：要求恰好两个电子算符，否则报错。
- 第四段：要求符号集合是 `{a^\dagger, a}`，否则报错。
- 第五段：根据 `a^\dagger` 和 `a` 的位置把它映射成 `(alpha, beta)`。
- 关联：
  被 `SplitHamTerm` 调用，是“哪个块接收这个哈密顿量项”的判定器。

##### `_reset_all_MsOp(self, op)`

- 第一段：复制输入 `Op`。
- 第二段：去掉所有电子自由度，只保留声子部分的 `symbol/qn/dofs`。
- 第三段：如果去完后为空，说明原项纯电子；此时用第一个声子自由度上的单位算符承载它的系数。
- 第四段：否则回写瘦身后的 `Op` 内容。
- 关联：
  被 `SplitHamTerm` 和 `ConstructInitModel` 调用，是“把电子-声子哈密顿量块化后，局域块里实际存什么算符”的关键。

##### `ConstructMsModel(self)`

- 双循环把 `MsOp[alpha][beta]` 封装成 `Model(basis=self.basis_set, ham_terms=...)`。
- 关联：
  直接为 `MultisetMpo` 提供输入。

##### `ConstructInitModel(self)`

- 只收集原始哈密顿量中 `len(op.dofs) == 1` 的单体项。
- 对这些项调用 `_reset_all_MsOp` 去电子化。
- 生成 `init_model`。
- 关联：
  `MultisetChargeDiffusionDynamics.init_mp`、`MultisetSpectraZeroT.init_mp`、有限温谱的声子热初态都依赖它。

##### `_active_mpo_select_grouping(self)`

- 第一段：失效环境缓存。
- 第二段：遍历所有非空 `msmpo[alpha][beta]`，记录到 `_active_pairs_index`、`_active_pair_mpos`、`_active_pairs_by_alpha`。
- 第三段：如果某个站点的不同块 MPO 局域张量形状相同，则把它们聚成一个 active_mpos_group。
- 第四段：为每个 active_mpos_group 预先堆叠：
  - `W`：局域 MPO 张量批。
  - `alpha_idx/beta_idx`：块索引。
  - `S`：把 pair 结果散射回目标 `alpha` 的 one-hot 矩阵。
- 关联：
  `_build_site_batched_data`、`_apply_hop_batched` 全依赖这里的模板。

##### `_build_pair_to_alpha_matrix(self, alpha_idx, dtype)`

- 构造形状 `(N_electron, n_pairs)` 的 one-hot 散射矩阵。
- 作用：把按 pair 计算的局域结果累加回按 `alpha` 编排的输出。
- 关联：
  `_active_mpo_select_grouping` 和 `_build_reverse_batched_data` 使用。

##### `_invalidate_environ_cache(self)`

- 直接清空 `_environ_cache`。

##### `_has_valid_environ_cache(self, ms_mps)`

- 第一段：先检查是否允许复用，以及缓存对象本身是否存在。
- 第二段：从参考分量提取当前缓存 cache key：
  - state token
  - 活动块个数
  - 站点数
  - 正交中心方向 `to_right`
  - 正交中心位置 `qnidx`
- 第三段：把当前签名和缓存里记录的 cache key比较。
- 第四段：再检查环境条目个数是否和当前活动块数一致。

##### `_build_environ_list(self, ms_mps, conj_mps)`

- 对每个活动块 `(alpha,beta)` 构造一个 `Environ`：
  - ket 用 `ms_mps[beta]`
  - bra 用 `conj_mps[alpha]`
  - MPO 用对应块 MPO
- 关联：
  TDVP 局域有效哈密顿量的左右环境都来自这里。

##### `_get_or_build_environ_list(self, ms_mps, conj_mps)`

- 如果缓存有效直接返回。
- 否则重建并写回缓存。

##### `_store_environ_cache(self, ms_mps, environ_list)`

- 给 state 打一个 `_environ_cache_token`。
- 用新签名和新环境对象回填缓存。
- 关联：
  `_ms_evolve_tdvp_ps` 结束时调用。

##### `_compute_multiset_norm(self, ms_mps)`

- ancilla 态：用 `total_norm`。
- 普通态：逐分量求 `_state_inner_product`。
- 返回的是总范数平方根。
- 关联：
  `evolve_state(..., normalize=True)` 使用。

##### `_rescale_environ_cache_after_normalize(self, ms_mps, scale_factor)`

- 第一段：只有在缓存仍然有效时才工作。
- 第二段：根据正交规范方向判断哪些环境块只会被整体缩放。
- 第三段：把缓存内对应方向的环境张量乘上 `|scale_factor|^2`。
- 第四段：如果规范位置不满足简单缩放前提，就直接失效缓存。
- 关联：
  让归一化后无需全部重建环境缓存。

##### `_get_qr_qn_plan(self, qnbigl, qnbigr, qntot)`

- 第一段：用左右大量子数和总量子数生成缓存键。
- 第二段：把左、右大 qn 拉平，找出每个允许的 `(left qn, right qn)` 匹配块。
- 第三段：把每个块保存成 `(lset, rset, nl, nr)`。
- 第四段：若没有任何匹配块则报错。
- 关联：
  `_batched_qr_qn` 使用它做量子数分块 QR。

##### `_batched_qr_qn(...)`

- 第一段：把批量局域系数重排成 `(batch, left_dim, right_dim)`。
- 第二段：对每个 qn 块分别做 QR。
  - `system == "L"`：直接 QR。
  - `system == "R"`：通过转置变成右规范版本。
- 第三段：把每个 qn 子块重新散回完整稀疏位置。
- 第四段：把所有块沿 rank 方向拼起来。
- 第五段：如果给了 `max_rank`，只截取前若干列。
- 关联：
  `_ms_evolve_tdvp_ps` 在每个站点上做 one-site TDVP 分裂时调用。

##### `_build_site_batched_data(self, imps, l_tensors, r_tensors)`

- 按当前站点 `imps` 的模板，填入真实左右环境 `L/R`。
- 输出给 `_apply_hop_batched` 的输入批。
- 关联：
  `_ms_evolve_tdvp_ps` 的局域传播使用。

##### `_build_reverse_batched_data(self, l_tensors, r_tensors)`

- 这是中心张量回传那一步的 batched 数据构造器。
- 与 `_build_site_batched_data` 不同：
  - 这里没有显式 `W`，因为传播对象是 QR 后的中心张量。
  - 主要按 `L` 张量中间维度形状分组。
- 关联：
  `_ms_evolve_tdvp_ps` 的 `Ut/Ct` 半步反向传播使用。

##### `reset_mps(self, init_mp=None, msmps=None)`

- 构造新的 `MultisetMps`。
- 立即做一次 `ms_normalize("mps_only")`。
- 清空环境缓存。

##### `set_mps(self, ms_mps)`

- 只是把外部态挂到 `self.MsMps`。

##### `evolve_state(self, ms_mps, evolve_dt, normalize=True)`

- 第一段：根据 `evolve_config.method` 选演化器，目前只有 `_ms_evolve_tdvp_ps`。
- 第二段：如果是 ancilla 态，就按 ancilla 块逐块抽出 `ancilla_set_state` 演化，再写回。
- 第三段：否则直接演化普通 multiset 态。
- 第四段：如果要求归一化，计算总范数、调用 `ms_normalize`，再尝试缩放环境缓存。
- 关联：
  是所有 job 的单步推进入口。

##### `evolve(self, evolve_dt, normalize=True)`

- 只是对 `self.MsMps` 原地应用 `evolve_state`。

##### `_ms_evolve_tdvp_ps(self, ms_mps_, ms_mpo, evolve_dt)`

- 准备段：
  - 如果 `evolve_dt` 是复数，说明是虚时间或复时间传播，保留/复制实态。
  - 如果是实时间，先把态转成复态。
  - 尽量继承 `_environ_cache_token`，让缓存可延续。
- 环境段：
  - 为每个分量构造共轭态。
  - 用 `_get_or_build_environ_list` 获取每个活动块的左右环境。
- 扫描段：
  - 做两次 sweep。
  - 每次遍历当前 canonical 方向下的所有站点 `imps`。
  - 读出该站点的左右环境 `l_array_ab/r_array_ab`。
  - 用 `_build_site_batched_data` 形成批量局域哈密顿量数据。
  - 把所有 `alpha` 分量当前局域张量拼成一个大向量 `Y0`。
  - 通过 `ivp_eq = lambda Y: self._apply_hop_batched(...)` 定义局域有效哈密顿量作用。
  - 当前实现只在 `ivp_solver == "krylov"` 分支里调用 `expm_krylov`。
- QR 分裂段：
  - 把传播后的局域张量 `mps_t` 做量子数守恒 QR。
  - 若是向左 sweep，中间量命名为 `U`；若是向右 sweep，中间量命名为 `C/Vt`。
  - `max_qr_rank` 来自 `compress_config` 当前 bond 的上限。
- 中心张量反传播段：
  - 对非边界站点，还需要把 QR 产生的中心张量再做半步传播。
  - 利用 `Environ.GetLR(...)` 更新一侧环境。
  - 用 `_build_reverse_batched_data` 和 `_apply_hop_batched` 对 `U0/C0` 做一次传播。
  - 再把传播后的中心张量吸收到相邻站点。
- 写回段：
  - 边界站点则直接把 `mps_t` 写回局域张量。
  - 每个 full sweep 结束后，对所有分量 `_switch_direction()`。
- 收尾段：
  - 统计 Krylov 子空间维数。
  - 记录到 `evolve_config.stat`。
  - 用 `_store_environ_cache` 缓存最新环境。
- 关联：
  整个 multiset TDVP 的主算法都在这里；它依赖 `_build_site_batched_data`、`_batched_qr_qn`、`_build_reverse_batched_data`、`_apply_hop_batched`。

##### `expand_bond_dimension_multiset(self, coef=1e-10, use_hint=True)`

- ancilla 分支：
  - 按 ancilla 块递归调用普通 multiset 扩维，再拼回整体。
- 非 ancilla 分支：
  - 先把每个分量的 `compress_config` 设置好。
  - 若 `use_hint=False`，直接对每个分量调用 `expand_bond_dimension_general`。
  - 若 `use_hint=True`：
    - 对角块 MPO 作为 `hint_mpo`。
    - 其余 `beta -> alpha` 的耦合块作用到 `original_mps[beta]` 上并求和，作为 `ex_mps`。
    - 让 `expand_bond_dimension_general` 同时参考对角哈密顿量和跨块驱动。
  - 扩维后把 `coeff` 吸收到张量本体，保持 `coeff=1`。
- 关联：
  `MultisetChargeDiffusionDynamics.init_mps` 和两个谱类初始化都依赖它来准备更合理的初始 bond 结构。

##### `population(self)` / `popultation(self)`

- 返回 `MsMps.e_occupations_multiset`。
- `popultation` 只是旧拼写兼容别名。

##### `Hamiltonian(self)`

- 对 ancilla 态：
  - 分 ancilla 和活动块求和 `<alpha,anc|H_{alpha,beta}|beta,anc>`。
  - 分母用 `total_norm()`。
- 对普通态：
  - 对活动块求和 `<alpha|H_{alpha,beta}|beta>`。
  - 分母是所有分量范数和。
- 返回实部。
- 关联：
  `MultisetChargeDiffusionDynamics.init_mps/process_mps`、`MultisetSpectraFiniteT.init_mps_emi` 使用。

##### `rho_el(self)` / `decoherence_metrics(self)` / `Inner_product(self)`

- `rho_el`：转发到态对象。
- `decoherence_metrics`：从 `rho` 计算迹和 purity。
- `Inner_product`：返回总范数。

##### `_apply_hop_batched(self, Y, batched_groups, dim, shape)`

- 第一段：把输入大向量 `Y` 还原成 `(N_electron, dim)`。
- 第二段：初始化输出 `Y_out`。
- 第三段：对每个 batched group：
  - 取出 `L/R/S/beta_idx/W/nsite`。
  - 先选出所有源分量 `beta` 对应的输入 `Y_exp`。
  - `nsite == 1` 时，说明是站点局域传播，要用 `W` 参与收缩。
    - shape 长度为 3 时用一套 `einsum`。
    - shape 长度为 4 时用另一套 `einsum`。
  - `nsite == 0` 时，说明传播的是中心张量，不含 `W`。
  - 把每个 pair 的输出 flatten 后，再通过 `S` 散回每个目标 `alpha`。
- 第四段：累计 matvec 次数。
- 关联：
  是 `_ms_evolve_tdvp_ps` 里 Krylov 指数传播器的实际“矩阵乘向量”核心。

### 1.5 `multiset_tdjob.py`

这个文件是“任务调度层”，不改变底层算法，只负责：准备初态、循环推进、记录观测量、dump。

#### 顶层函数

##### `_thermal_coefficients_from_theta(basis, temperature)`

- 用 `theta = arctanh(exp(-beta*omega/2))` 构造 thermo-field 声子热权重。
- 最后做归一化。
- 关联：
  `MultisetChargeDiffusionDynamics.init_mp(method="thermo_field")` 使用。

##### `_calc_r_square_multiset(e_occupations)`

- 用电子占据作为概率权重计算 `⟨r^2⟩ - ⟨r⟩^2`。
- 若全部占据为 0，返回 0。
- 关联：
  `MultisetChargeDiffusionDynamics.process_mps` 记录电荷扩散均方位移时使用。

##### `_state_bond_dims(state)`

- 如果是 `MultisetMps`，返回每个分量的 bond dims。
- 如果对象自身有 `bond_dims`，直接返回。
- 如果是元组/列表，递归处理。
- 关联：
  `MultisetTdJob.__init__/evolve` 和两个谱类 `process_mps` 用来记录结构复杂度。

#### `class MultisetTdJob`

##### `__init__(...)`

- 第一段：保存 `evolve_config`、dump 选项、输出目录、作业名、startup substeps 参数。
- 第二段：调用子类实现的 `init_mps()`。
- 第三段：记录初始 bond dims。
- 第四段：把初态存入 `latest_mps`，并调用 `process_mps(mps)` 记录 `t=0` 观测量。
- 关联：
  所有 multiset 任务类都继承它。

##### `init_mps/process_mps/evolve_single_step/get_dump_dict`

- 这四个方法都是模板方法，占位给子类实现。

##### `stop_evolve_criteria(self)`

- 基类默认永不停止。
- `MultisetChargeDiffusionDynamics` 会重写。

##### `_run_startup_substeps(self, evolve_dt)`

- 把第一步 `evolve_dt` 对数切分成若干个越来越大的子步长。
- 每个子步都调用一次 `evolve_single_step`。
- 作用：让一开始更稳定地进入 TDVP 轨道。
- 关联：
  `evolve()` 在第一次真正步进前可选调用。

##### `_checkpoint_state_path(self)`

- 根据 `dump_mps` 模式决定 checkpoint 命名：
  - `"all"`：每一步一个文件。
  - 其它：固定覆盖一个文件。

##### `_dump_state(self, state, fname)`

- 如果对象有 `dump()` 方法，直接调它。
- 如果是 tuple/list，递归保存每个分量，再写一个总索引 `.npz`。
- 关联：
  支持同时保存 `(bra, ket)` 这样的谱计算状态。

##### `dump_dict(self)`

- 先让子类提供 `get_dump_dict()`。
- 再负责：
  - 创建目录
  - `.bak` 备份旧输出
  - `np.savez`
  - 可选 checkpoint 保存
- 关联：
  `evolve()` 每个记录步后调用。

##### `evolve(self, evolve_dt=None, nsteps=None, evolve_time=None)`

- 参数判定段：
  - 支持给 `(dt, nsteps)`、`(dt, evolve_time)`、`(nsteps, evolve_time)` 三种组合。
  - 也支持只给 `dt`，然后靠 `stop_evolve_criteria` 终止。
- startup 段：
  - 如果启用 startup substeps 且仍在起点，先跑 `_run_startup_substeps`。
  - 然后手动做一次和主循环相同的记录/dump。
- 主循环段：
  - 每步先检查停止条件。
  - 调 `evolve_single_step`。
  - 更新时间、记录观测量、更新 `latest_mps`。
  - 根据 `info_interval` 决定是否打印 bond dims，并决定是否 dump mps。
  - 如果定义了输出路径，就调用 `dump_dict()`。
- 收尾段：
  - 打印总耗时并返回 `self`。
- 关联：
  所有任务类共享同一套时推进框架。

##### `latest_evolve_time` / `evolve_times_array` / `_defined_output_path`

- 都是简单属性封装。

#### `class MultisetChargeDiffusionDynamics(MultisetTdJob)`

这个类把 `MultisetModel` 封装成“电荷/激子扩散任务”。

##### `__init__(...)`

- 第一段：如果用户没给 `ms_model`，就从普通 `model` 构造 `MultisetModel`。
- 第二段：确定初始电子位置 `initial_site`、边界停止规则、扩维策略、观测量开关。
- 第三段：初始化各类时间序列缓存数组。
- 第四段：调用父类 `MultisetTdJob.__init__`，从而自动执行初态准备和 `t=0` 记录。
- 关联：
  `multiset_202605` 里所有扩散算例都直接用它。

##### `init_mp(self, method=None)`

- 零温分支：直接返回 `Mps.hartree_product_state(init_model)`。
- `imaginary_time_exact` 分支：
  - 对每个 SHO 模式按玻尔兹曼因子直接写系数。
  - 构造热 `Mps` 后转成 `MpDm`。
- `thermo_field` 分支：
  - 调 `_thermal_coefficients_from_theta` 得到 thermo-field 权重。
  - 同样转为 `MpDm`。
- `imaginary_time_propagate` 分支：
  - 从 `MpDm.max_entangled_gs` 出发。
  - 用 `ThermalProp` 做虚时间传播。
  - 返回传播后的热态。
- 关联：
  这是扩散任务的局域初态生成器；后续 `_init_msmps` 和 `init_mps` 会再把它包装成 multiset。

##### `_init_msmps(self, local_state)`

- 如果传进来的已经是 `MultisetMps`，直接返回。
- 否则把单个局域态广播成 `N_electron` 个分量。
- 关联：
  `init_mps` 使用。

##### `_fc_excitation(self, state, alpha)`

- 把目标分量 `alpha` 以外的所有电子分量缩到 `1e-10`。
- 最后重新归一化。
- 作用：实现 Franck-Condon 式“电子刚被放到某个站点”的初态。
- 关联：
  `init_mps` 调用。

##### `_set_hamiltonian_offset(self, energy)`

- 双循环重建 `MsMpo.msmpo[alpha][beta]`。
- 对角块加上 `offset=energy`，非对角块 offset 置 0。
- 最后 `_active_mpo_select_grouping()`。
- 作用：把初始载流子能量整体减掉，减轻相位振荡。
- 关联：
  `init_mps` 使用；有限温谱里有一份近似同构版本 `_set_multiset_hamiltonian_offset`。

##### `init_mps(self)`

- 第一段：先由 `init_mp()` 得到局域态，再 `_init_msmps()` 广播成 multiset。
- 第二段：调用 `_fc_excitation` 把初始电子放到指定站点。
- 第三段：把态挂到 `ms_model`，计算初始总能量。
- 第四段：调用 `_set_hamiltonian_offset` 重新定义哈密顿量零点。
- 第五段：调用 `expand_bond_dimension_multiset` 给各分量准备更合理的初始 bond 维数。
- 第六段：再次归一化后返回。
- 关联：
  是所有 multiset 扩散任务的标准初始化路径。

##### `process_mps(self, mps)`

- 第一段：把新态挂回 `ms_model`。
- 第二段：始终记录 `e_occupations_array`。
- 第三段：根据 `observables` 开关按需记录：
  - `energy`
  - `r_square`
  - `ph_occupations`
  - `rho`
  - `coherent_length`
  - `trace`
  - `purity`
- 第四段：打印当前占据。
- 关联：
  `MultisetTdJob.evolve` 每一步都会调用它。

##### `evolve_single_step(self, evolve_dt)`

- 直接调 `ms_model.evolve_state(self.latest_mps, evolve_dt)`。
- 关联：
  把时间推进真正委托给 `MultisetModel`。

##### `stop_evolve_criteria(self)`

- 如果启用 `stop_at_edge`，当第 0 个站点占据超过阈值时停止。
- 作用：适合一维链上“传播回边界”即结束的任务。

##### `get_dump_dict(self)`

- 把当前模型、温度、时间序列和已启用的观测量打包成字典。
- 关联：
  由基类 `dump_dict()` 负责落盘。

### 1.6 `multiset_spectra.py`

这个文件建立在 `MultisetTdJob` 之上，专门做零温/有限温光谱关联函数。

#### 顶层函数

##### `_multiset_pair_overlaps(bra, ket)`

- 普通态：直接计算所有 `(alpha,beta)` 分量重叠。
- ancilla 态：在每个 `(alpha,beta)` 上再对 ancilla 求和。
- 关联：
  是后面所有 multiset 关联函数的基础。

##### `_multiset_overlap(bra, ket, return_pair_overlaps=False)`

- 调 `_multiset_pair_overlaps`。
- 若 `return_pair_overlaps=False`，返回迹 `sum_alpha <alpha|alpha>`。
- 否则返回整块矩阵。
- 关联：
  零温谱的 `process_mps` 和 `_expand_ket_bonddim` 使用。

##### `_scale_multiset_state(state, factor)`

- 把所有分量统一缩放。
- 关联：
  `MultisetSpectraZeroT._expand_ket_bonddim` 用它在扩维后恢复原始范数。

#### `class MultisetSpectraZeroT(MultisetTdJob)`

##### `__init__(...)`

- 第一段：检查 `spectratype in {"abs","emi"}`。
- 第二段：保存类型、温度 0 K、offset、是否扩维、以及各类输出数组。
- 第三段：构造 `MultisetModel`。
- 第四段：构造单个局域基底上的 `h_mpo`，给非 multiset 局域态传播用。
- 第五段：调用父类，触发初始化与 `t=0` 记录。

##### `init_mps(self)`

- `emi` 走 `init_mps_emi()`，否则走 `init_mps_abs()`。

##### `init_mps_abs(self)` / `init_mps_emi(self)`

- 当前两者逻辑一致：
  - 由 `_init_dipole()` 构造偶极激发后的 `ket`。
  - 若 `expand=True`，先扩维。
  - 返回 `(ket.copy(), ket)`，即 `bra` 固定，`ket` 演化。
- 关联：
  `process_mps` 的自相关函数正是对这对 `(bra, ket)` 取重叠。

##### `evolve_single_step(self, evolve_dt)`

- 拆包 `bra, ket`。
- 如果 `ket` 是 `MultisetMps`，则用 `ms_model.evolve_state` 演化。
- 否则按单个 `Mps/MpDm` 调局域 `h_mpo` 演化。
- 关联：
  支持把 zero-T 光谱统一写成“固定 bra，演化 ket”。

##### `process_mps(self, mps)`

- 第一段：若 `bra` 是 multiset，就计算分量矩阵 `component_autocorr`。
- 第二段：
  - 发射 `emi`：取全部元素求和。
  - 吸收 `abs`：取迹。
- 第三段：若不是 multiset，则退化成单个复数自相关。
- 第四段：记录 bond dims。
- 关联：
  输出数组 `autocorr` 与 `autocorr_components` 都在这里维护。

##### `autocorr` / `autocorr_components` / `bond_dims`

- 都只是把内部列表转成数组。

##### `get_dump_dict(self)`

- 返回温度、时间序列、自相关和 bond dims。

##### `init_mp(self)`

- 返回零温基态的 `hartree_product_state`。
- 关联：
  `_init_dipole` 直接以它为局域参考态。

##### `_get_dipole(self)`

- 第一段：从 `model.dipole` 取偶极强度。
- 第二段：若是 dict，则按 `model.e_dofs` 顺序展开。
- 第三段：若是标量则广播到每个电子激发。
- 第四段：做长度一致性检查。
- 关联：
  `_init_dipole` 使用；有限温谱类有一份几乎相同的实现。

##### `_init_dipole(self)`

- 第一段：读偶极权重。
- 第二段：构造局域基态 `phi_g`。
- 第三段：对每个电子分量复制一份 `phi_g`，并乘上对应偶极系数。
- 第四段：把这些分量包装成 `MultisetMps`。
- 关联：
  这是 zero-T absorption/emission 初始态的生成器。

##### `_expand_ket_bonddim(self, ket, coef=1e-10, use_hint=True)`

- 第一段：先算扩维前总范数。
- 第二段：把 `ket` 挂到 `ms_model`，调用 `expand_bond_dimension_multiset`。
- 第三段：算扩维后总范数。
- 第四段：如果范数改变，则统一缩放恢复原值。
- 关联：
  保证“扩维只改变表达能力，不改变初态物理归一化”。

#### `class MultisetSpectraFiniteT(MultisetTdJob)`

这个类比 zero-T 复杂得多，因为它要同时处理：

- 声子有限温初态；
- absorption 和 emission 不同的初态构造；
- 可选电子 ancilla purification；
- ground branch 和 excited branch 的交替传播。

##### `__init__(...)`

- 第一段：检查 `spectratype`、`temperature != 0`、`thermal_init_method` 等输入。
- 第二段：保存所有有限温与电子 ancilla 相关参数。
- 第三段：创建 `MultisetModel`，其 `method` 保存为 `thermal_init_method`。
- 第四段：准备几个单集传播对象：
  - `h_mpo`：局域传播器，带用户 offset。
  - `h_mpo_gs`：ground-state 局域传播器，不带 offset。
  - `icompress_config`：局域/热态初始化用压缩参数。
  - `local_evolve_config`：对单个 `MpDm` 做普通 TDVP。
  - `ievolve_config`：对 multiset 做虚时间初始化时的演化配置。
- 第五段：根据 `spectratype` 选择 job 层使用的 `evolve_config`。
  - absorption：直接用 multiset 演化配置。
  - emission：ground branch 局域传播更多，因此 job 层先绑 `local_evolve_config`。
- 第六段：调用父类进入标准 job 生命周期。

##### `init_mps(self)`

- 根据 `spectratype` 分派到 `init_mps_abs` 或 `init_mps_emi`。

##### `init_mps_abs(self)`

- 第一段：调用 `_init_ground_thermal_state()` 得到有限温基态 `MpDm`。
- 第二段：调用 `_init_dipole(thermal_mpdm)` 把它广播成 multiset 光激发态。
- 第三段：按需扩维。
- 第四段：把 multiset 哈密顿量 offset 设成用户给的 `offset`。
- 第五段：返回 `(ket.copy(), ket)`。
- 物理意义：
  absorption 中热平均只在 ground phonon branch 上。

##### `init_mps_emi(self)`

- 第一段：调用 `_init_excited_thermal_state()` 生成激发态有限温分布。
- 第二段：施加 dipole。
- 第三段：按需扩维。
- 第四段：把该 excited multiset 态挂到 `ms_model`，测其能量。
- 第五段：把这个 excited carrier energy 加到 offset 中减掉。
- 第六段：返回 `(ket.copy(), ket)`。
- 物理意义：
  emission 要从激发态热分布出发，所以 offset 的参照也不同。

##### `_electronic_ancilla_dim(self)`

- 返回电子 ancilla 维数，默认等于物理电子维数。

##### `_electronic_beta_value(self)`

- 优先级：
  - 显式 `electronic_beta`
  - `electronic_temperature`
  - 否则退回到格点温度 `self.temperature`

##### `_electronic_hamiltonian_matrix(self)`

- 如果用户显式给了 `electronic_hamiltonian`，直接转成矩阵。
- 否则若来源是 `"msmodel"`：
  - 用 `model.j_matrix` 做非对角。
  - 用每个 `mol.elocalex + mol.e0` 填对角。
- 最后检查尺寸是否匹配。
- 关联：
  仅在电子 ancilla thermal purification 时使用。

##### `_build_ancilla_purification_coefficients(self)`

- `diagonal` 模式：
  - 根据用户给的 `electronic_initial_distribution` 构造一个对角概率分布的 purification 系数矩阵。
- `thermal` 模式：
  - 对电子哈密顿量对角化。
  - 用 `exp(-beta E)` 生成热权重。
  - 构造 `evecs @ diag(sqrt(weights))`。
  - 如果 ancilla 维数不足以容纳非零本征态，则报错。
- 关联：
  `_build_electronic_ancilla_state` 调用。

##### `_build_electronic_ancilla_state(self, local_state, weights=None)`

- 第一段：得到电子 purification 系数矩阵 `coeff[alpha, ancilla]`。
- 第二段：给每个 `(alpha, ancilla)` 复制一份 `local_state`。
- 第三段：把它的 `coeff` 乘上 `weights[alpha] * coeff[alpha, ancilla]`。
- 第四段：包装成 `ElectronicAncillaMultisetMps`。
- 关联：
  `_broadcast_local_state`、`_init_excited_thermal_state` 使用。

##### `_init_ground_thermal_state(self)`

- 只是按 `thermal_init_method` 分派：
  - exact
  - imaginary-time propagate

##### `_exact_ground_thermal_mpdm(self, model)`

- 对每个 SHO 模式直接写玻尔兹曼权重。
- 构造 `thermal_mps`，再转 `MpDm`。
- 设置压缩与演化配置。
- 关联：
  finite-T absorption 最常规的热初态生成器。

##### `_propagate_ground_thermal_mpdm(self, model)`

- 从最大纠缠 ground `MpDm` 出发。
- 用 `ThermalProp` 做虚时间传播到 `beta/2`。
- 关联：
  用传播方式生成声子热初态，适合不直接写出解析权重的场景。

##### `_propagate_excited_thermal_msmpdm(self, state)`

- 第一段：要求 `insteps` 已定义。
- 第二段：把总虚时间 `beta/2` 均匀拆成 `insteps`。
- 第三段：暂时把 `ms_model.evolve_config` 切到 `ievolve_config`。
- 第四段：循环调用 `ms_model.evolve_state(..., normalize=True)`。
- 第五段：`finally` 里恢复原配置。
- 关联：
  excited-state thermal multiset 初始化统一都走这里。

##### `_max_entangled_ground_mpdm(self, model, set_evolve_config=True)`

- 返回 `MpDm.max_entangled_gs(model)`，并设置压缩与演化配置。
- 关联：
  作为 imaginary-time propagate 路线的起点。

##### `_init_excited_thermal_state(self)`

- 若启用 `electronic_ancilla`：
  - 先构造局域最大纠缠 ground `MpDm`。
  - 再通过 `_build_electronic_ancilla_state` 赋予电子 purification。
  - 最后整体做 multiset 虚时间传播。
- 若不用 ancilla：
  - `imaginary_time_exact` 会给出 warning，因为激发态块的“精确对角热初始化”并不物理严格。
  - 因此无论 exact 还是 propagate，最终都退化为：
    - 对每个对角电子块 `MsModel[alpha][alpha]` 建最大纠缠 `MpDm`
    - 用 `_build_multiset_state` 拼起来
    - 再通过 `_propagate_excited_thermal_msmpdm` 传播
- 关联：
  emission 初态构造核心。

##### `_build_multiset_state(self, msmps)`

- 给每个分量写入压缩与演化配置。
- 如果启用 ancilla，就返回 `ElectronicAncillaMultisetMps`。
- 否则返回普通 `MultisetMps`。
- 关联：
  是 finite-T 类里所有“把若干局域态收集成 multiset”操作的统一出口。

##### `_broadcast_local_state(self, local_state, weights=None)`

- 无 ancilla：
  - 把一个局域态复制到所有电子分量。
  - 每个分量乘上 `weights[alpha]`。
  - 再 `_build_multiset_state`。
- 有 ancilla：
  - 直接走 `_build_electronic_ancilla_state`。
- 关联：
  `_init_dipole` 在输入不是 `MultisetMps` 时就走这里。

##### `_init_dipole(self, state)`

- 第一段：先读取 dipole。
- 第二段：如果 `state` 还只是单个局域态，就通过 `_broadcast_local_state` 广播并乘偶极。
- 第三段：如果 `state` 是 ancilla multiset，就对每个 `(alpha, ancilla)` 分量复制并乘 `dipole[alpha]`。
- 第四段：如果是普通 multiset，就对每个 `alpha` 分量复制并乘 `dipole[alpha]`。
- 关联：
  absorption/emission 的偶极作用都统一通过这里。

##### `_expand_initial_multiset_state(self, state, coef=1e-10)`

- 把初态挂到 `ms_model`。
- 调 `expand_bond_dimension_multiset(use_hint=True)`。
- 给扩展后的每个分量重新写入压缩和演化配置。

##### `_set_multiset_hamiltonian_offset(self, energy)`

- 与 `MultisetChargeDiffusionDynamics._set_hamiltonian_offset` 逻辑同构：
  - 对角块 offset=energy
  - 非对角块 offset=0
  - 然后刷新 MPO 缓存

##### `evolve_single_step(self, evolve_dt)`

- 如果当前步数是奇数，就演化 `ket`。
- 如果是偶数，就按相反时间演化 `bra`。
- 关联：
  finite-T 相关函数采用双分支交替传播。

##### `_evolve_finite_temperature_branch(self, state, evolve_dt)`

- 如果分支是 `MultisetMps`：
  - 先对其中每个局域 `MpDm` 做 ground branch 的局域传播。
  - 再用 `ms_model.evolve_state` 做 excited multiset 传播。
- 如果只是局域 `MpDm`，则只做 ground propagation。
- 关联：
  把 finite-T 关联函数写成“ground branch + excited branch”的组合。

##### `_evolve_ground_multiset(self, state)`

- 遍历 multiset 的每个局域 `MpDm`，逐个调用 `_evolve_ground_mpdm`。

##### `_evolve_ground_mpdm(self, mpdm, evolve_dt)`

- 确保 `qn` 是数组。
- 构造精确局域 ground propagator。
- 调 `mpdm.apply(mpo_prop, canonicalise=True)`。
- 写回局域 `evolve_config/compress_config`。
- 关联：
  finite-T absorption/emission 里基态声子热支路的精确传播器。

##### `_exact_local_ground_propagator(self, x)`

- 用每个 `BasisSHO` 的对角能谱 `exp(x * omega * n)` 构造一个纯对角 MPO。
- 手工填好 `qn/qnidx/qntot/to_right`。
- 关联：
  `_evolve_ground_mpdm` 调用。

##### `_ensure_array_qn(mpdm)`

- 只是把 `mpdm.qn` 中每个元素转成 `np.asarray`。

##### `process_mps(self, mps)`

- 如果 `bra` 是 multiset：
  - 算 pair overlaps。
  - 发射时取共轭并求全和。
  - 吸收时取迹。
- 如果是局域态：
  - 退化成单个 `_state_inner_product`。
- 最后记录 `autocorr`、`autocorr_components`、`bond_dims`。

##### `stop_evolve_criteria(self)`

- 当前总是 `False`，光谱任务不自动截停。

##### `autocorr` / `autocorr_components` / `bond_dims` / `get_dump_dict`

- 与 zero-T 类同样只是输出封装。

##### `_get_dipole(self)`

- 与 zero-T 版本重复，作用也完全相同。

---

## 2. `multiset_202605/` 中各物理模型、所用 multiset 部分与任务

先说结论：这个目录并不全是 multiset 核心算例，还混有 singleset 基线、绘图、结果文件和批处理脚本。下面按“真正定义物理模型的 Python 文件”归类。

### 2.1 真正使用 multiset 的模型

| 路径 | 物理模型 | 用到的 multiset 部分 | 完成的任务 |
| --- | --- | --- | --- |
| `pbi_zt/multiset/PBI.py` | PBI 单体/二聚体/六聚体 Holstein 聚集体；10 个分子振动模，固定分子间耦合 `-500 cm^-1` | `MsEvolveMethod`、`MultisetSpectraZeroT`，下层会进入 `MultisetModel`、`MultisetMps`、`MultisetTdJob` | 计算 0 K 吸收或发射时间关联函数，输出 `autocorr`、bond dims 等 |
| `pbi_ft/multiset/PBI.py` | 同一 PBI 聚集体，但在 298 K 下考虑有限温声子热分布 | `MultisetSpectraFiniteT`，下层会走 `_exact_ground_thermal_mpdm` / `_init_excited_thermal_state` / multiset 演化 | 计算有限温光谱关联函数，当前脚本主设定是 trimer 发射 |
| `pbi_ft_ancilla/multiset/PBI.py` | 同一 PBI 聚集体，但额外对电子自由度做 ancilla purification | `MultisetSpectraFiniteT` + `ElectronicAncillaMultisetMps` 路径 | 计算“声子有限温 + 电子 purification”下的有限温发射关联函数 |
| `P3HT:PCBM/multiset/P3HT.py` | P3HT:PCBM 界面模型；13 个 LE 态 + 13 个 CS 态，含一个反应坐标 `R`、8 个 `F` 模、每个 OT 单元 8 个 `OT` 模 | `MultisetChargeDiffusionDynamics`，下层依赖 `MultisetModel.SplitHamTerm/_ms_evolve_tdvp_ps/expand_bond_dimension_multiset` | 从 `LE1` Franck-Condon 激发出发，跟踪 LE/CS 电荷转移与扩散动力学 |
| `fmo_0k/multiset/fmo.py` | FMO 7 站点激子模型，离散化 35 个声子模，总 Huang-Rhys 因子 0.42 | `MultisetChargeDiffusionDynamics` | 在 0 K 下传播 FMO 激子占据，输出时间序列到 Excel |
| `holstein75_krylov32/multiset/Holstein75_multiset.py` | 75 站点周期性 Holstein 链，每站点一个局域声子模，`g=1.5, omega=1, J=1` | `MultisetChargeDiffusionDynamics`；并猴子补丁 `renormalizer.lib.expm_krylov`、`multiset_model.expm_krylov`、`mps_model.expm_krylov` | 用“固定 32 维 Krylov 子空间”测试/运行 multiset TDVP 传播，并输出每步占据到 Excel |
| `Holstein75_krylovallclose/multiset/Holstein75_multiset.py` | 同一个 75 站点周期性 Holstein 链 | 同上，但 Krylov 收敛准则改成“相邻近似结果 allclose 即停止扩张” | 对比另一种 Krylov 截断策略对 multiset 动力学的影响 |

### 2.2 同目录中的对照/非 multiset 模型

这些文件仍然是物理模型，但不是 multiset 主流程的一部分。

| 路径 | 物理模型 | 是否使用 multiset | 作用 |
| --- | --- | --- | --- |
| `P3HT:PCBM/singleset/P3HT.py` | 和 multiset 版本相同的 P3HT:PCBM LE/CS + 声子模型 | 否，使用 `ChargeDiffusionDynamics` | 作为 multiset 的 singleset 对照算例 |
| ` Holstein_tri/tri.py` | 3 站点三角形 Holstein 模型 | 否，使用 `ChargeDiffusionDynamics` | 作为一个小体系示例/对照，输出 `HolsteinTri3.npz` |

### 2.3 后处理和数据文件

- `plot.py`、`plot/*.ipynb`
  作用：读取 `.npz` / `.xlsx` 结果并可视化，不定义新物理模型。
- `*.npz`、`*.xlsx`
  作用：已有计算结果。
- `run_gpu*.sh`
  作用：SLURM 提交脚本，只负责环境与参数，不改变模型本身。

### 2.4 每个模型更细一点的“物理内容 + multiset 落点”

#### PBI 系列

- 物理模型：
  - 每个分子一个局域电子激发 `elocalex=2.13 eV`。
  - 10 个显著振动模，频率和 Huang-Rhys 因子取自给定数组。
  - 聚集体之间用统一 `-500 cm^-1` 耦合。
- multiset 落点：
  - `construct_model()` 只负责生成普通 `HolsteinModel`。
  - 真正进入 multiset 的入口是 `MultisetSpectraZeroT` 或 `MultisetSpectraFiniteT`。
  - finite-T ancilla 版本会进一步走到 `ElectronicAncillaMultisetMps`、`_build_ancilla_purification_coefficients()`。
- 任务：
  - zero-T：算 0 K absorption/emission 关联函数。
  - finite-T：算 298 K 下的关联函数。
  - ancilla：在 finite-T 上额外测试电子 purification。

#### P3HT:PCBM

- 物理模型：
  - 13 个局域激发态 `LE1~LE13`。
  - 13 个电荷分离态 `CS1~CS13`。
  - LE 链与 CS 链各自有最近邻电子耦合。
  - `LE1 <-> CS1` 有界面耦合。
  - 声子包括一个反应坐标 `R`、8 个共享 `F` 模、每个 OT 单元各自的 8 个局域模。
- multiset 落点：
  - 入口是 `MultisetChargeDiffusionDynamics`。
  - `MultisetModel.SplitHamTerm()` 会把 LE/CS 电子跃迁和声子耦合拆到不同块。
  - `expand_bond_dimension_multiset(use_hint=True)` 很重要，因为初态是 FC 局域激发，跨块耦合会很快驱动其他分量。
- 任务：
  - 从 `LE1` 初态出发，跟踪 LE/CS 占据随时间迁移，比较电荷分离行为。

#### FMO 0 K

- 物理模型：
  - 7 个色素站点，站点能和耦合矩阵来自 `j_matrix_cm`。
  - 用离散化谱密度生成 35 个声子模。
  - 总 Huang-Rhys 因子归一到 0.42。
- multiset 落点：
  - 入口同样是 `MultisetChargeDiffusionDynamics`。
  - 由于是 Holstein 型激子-声子模型，主要走 `MultisetModel` 的标准块拆分和 TDVP 传播。
- 任务：
  - 计算激子在 FMO 中的时间传播，占据写到 Excel。

#### Holstein75 系列

- 物理模型：
  - 75 个等价格点。
  - 周期性边界。
  - 每个格点一个本地声子模。
  - 标准 Holstein 链参数 `omega_0=1, J=1, g=1.5`。
- multiset 落点：
  - 仍然是 `MultisetChargeDiffusionDynamics`。
  - 特别之处不在模型，而在“把 `expm_krylov` 替换成不同策略”，因此真正测试的是 `MultisetModel._ms_evolve_tdvp_ps` 里的局域指数传播器。
- 任务：
  - 比较固定 32 维 Krylov 与严格 allclose 收敛判据对传播结果和成本的影响。

#### Holstein_tri

- 物理模型：
  - 3 个站点构成完全图三角形。
  - 每站点一个局域声子模。
- multiset 落点：
  - 没有进入 multiset，使用的是普通 `ChargeDiffusionDynamics`。
- 任务：
  - 给一个非常小的对照体系，便于画出简单站点占据图。

---

## 3. 不改变算法前提下的简化建议

下面只提“重构、去重复、提高清晰度、减小维护成本”的建议，不涉及改动物理模型或 TDVP / Krylov / 热态初始化算法本身。

### 3.1 `renormalizer/multiset/` 核心层建议

#### A. `multiset_mps.py`

- 把 `MultisetMps` 和 `ElectronicAncillaMultisetMps` 里重复的 `copy`、`to_complex`、`ph_occupations_multiset`、`dump/load` 骨架抽成私有 helper。
  原因：当前两类状态只是在“索引展开方式”不同，序列化和逐分量遍历逻辑基本重复。
- 增加一个统一的“逐物理分量迭代器”接口。
  原因：`rho_el`、`e_occupations_multiset`、`ph_occupations_multiset`、`multiset_spectra.py` 里都在手写“普通态/ancilla 态双分支”。
- `ms_normalize` 只支持 `"mps_only"`，但接口名字像是支持多模式。
  建议：要么明确收窄接口名，要么保留兼容但把未实现模式集中报错到统一位置。

#### B. `multiset_mpo.py`

- `apply()` 里的 ancilla 分支和普通分支高度重复，建议抽成“给定目标分量 alpha，收集所有 beta 贡献”的共享 helper。
- `zero_template` 的构造逻辑重复两次，建议抽私有函数。
- `compress_config` 的反复写入可以封装成小函数。
  原因：这些都是结构性去重复，不碰算法。

#### C. `multiset_model.py`

- `SplitHamTerm`、`ConstructMsModel`、`ConstructInitModel` 建议统一成 snake_case，并保留旧名别名一段时间。
  原因：现在新旧命名风格混用，阅读负担大。
- `_set_hamiltonian_offset` / `_set_multiset_hamiltonian_offset` 的逻辑应下沉到 `MultisetModel`。
  原因：`multiset_tdjob.py` 和 `multiset_spectra.py` 各写了一份几乎相同的版本。
- `_ms_evolve_tdvp_ps` 过长，建议拆成几个私有步骤：
  - `_prepare_evolve_state`
  - `_build_local_ivp`
  - `_split_site_tensor_qr`
  - `_propagate_center_tensor`
  - `_writeback_sweep_step`
  原因：当前算法主干是清楚的，但函数体太长，不利于单独检查 bug。
- 把“普通态/ancilla 态”的递归扩维逻辑从 `expand_bond_dimension_multiset` 中拆出去。
  原因：这会让非 ancilla 主体更清晰。
- `population()` 与 `popultation()` 至少要在文档里保留一个主名，另一个仅做兼容别名。
  原因：现在拼写错误会让外部 API 看起来不稳定。

#### D. `multiset_tdjob.py`

- `evolve()` 里 startup substeps 之后那一大段“记录时间、打印、dump”的逻辑与主循环重复，建议抽成一个 `_finalize_step(new_mps, step_index)`。
- `init_mp()` 里三种有限温初态方法可以改成字典分派，而不是连续 `if method in [...]`。
- `observables` 建议定义成常量默认表或 dataclass。
  原因：当前构造函数里直接内嵌字典，可读性一般。

#### E. `multiset_spectra.py`

- `MultisetSpectraZeroT._get_dipole` 和 `MultisetSpectraFiniteT._get_dipole` 完全重复，应抽成共享 helper。
- `init_mps_abs()` 和 `init_mps_emi()` 在 zero-T 类里目前实现完全一样，建议合并。
- `process_mps()` 在 zero-T 和 finite-T 中也有很强相似性，建议抽共享“记录 autocorr/bond_dims”的逻辑。
- 电子 ancilla 相关逻辑已经很长，建议单独抽到一个内部模块，例如 `multiset_electronic_purification.py`。
  原因：不是为了改算法，而是为了把 finite-T 主流程从 ancilla 细节里解耦。

### 3.2 `multiset_202605/` 算例层建议

#### A. PBI 三套脚本应合并公共部分

- `pbi_zt/multiset/PBI.py`、`pbi_ft/multiset/PBI.py`、`pbi_ft_ancilla/multiset/PBI.py` 的 `construct_model()` 基本相同，建议抽成一个共享模块，例如 `multiset_202605/pbi_common.py`。
- 三个 `main()` 的差别主要只是：
  - 选 zero-T 还是 finite-T 类；
  - 是否开 `electronic_ancilla`；
  - 默认聚集体大小和温度。
  这些都可以变成参数，而不必复制三份脚本。

#### B. P3HT 的 multiset / singleset 版本应共享模型定义

- `P3HT:PCBM/multiset/P3HT.py` 和 `P3HT:PCBM/singleset/P3HT.py` 的 `P3HTPCBMModel` 与 `nbas_from_gw()` 完全重复。
- 建议抽成同目录下的 `p3ht_model.py`，multiset 与 singleset 脚本只保留各自的作业入口。
- 这样不会改变任何哈密顿量，只是减少维护时“双处改同一个模型”的风险。

#### C. Holstein75 两个脚本应只保留一个主体

- `holstein75_krylov32/...` 和 `Holstein75_krylovallclose/...` 除了 `expm_krylov` 策略不同，其余模型与作业逻辑几乎完全重复。
- 建议：
  - 保留一个共享 `Holstein75_base.py`；
  - 把 Krylov 策略实现写成两个函数；
  - 用环境变量或命令行参数选择 `fixed32` / `strict_allclose`。
- 这不会改变传播算法，只是把“策略差别”从“整文件复制”改成“单个可切换函数”。

#### D. FMO 脚本应清理无关导入和输出逻辑

- `fmo.py` 里有不少没用到的导入：`ChargeDiffusionDynamics`、`InitElectron`、`EvolveConfig`、`CompressConfig`、`CompressCriteria`、`EvolveMethod`、`sys` 等。
- 建议删掉未用导入，并把“保存 Excel”的逻辑封成一个小函数。
- 这样不改模型、不改传播，只是让文件焦点更集中。

#### E. `run_gpu*.sh` 应提炼公共模板

- 多个 shell 脚本都在重复：
  - 加载 `.bashrc`
  - 加载 conda/env
  - `module load cuda`
  - 打印环境信息
  - 执行 Python
- 建议抽一个公共 SLURM 模板，脚本里只保留作业名和入口 Python 文件。
- 另外，`pbi_ft` 和 `pbi_ft_ancilla` 的 `python PBI.py \` 这一行末尾反斜杠没有实际意义，应该去掉。
  这属于脚本清理，不改算法。

### 3.3 可以按优先级推进的精简顺序

如果你现在准备 `push`，建议按下面顺序做，风险最低。

1. 先做算例层去重复。
   包括：PBI 公共模型、P3HT 公共模型、Holstein75 公共主体、清理 FMO 导入、统一 shell 模板。
   原因：这些都不碰底层 multiset 算法，回归风险最低。

2. 再做工具层去重复。
   包括：`_get_dipole` 共享、offset 设置共享、`MultisetTdJob.evolve` 的记录/dump 抽函数。
   原因：逻辑清楚、单测点容易补。

3. 最后才拆 `MultisetModel._ms_evolve_tdvp_ps`。
   原因：这是算法核心，虽然最值得重构，但也最容易在无意中改变行为。建议在你 push 前如果时间紧，就先不动这里，只做“抽私有 helper、不改调用顺序”的最小重构。

### 3.4 我认为最值得优先改的 8 条

- 抽出 `pbi_common.construct_model()`。
- 抽出 `P3HTPCBMModel` 到共享文件。
- 合并 Holstein75 两个脚本的主体。
- 删除 `fmo.py` 未用导入。
- 去掉 shell 脚本里无效的行尾反斜杠。
- 抽出共享 `_get_dipole()`。
- 抽出共享 `set_multiset_hamiltonian_offset()`。
- 给 `MultisetTdJob.evolve()` 抽一个统一的 step 后处理函数。

---

## 结语

从结构上看，`multiset` 这套实现已经形成了比较清楚的三层：

- 表示层：`MultisetMps` / `ElectronicAncillaMultisetMps` / `MultisetMpo`
- 算法层：`MultisetModel`
- 任务层：`MultisetTdJob` / `MultisetChargeDiffusionDynamics` / `MultisetSpectraZeroT` / `MultisetSpectraFiniteT`

真正适合在当前阶段做的“简化”主要不是改算法，而是：

- 去重复；
- 把共享逻辑下沉；
- 把超长函数拆成辅助函数；
- 把算例脚本从“复制文件”改成“参数化入口”。

这样最有利于你后续 `push`、review 和继续扩展。
