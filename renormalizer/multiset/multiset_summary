# `renormalizer/multiset` 代码归纳与 `test/` 中 multiset 结果汇总

本文档完全替换原来的“优化说明”。  
这里不再记录某一次优化过程，而是基于当前工作区源码，对 `renormalizer/multiset` 的实现做系统归纳，并把 `test/` 目录下已经提交的 multiset 体系、脚本和结果文件统一整理出来。

## 1. 整体设计总览

### 1.1 multiset 这一层到底做了什么

当前这套 multiset 实现，本质上是把 singleset 的

- `单个 Mps / MpDm`
- `单个 Mpo`
- `单条 TD 演化作业`
- `单条光谱关联函数流程`

扩展成了“电子态分块 + 每个电子块各自携带一个声子 MPS/MpDm”的形式。

其抽象可以写成

\[
|\Psi\rangle = \sum_\alpha |\alpha\rangle \otimes |\phi_\alpha\rangle
\]

其中：

- `|alpha>` 是电子局域态或电子激发态标签。
- `|phi_alpha>` 是对应电子块上的声子 MPS / MPDM。
- 哈密顿量被拆成 `H^{alpha,beta}` 的 block 结构。
- 动力学和光谱都在这些 block 上做批量收缩与演化。

### 1.2 各文件与 singleset 对应关系

| 文件 | 对应 singleset 基础 | 当前 multiset 角色 |
| --- | --- | --- |
| `multiset_mps.py` | `renormalizer/mps/mps.py` | 管理一组 `Mps/MpDm`，形成整体 multiset 态 |
| `multiset_mpo.py` | `renormalizer/mps/mpo.py` | 管理 `N x N` block MPO，以及通用 block 算符 |
| `multiset_model.py` | `renormalizer/mps/mps.py` + `mpo.py` | 从原始 `Model` 拆出 multiset block 模型，并实现 multiset TDVP 演化核心 |
| `multiset_tdjob.py` | `renormalizer/utils/tdmps.py` + `renormalizer/transport/dynamics.py` | 多体系动力学驱动器与电荷扩散作业 |
| `multisetspectra.py` | `renormalizer/spectra/zerot.py` + `renormalizer/spectra/finitet.py` | multiset 零温/有限温光谱关联函数流程 |
| `__init__.py` | 包导出层 | 对外暴露 multiset API |

### 1.3 当前代码的几个关键特征

- `MultisetModel` 会把原始含电子自由度的 `Model` 拆成只含声子基的 `MsModel[alpha][beta]`。
- `MultisetMpo` 把这些 block `Model` 变成 `N x N` 的 `Mpo` 矩阵。
- `MultisetMps` 则保存长度为 `N_electron` 的 `msmps` 列表，每个元素是一个 `Mps` 或 `MpDm`。
- TDVP 演化核心在 `MultisetModel._ms_evolve_tdvp_ps`，并且已经做了 active pair 稀疏化和 batched contraction。
- 光谱部分已经同时支持：
  - 零温吸收
  - 零温发射
  - 有限温吸收
  - 有限温发射
- 动力学部分已经支持：
  - 零温电荷扩散
  - 三种有限温初态构造：`imaginary_time_exact`、`thermo_field`、`imaginary_time_propagate`

### 1.4 当前代码中能直接观察到的实现状态

- 当前 `MsEvolveMethod` 只定义了一个方法：`ms_evolve_tdvp_ps`。
- `MultisetModel.evolve_state()` 实际也只分派到 `_ms_evolve_tdvp_ps`。
- `_ms_evolve_tdvp_ps()` 当前真正完整实现的是 `ivp_solver == "krylov"` 分支；代码里对非 Krylov 分支只留下了部分变量准备，没有完整演化路径。
- `renormalizer/multiset/__init__.py` 里导出了 `MultisetTransportKubo`，但在当前 `multiset_tdjob.py` 中并没有这个类定义；这说明包导出层与实际源码目前不完全一致。
- `MultisetModel` 里同时保留了 `population()` 和拼写别名 `popultation()`。

## 2. 包导出层：`renormalizer/multiset/__init__.py`

当前对外导出的对象有：

- `MsEvolveMethod`
- `MultisetMps`
- `MultisetBlockMpo`
- `MultisetMpo`
- `MultisetModel`
- `MultisetSpectraFiniteT`
- `MultisetSpectraZeroT`
- `MultisetChargeDiffusionDynamics`
- `MultisetTdJob`
- `MultisetTransportKubo`

其中最后一个 `MultisetTransportKubo` 在当前源码中没有实际定义，是一个导出层残留/预留项。

## 3. `multiset_mps.py` 归纳

### 3.1 `MsEvolveMethod` 类

#### `ms_evolve_tdvp_ps`（L17）

- 枚举项字符串为 `TDVP PS one-site with multiset-mps ansatz`。
- 当前 multiset 演化方法唯一的合法枚举值。

### 3.2 `MultisetMps` 类

这个类是 multiset 态容器。  
它不直接定义物理模型，而是保存一组并列的 `Mps/MpDm`，每个分量对应一个电子块。

#### `__init__(self, msmodel, N_electron, temperature, init_model, method, init_mp, msmps)`（L26）

- 初始化 multiset 态。
- 强制要求二选一：
  - 要么提供 `init_mp`，表示复制一个初始局域态到所有电子块；
  - 要么直接提供 `msmps`，表示显式给出每个 block 的状态。
- 保存：
  - `MsModel`
  - `N_electron`
  - `temperature`
  - `init_model`
  - `method`
- 最后调用 `_ConstructMsMps()` 真正构造 `self.msmps`。

#### `_ConstructMsMps(self)`（L50）

- 如果输入的是 `msmps`：
  - 检查长度是否与 `N_electron` 一致；
  - 对每个状态做 `.copy()`，避免共享对象。
- 如果输入的是 `init_mp`：
  - 把同一个 `init_mp` 深拷贝到每个电子块。

#### `copy(self)`（L63）

- 返回一个新的 `MultisetMps` 深拷贝。
- 会复制：
  - `MsModel`
  - `N_electron`
  - `temperature`
  - `init_model`
  - `method`
  - 每个 `mps` 分量

#### `to_complex(self)`（L76）

- 与 `copy()` 类似，但把每个分量都转换成复数 MPS。
- 主要用于实时间演化前，确保张量 dtype 适合复数传播。

#### `total_mps(self)`（L89）

- 用 `_sum(..., compress=False)` 把所有 block 态直接相加成一个总 MPS。
- 这个总 MPS 是“把 multiset 分量叠加回去”的一个便捷观察视角。

#### `ms_normalize(self, kind)`（L92）

- 计算全 multiset 态的总范数：
  - `sum_alpha <phi_alpha|phi_alpha>`
- 当前只支持 `kind == "mps_only"`：
  - 把每个 `mps` 分量都缩放同样的系数 `1 / sqrt(total_norm)`。
- 不支持的模式会直接报错。

#### `rho_el(self)`（L120）

- 构造电子约化密度矩阵。
- 元素定义为：
  - `rho[alpha, beta] = <phi_alpha | phi_beta>`
- 返回复数矩阵，大小为 `N_electron x N_electron`。

#### `e_occupations_multiset`（L129）

- 电子占据数属性。
- 返回 `rho_el()` 对角元的实部。

#### `ph_occupations_multiset`（L133）

- 声子占据数属性。
- 对每个 block MPS 的 `ph_occupations` 求和。
- 如果总结果虚部接近 0，则只返回实部。
- 如果一个分量都没有，则返回空数组。

#### `dump(self, fname)`（L148）

- 保存 multiset 态到磁盘。
- 机制不是把所有张量塞进一个文件，而是：
  - 每个 block 单独写成 `root_set{i}.npz`
  - 主 manifest 文件记录：
    - `version`
    - `N_electron`
    - `temperature`
    - `method`
    - `state_paths`
    - `state_types`

#### `load(cls, msmodel, N_electron, fname, init_model)`（L173）

- 从 manifest 文件恢复 `MultisetMps`。
- 逐个读取 `state_paths` 对应的文件。
- 仅支持两种已知子状态类型：
  - `MpDm`
  - `Mps`
- 恢复完成后把所有分量塞回 `new.msmps`。

## 4. `multiset_mpo.py` 归纳

### 4.1 `MultisetMpo` 类

这是最基础的 block-MPO 容器。  
它维护一个 `N_electron x N_electron` 的 `msmpo` 矩阵，每个元素都是一个 `Mpo` 或空列表。

#### `__init__(self, msmodel, N_electron)`（L17）

- 接收 block 形式的 `MsModel`。
- 创建空的 `msmpo` 二维列表。
- 调用 `_ConstructMsMpo()` 完成真实构造。

#### `_ConstructMsMpo(self)`（L23）

- 遍历每个 `MsModel[i][j]`：
  - 如果 `ham_terms` 为空，则该 block 记为 `[]`
  - 否则构造对应的 `Mpo(model=self.MsModel[i][j], terms=None)`

#### `total_mpo(self)`（L31）

- 先对每一行 `self.msmpo[alpha]` 做一次 `_sum`
- 再对所有行和再做一次 `_sum`
- 最后得到一个整体单 MPO 视角的“总 block 算符”

### 4.2 `MultisetBlockMpo` 类

这是一个更通用的 block 算符类。  
它表示

\[
O = \sum_{\alpha,\beta} |\alpha\rangle\langle\beta| \otimes O^{\alpha\beta}
\]

也就是“电子块标签 + 声子局域 MPO”的通用算符形式。

#### `__init__(self, basis_set, ms_terms, N_electron, compress_config)`（L48）

- `ms_terms` 是 `N x N` 的 block 哈密顿量/算符项列表。
- 用这些项先构造 `msmodel[a][b] = Model(basis=basis_set, ham_terms=ms_terms[a][b])`。
- 再调用父类 `MultisetMpo` 来构造真正的 `msmpo`。
- 同时生成 `_active_pairs`：
  - 只记录那些 `len(self.msmpo[alpha][beta]) != 0` 的活跃 block。

#### `has_terms`（L70）

- 属性。
- 若 `_active_pairs` 非空则返回 `True`。

#### `apply(self, ms_state)`（L73）

- 对一个 `MultisetMps` 施加该 block MPO。
- 对每个输出电子块 `alpha`：
  - 枚举所有输入块 `beta`
  - 若 `msmpo[alpha][beta]` 非空，则对 `ms_state.msmps[beta]` 做 `mpo.contract(source)`
  - 每个贡献先单独 `normalize("mps_norm_to_coeff")`
  - 多个贡献用 `_sum(..., compress=True)` 合并
- 若某个 `alpha` 没有任何贡献，则返回一个全零模板状态。

#### `matrix_element(self, bra_state, ket_state)`（L114）

- 计算 block 算符矩阵元 `<bra|O|ket>`。
- 只遍历 `_active_pairs`，避免空块。
- 每一项调用：
  - `ket.expectation(mpo, self_conj=bra_conj[alpha])`
- 最后再乘以 `bra.coeff` 与 `ket.coeff` 的复数因子。

## 5. `multiset_model.py` 归纳

这个文件是当前 multiset 实现里最核心的一层。  
它负责：

- 从原始 `Model` 拆出 block Hamiltonian
- 构造 `MsModel / MsMpo / MsMps`
- 缓存活跃 block 和 site 分组模板
- 实现 batched TDVP-PS 演化
- 实现观测量与 bond expansion

### 5.1 `MultisetModel` 类

#### `__init__(self, model, max_bonddim, temperature, method, evolve_config, compress_config, auto_init)`（L26）

- 保存原始 `model`、温度和初态构造方法。
- 若未显式传入 `evolve_config`，默认采用：
  - `EvolveConfig(method=MsEvolveMethod.ms_evolve_tdvp_ps)`
- 若未显式传入 `compress_config`，默认采用固定 bond 维：
  - `CompressCriteria.fixed`
  - `max_bonddim=max_bonddim`
- 关键数据成员：
  - `N_electron = model.n_edofs`
  - `basis_set`：删除了 `BasisSimpleElectron` 之后的基组，只保留声子/辅助基
  - `MsModel`：`N x N` block `Model`
  - `MsOp`：`N x N` block `Op` 列表
- 初始化流程：
  1. `SplitHamTerm()`
  2. `ConstructInitModel()`
  3. `ConstructMsModel()`
  4. 构造 `self.MsMpo`
  5. 初始化 active pair / QR cache / site template cache
  6. 调用 `_refresh_mpo_cache()`
- `auto_init` 当前不会自动生成 `MsMps`，而只打印提示，要求由上层 `MultisetTdJob` 子类显式准备初态。
- 还维护两个统计计数器：
  - `_matvec_calls`
  - `_ivp_calls`

#### `SplitHamTerm(self)`（L80）

- 遍历原始 `model.ham_terms`，把每一项分发到对应的 multiset block。
- 规则：
  - 如果某一项不包含电子跃迁，则复制到所有对角块 `MsOp[alpha][alpha]`
  - 如果包含电子跃迁，则只放进对应的 `MsOp[alpha][beta]`
- 这里实际用到 `_get_electron_transition()` 和 `_reset_all_MsOp()`。

#### `_get_electron_transition(self, op)`（L91）

- 从一个 `Op` 里解析电子部分。
- 它会扫描 `op.dofs` 与 `op.split_symbol`，只收集属于 `self.model.e_dofs` 的电子算符。
- 约束非常明确：
  - 若电子算符数为 0，则返回 `None`
  - 若电子算符数不是 2，则报错
  - 两个电子符号必须正好是 `{a^\dagger, a}`，否则报错
- 返回规则：
  - 若第一个符号是 `a^\dagger`，返回 `(dof1, dof2)`
  - 否则返回 `(dof2, dof1)`
- 也就是说，返回的是“从 `beta` 跳到 `alpha`”的 block 索引。

#### `_reset_all_MsOp(self, op)`（L112）

- 把原始算符里的电子部分全部剥离，只留下声子/非电子自由度上的那部分算符。
- 若剥离后一个符号都不剩：
  - 就把该算符重置为单位算符 `I`
  - `qn_list = [0]`
  - `dofs = [(0, 0)]`
- 这个处理对纯电子耦合项非常关键，因为它们在 phonon-only block 表示里应该对应“电子块间耦合系数乘以声子空间恒等算符”。

#### `ConstructMsModel(self)`（L138）

- 把 `MsOp[i][j]` 逐块包装成 `Model(basis=self.basis_set, ham_terms=self.MsOp[i][j])`。
- 结果写回 `self.MsModel[i][j]`。

#### `ConstructInitModel(self)`（L143）

- 构造一个用于初态/热初态准备的 `init_model`。
- 它只保留原始哈密顿量里满足 `len(op.dofs) == 1` 的项，再做一次 `_reset_all_MsOp(op)`。
- 这意味着：
  - `init_model` 主要是局域单体项
  - 不含电子自由度
  - 也不含需要跨 site 的耦合项

#### `_refresh_mpo_cache(self)`（L150）

- 当前 multiset 实现的一个核心优化与缓存函数。
- 它做两类事情：

1. 识别活跃 block

- 遍历所有 `alpha, beta`
- 只把 `len(mpo) > 0` 的 block 加入：
  - `_active_pairs`
  - `_active_pair_mpos`
  - `_active_pairs_by_alpha`
  - `_active_alpha`
  - `_active_beta`

2. 为每个 site 预构造 batched 模板

- 对每个活跃 block 的每个 site：
  - 读取 `local_mpo.array`
  - 按 `W.shape` 分桶
  - 累积：
    - `pair_ids`
    - `alpha_idx`
    - `beta_idx`
    - `w_tensors`
- 最后为每个 site 生成 `self._site_group_templates`，每个模板项里保存：
  - `pair_ids`
  - `alpha_idx`
  - `beta_idx`
  - `S`：scatter matrix
  - `W`：stack 后的局域 MPO 张量
  - `nsite`
  - `n_pairs`

#### `_build_scatter_matrix(self, alpha_idx, dtype)`（L215）

- 根据 `alpha_idx` 构造一个 `N_electron x n_pairs` 的散射矩阵 `S`。
- 它的作用是把 pair-space 的 batched 输出重新加回 alpha-space。

#### `_get_qr_qn_plan(self, qnbigl, qnbigr, qntot)`（L221）

- 为 batched QR / RQ 预生成量子数分块方案。
- 缓存键由：
  - `qnbigl.tobytes()`
  - `qnbigr.tobytes()`
  - `qntot.tobytes()`
 组成。
- 具体逻辑：
  - 枚举左量子数 `nl`
  - 用 `nr = qntot - nl`
  - 在右量子数集合里寻找匹配 block
- 输出的 `plan` 每项是：
  - `(lset, rset, nl, nr)`

#### `_batched_qr_qn(self, coef_batch, qnbigl, qnbigr, qntot, system, max_rank)`（L259）

- 对一批 multiset branch 系数矩阵同时做量子数分块 QR / RQ。
- 输入 `coef_batch` 的第 0 维是 batch，也就是 `alpha` 分量。
- `system == "L"`：
  - 直接做 `xp.linalg.qr(block, mode="reduced")`
- `system == "R"`：
  - 通过对转置块做 QR，再转回，等价实现 RQ
- 每个 qn block 做完后，再恢复为完整大矩阵的稀疏填充形式：
  - `u_full`
  - `vt_full`
- 最后把所有 qn block 的结果在最后一维拼起来。
- 若给了 `max_rank`，会在拼接后截断到对应 rank。

#### `_build_site_batched_data(self, imps, l_tensors, r_tensors)`（L304）

- 把 `_site_group_templates[imps]` 与当前 sweep 时读出的 `L/R` 环境张量结合起来，生成真正用于 matvec 的 batched group。
- 每个 group 最终包含：
  - `L`
  - `R`
  - `S`
  - `W`
  - `alpha_idx`
  - `beta_idx`
  - `nsite`
  - `n_pairs`

#### `_build_reverse_batched_data(self, l_tensors, r_tensors)`（L322）

- 用于 one-site TDVP 中间 `U` / `C` 的反向半步传播。
- 与 `_build_site_batched_data()` 的差别是：
  - 这里没有局域 `W`
  - `nsite = 0`
  - 按 `l_tensor.shape[1]` 对 group 做分桶

#### `reset_mps(self, init_mp, msmps)`（L359）

- 新建一个 `MultisetMps` 并保存到 `self.MsMps`。
- 初始化完后立刻做一次 `ms_normalize("mps_only")`。

#### `set_mps(self, ms_mps)`（L372）

- 单纯把外部 `MultisetMps` 绑定到 `self.MsMps`。

#### `evolve_state(self, ms_mps, evolve_dt, normalize)`（L376）

- 演化入口。
- 按 `self.evolve_config.method` 做分派。
- 当前字典里只有一个键：
  - `MsEvolveMethod.ms_evolve_tdvp_ps -> self._ms_evolve_tdvp_ps`
- 演化后若 `normalize=True`，会对结果再做一次 `ms_normalize("mps_only")`。

#### `evolve(self, evolve_dt, normalize)`（L384）

- `evolve_state()` 的 in-place 包装。

#### `_ms_evolve_tdvp_ps(self, ms_mps_, ms_mpo, evolve_dt)`（L387）

- 这是当前 multiset 演化最核心的方法。
- 本质是 one-site TDVP projector-splitting 在 multiset block 上的 batched 版本。

其主要流程如下：

1. 处理时间步与 dtype

- 若 `evolve_dt` 为复数，说明在做虚时间或带复数时间演化：
  - 复制 `ms_mps`
- 若 `evolve_dt` 为实数：
  - 调用 `to_complex()` 转成复数态

2. 只为活跃 block 构造环境

- 对每个 `pair_id` 创建 `Environ(...)`
- bra 使用 `alpha` 分量的共轭
- ket 使用 `beta` 分量本身

3. 双向 sweep

- 外层 `for i in range(2)`，表示完整左扫 + 右扫两轮
- 对每个 MPS site：
  - 读所有活跃 pair 的 `L/R` 环境
  - 用 `_build_site_batched_data()` 组装 batched contraction 数据
  - 把所有 `alpha` 的当前 site tensor 堆成一个 `Y0`
  - 构造 `ivp_eq = lambda Y: self._apply_hop_batched(...)`
  - 当前实现通过 `expm_krylov()` 做半步传播

4. batched QR / RQ 正交化

- 用 `_batched_qr_qn()` 同时处理所有 `alpha` 分量的局域张量
- 根据 sweep 方向把结果写回：
  - 向右规范时写 `u_batch`
  - 向左规范时写 `vt_batch`

5. 处理中间 `U` / `C` 的反向半步

- 若不是边界 site：
  - 构造 `r_array_u` 或 `l_array_c`
  - 再用 `_build_reverse_batched_data()` 生成 batched group
  - 对 `U0` 或 `C0` 再做一次 Krylov 半步
  - 再把演化结果 tensordot 回相邻 site

6. sweep 完成后切换规范方向

- 每轮 sweep 后，对每个 `alpha` 分量调用 `_switch_direction()`

7. 统计 Krylov 空间维数

- 所有局部 Krylov 维度收集在 `local_steps`
- 最后用 `stats.describe(local_steps)` 保存到 `self.evolve_config.stat`

这个方法的实现特点：

- 不是逐个 `alpha, beta` 分别演化，而是尽可能按 `pair` 和 `site` 做 batched 计算。
- `active_pairs` 明显减少了对空 block 的遍历。
- `site_group_templates` 避免了每个 step 重复分组 `W` 张量。
- 当前代码里真正有完整结果赋值的是 Krylov 分支，所以这个实现目前实际依赖 `ivp_solver == "krylov"`。

#### `expand_bond_dimension_multiset(self, coef, use_hint)`（L539）

- multiset 版本的 bond-dimension 扩张。
- 所有分量先统一绑定 `self.compress_config`。

`use_hint=False` 时：

- 每个 `alpha` 分量单独调用 `expand_bond_dimension_general(...)`
- 不使用任何对角/非对角 block 信息。

`use_hint=True` 时：

- `hint_mpo` 取对角块 `MsMpo.msmpo[alpha][alpha]`
- 额外驱动方向 `ex_mps` 来自所有活跃非对角块：
  - `self._active_pair_mpos[pair_id].apply(original_mps[beta])`
- 若存在多个非对角驱动态，则 `_sum(cross_states, compress=False)` 合并成 `ex_mps`
- 最后再送入 `expand_bond_dimension_general(...)`

这个函数的意义是：

- 对角哈密顿量提供局域结构提示
- 非对角 block 提供跨电子块耦合引起的额外纠缠方向提示

#### `population(self)`（L588）

- 返回 `self.MsMps.e_occupations_multiset`。

#### `popultation(self)`（L591）

- `population()` 的拼写别名。

#### `Hamiltonian(self)`（L594）

- 计算归一化期望值 `<Psi|H|Psi>/<Psi|Psi>`。
- 只遍历活跃 block：
  - `num += <phi_alpha | H^{alpha,beta} | phi_beta>`
- 分母是所有 block 范数之和。
- 最后只返回实部。

#### `rho_el(self)`（L605）

- 直接转调 `self.MsMps.rho_el()`。

#### `decoherence_metrics(self)`（L608）

- 基于 `rho_el()` 返回三项：
  - `tr = trace(rho).real`
  - `purity = trace(rho @ rho).real`
  - `rho`

#### `Inner_product(self)`（L616）

- 先把所有 block 态通过 `total_mps()` 求和，再做总内积。

#### `_apply_hop_batched(self, Y, batched_groups, dim, shape)`（L619）

- 这是 Krylov matvec 所调用的 batched Hamiltonian 作用函数。
- 输入 `Y` 先 reshape 成 `(N_electron, dim)`。
- 对每个 batched group：
  - 取出 `beta_idx` 对应的输入分量 `Y_exp`
  - 若 `nsite == 1`：
    - 使用 `W`
    - 对三阶 local tensor 走一套 einsum
    - 对四阶 local tensor 再走一套 einsum
  - 若 `nsite == 0`：
    - 不用 `W`
    - 只处理 `U/C` 类 rank-2 张量
- 最终把 pair-space 输出乘 `S` 再加到 `Y_out`
- 每调用一次，`self._matvec_calls += 1`

## 6. `multiset_tdjob.py` 归纳

这个文件提供的是“作业驱动层”。  
它相当于 singleset 的 `TdMpsJob` + `ChargeDiffusionDynamics` 的 multiset 化版本。

### 6.1 模块级辅助函数

#### `_thermal_coefficients_from_theta(basis, temperature)`（L29）

- 输入一个 `BasisSHO` 和温度。
- 通过：
  - `ratio = exp(-beta * omega / 2)`
  - `theta = arctanh(ratio)`
  - `weights = tanh(theta)^n / cosh(theta)`
- 构造 thermofield 型初态系数。
- 最后归一化。

#### `_calc_r_square_multiset(e_occupations)`（L39）

- 根据电子占据分布计算离散站点索引的方差：
  - `mean(r^2) - mean(r)^2`
- 用于表征扩散宽度。

#### `_state_bond_dims(state)`（L48）

- 提取状态的 bond 维摘要。
- 支持：
  - `MultisetMps`
  - 任何有 `bond_dims` 属性的对象
  - tuple/list 的递归嵌套情况

### 6.2 `MultisetTdJob` 类

这是一个抽象的 multiset 时间演化作业框架，负责：

- 初态创建
- 时间步推进
- 日志
- dump/checkpoint
- 启动子步（startup substeps）

#### `__init__(self, evolve_config, dump_mps, dump_dir, job_name, if_startup_substeps, startup_substeps_n)`（L68）

- 保存作业级配置。
- 允许的 `dump_mps` 取值：
  - `None`
  - `"all"`
  - `"one"`
- 初始化：
  - `evolve_times = [0]`
  - `info_interval = 1`
  - 各类 dump 参数
- 然后立即：
  1. 调用 `self.init_mps()`
  2. 保存为 `self.latest_mps`
  3. 调用 `self.process_mps(mps)` 记录第 0 个时间点

#### `init_mps(self)`（L113）

- 抽象方法。
- 子类必须返回一个 multiset 态或与之配套的态组。

#### `process_mps(self, mps)`（L116）

- 抽象方法。
- 子类负责把当前态转换成观测量并记录。

#### `evolve_single_step(self, evolve_dt)`（L119）

- 抽象方法。
- 子类负责真正的一步演化。

#### `get_dump_dict(self)`（L122）

- 抽象方法。
- 子类负责把当前已累计的观测量打包成可 `np.savez` 的字典。

#### `stop_evolve_criteria(self)`（L125）

- 默认返回 `False`。
- 子类可重载停止条件。

#### `_run_startup_substeps(self, evolve_dt)`（L128）

- 实现启动阶段的渐进式子步传播。
- 子步时间长度在 `|dt| * 1e-5` 到 `|dt|` 之间按对数均匀分布。
- 用于在第一个时间步时更平滑地启动演化。

#### `_checkpoint_state_path(self)`（L146）

- 计算 checkpoint 路径：
  - 若 `dump_mps == "all"`，文件名带步号
  - 否则始终覆盖 `job_name_mps.npz`

#### `_dump_state(self, state, fname)`（L153）

- 通用递归 dump 逻辑。
- 若对象有 `.dump()` 方法，则直接调用。
- 若是 tuple/list，则递归存储每个元素，再写一个 manifest 文件记录 `item_paths`。

#### `dump_dict(self)`（L179）

- 把 `get_dump_dict()` 的结果写入 `job_name.npz`。
- 为避免意外中断损坏，会先用 `.bak` 备份旧文件。
- 若开启了 `dump_mps`，还会把 `self.latest_mps` 一并 checkpoint。

#### `evolve(self, evolve_dt, nsteps, evolve_time)`（L201）

- 通用演化控制器。
- 支持多种调用方式：
  - 给 `evolve_dt + nsteps`
  - 给 `evolve_dt + evolve_time`
  - 给 `nsteps + evolve_time`
  - 只给 `evolve_dt`，则无限步直到 `stop_evolve_criteria()`
- 若开启 startup substeps，并且当前还是第一个大步，则会先走 `_run_startup_substeps()`。
- 每一步都做：
  1. `new_mps = evolve_single_step(evolve_dt)`
  2. 更新 `evolve_times`
  3. `process_mps(new_mps)`
  4. 视需要 dump
  5. 检查 `stop_evolve_criteria()`

#### `latest_evolve_time`（L340）

- 当前最后一个时间点。

#### `evolve_times_array`（L344）

- `evolve_times` 的 `numpy` 形式。

#### `_defined_output_path`（L348）

- 只有当 `dump_dir` 和 `job_name` 都存在时才返回 `True`。

### 6.3 `MultisetChargeDiffusionDynamics` 类

这是电荷扩散作业的 multiset 版本。  
它把 `MultisetModel` 当作演化器，并在每个时间步记录扩散观测量。

#### `__init__(...)`（L358）

- 两种用法：
  - 直接给 `ms_model`
  - 给原始 `model + max_bonddim`，由本类内部创建 `MultisetModel`
- 若内部新建 `MultisetModel`：
  - `auto_init=False`
- 额外控制参数：
  - `initial_site`
  - `stop_at_edge`
  - `edge_threshold`
  - `use_init_hint`
  - `dump_mps`
  - `startup_substeps`
- 初始化观测量容器：
  - `energies`
  - `r_square_array`
  - `e_occupations_array`
  - `ph_occupations_array`
  - `reduced_density_matrices`
  - `coherent_length_array`
  - `purity_array`
  - `trace_array`
- 最后调用父类 `MultisetTdJob.__init__()`，从而自动构造初态并记录第一个数据点。

#### `init_mp(self, method)`（L420）

- 负责构造“单个局域初态”，再由 `_init_msmps()` 包装成 multiset。

零温时：

- 直接返回 `Mps.hartree_product_state(model=self.ms_model.init_model)`

有限温时支持三种方法：

1. `imaginary_time_exact`

- 对每个 `BasisSHO` 构造玻尔兹曼系数：
  - `exp(-beta * omega * n / 2)`
- 先做 thermal `Mps`
- 再转成 `MpDm.from_mps(...)`

2. `thermo_field`

- 使用 `_thermal_coefficients_from_theta()`
- 同样先做 thermal `Mps`
- 再转成 `MpDm`

3. `imaginary_time_propagate`

- 先构造 `MpDm.max_entangled_gs(init_model)`
- 指定一个固定 `bond_dim=1` 的 `icompress_config`
- 用 `ThermalProp(..., tdvp_ps)` 沿虚时间传播 `beta/2j`
- 得到有限温 `MpDm`

#### `_init_msmps(self, local_state)`（L472）

- 若 `local_state` 已经是 `MultisetMps`，直接返回。
- 否则把这个局域态复制到所有电子块中，构造成一个新的 `MultisetMps`。

#### `_fc_excitation(self, state, alpha)`（L484）

- 构造 Franck-Condon 型局域电子激发。
- 做法不是把其他 block 设成精确 0，而是：
  - 对所有 `beta != alpha` 的分量乘上 `1e-10`
- 然后重新做一次 `ms_normalize("mps_only")`。

#### `_set_hamiltonian_offset(self, energy)`（L490）

- 重建 `MsMpo` 中每个 block 的 `Mpo`，把对角块减去一个能量偏移。
- 规则：
  - 空 block 仍然是 `[]`
  - 对角块 `alpha == beta` 用 `offset=energy`
  - 非对角块用 `offset=0`
- 最后必须重新 `_refresh_mpo_cache()`。

#### `init_mps(self)`（L511）

- 初始化整条扩散作业的 multiset 初态。
- 具体流程：
  1. `state = self._init_msmps(self.init_mp())`
  2. 对 `initial_site` 做 `_fc_excitation`
  3. 把这个初态绑定到 `self.ms_model`
  4. 计算初始总能 `E0`
  5. 调 `_set_hamiltonian_offset(E0)`，把能量零点移到初态附近
  6. 调 `expand_bond_dimension_multiset(coef=1e-10, use_hint=self.use_init_hint)`
  7. 再做一次 `ms_normalize("mps_only")`

#### `process_mps(self, mps)`（L534）

- 把当前 multiset 态变成扩散观测量。
- 会记录：
  - `energy`
  - `rho = mps.rho_el()`
  - `e_occupations`
  - `ph_occupations`
  - `r_square`
  - `coherent_length = abs(rho).sum() - trace(rho).real`
  - `trace(rho).real`
  - `trace(rho @ rho).real`

#### `evolve_single_step(self, evolve_dt)`（L553）

- 单纯调用：
  - `new_mps = self.ms_model.evolve_state(self.latest_mps, evolve_dt)`

#### `stop_evolve_criteria(self)`（L558）

- 若 `stop_at_edge=True`，且左边界第一个站点占据超过 `edge_threshold`，则停止。
- 当前只检查左边界 `self.e_occupations_array[-1][0]`。

#### `get_dump_dict(self)`（L565）

- 返回最终保存到 `.npz` 的观测量字典：
  - `mol list`
  - `temperature`
  - `time series`
  - `energy array`
  - `r square array`
  - `electron occupations array`
  - `phonon occupations array`
  - `reduced density matrices`
  - `coherent length array`
  - `rho trace array`
  - `purity array`

## 7. `multisetspectra.py` 归纳

这个文件把 singleset 的零温/有限温光谱流程，扩展成了 multiset 版本的关联函数作业。

### 7.1 模块级辅助函数

#### `_state_inner_product(bra, ket)`（L19）

- 计算两个单状态的内积，并显式乘上 `coeff`：
  - `bra.conj().dot(ket) * conj(bra.coeff) * ket.coeff`

#### `_multiset_overlap(bra, ket)`（L27）

- 只对相同电子块求和：
  - `sum_alpha <phi_alpha^bra | phi_alpha^ket>`

#### `_multiset_dipole_overlap(bra, ket, dipole)`（L34）

- 做带 dipole 权重的双重求和：
  - `sum_{alpha,beta} mu_alpha mu_beta <phi_alpha^bra | phi_beta^ket>`
- 当前两个主类里并没有直接调用这个辅助函数，它更像是预留接口。

#### `_multiset_cross_overlap(bra, ket)`（L44）

- 对全部 `alpha, beta` 做双重求和。
- 在 multiset 发射关联函数里会用到。

#### `_scale_multiset_state(state, factor)`（L52）

- 把所有 block 分量统一乘上同一个实数因子。

### 7.2 `MultisetSpectraZeroT` 类

#### `__init__(...)`（L59）

- 只支持：
  - `spectratype == "abs"`
  - `spectratype == "emi"`
- 温度固定为 0 K。
- 内部创建一个 `MultisetModel(..., auto_init=False)`。
- 另外构造一个局域 ground-state `h_mpo = Mpo(self.ms_model.init_model, offset=self.offset)`。
- 维护两个结果缓存：
  - `_autocorr`
  - `_bond_dims`
- 最后调用 `MultisetTdJob.__init__()`，从而立刻生成初态并记录第一个时间点。

#### `init_mps(self)`（L102）

- 根据 `spectratype` 分派到 `init_mps_abs()` 或 `init_mps_emi()`。

#### `init_mps_abs(self)`（L107）

- 通过 `_init_dipole()` 构造初始 multiset `ket`
- 若 `expand=True`，则 `_expand_ket_bonddim(ket)`
- 返回 `(ket.copy(), ket)`，也就是固定 bra + 演化 ket

#### `init_mps_emi(self)`（L113）

- 当前实现与 `init_mps_abs()` 完全相同。
- 也就是说，零温发射与吸收的区别目前不在初态构造，而在后续关联函数的重叠定义。

#### `evolve_single_step(self, evolve_dt)`（L119）

- 若 `ket` 是 `MultisetMps`：
  - 调用 `self.ms_model.evolve_state(ket, evolve_dt, normalize=False)`
- 否则退回 singleset 的 `ket.evolve(...)`

#### `process_mps(self, mps)`（L127）

- 若 `bra` 是 multiset：
  - 对 `emi` 使用 `_multiset_cross_overlap(bra, ket)`
  - 对 `abs` 使用 `_multiset_overlap(bra, ket)`
- 若 `bra` 不是 multiset，则退回单态内积。
- 同时记录当前步的 bond 维信息。

#### `autocorr`（L139）

- 返回复数自相关数组。

#### `bond_dims`（L143）

- 返回 object dtype 数组，保存每步的 bond 维快照。

#### `get_dump_dict(self)`（L146）

- 保存字段：
  - `temperature`
  - `time series`
  - `time_series`
  - `autocorr`
  - `bond_dims`

#### `init_mp(self)`（L155）

- 构造一个 Hartree product ground state。
- 并把 `compress_config` 绑定到它上面。

#### `_get_dipole(self)`（L160）

- 从 `model.dipole` 提取每个电子自由度对应的 dipole 权重。
- 支持三种输入风格：
  - `dict`
  - 标量
  - 向量/数组
- 最终一定会整理成长度为 `model.n_edofs` 的一维实数组。

#### `_init_dipole(self)`（L186）

- 用 `_get_dipole()` 得到权重后：
  - 对每个 `alpha`
  - 从 ground-state `phi_g` 复制一个局域态
  - 乘上 `dipole_weights[alpha]`
- 最终包装成一个 `MultisetMps`

#### `_expand_ket_bonddim(self, ket, coef, use_hint)`（L208）

- 对初始 dipole 激发态做 multiset bond expansion。
- 为了避免 expansion 改变初始总范数：
  - 先算 `initial_norm`
  - expansion 后算 `expanded_norm`
  - 若两者不同，则整体 rescale 到原范数

### 7.3 `MultisetSpectraFiniteT` 类

#### `__init__(...)`（L226）

- 只支持有限温：
  - `temperature != 0`
- 热初态只支持：
  - `imaginary_time_exact`
  - `imaginary_time_propagate`
- 内部创建：
  - `self.ms_model = MultisetModel(...)`
  - `self.h_mpo`：带 offset 的局域基哈密顿量
  - `self.h_mpo_gs`：offset 为 0 的局域 ground Hamiltonian
  - `self.icompress_config`
  - `self.local_evolve_config = tdvp_ps`
  - `self.ievolve_config`：默认为 multiset TDVP-PS
- 如果 `spectratype == "abs"`，作业层配置用 `self.ms_model.evolve_config`
- 如果 `spectratype == "emi"`，作业层配置用 `self.local_evolve_config`

#### `init_mps(self)`（L289）

- 按 `spectratype` 分发到：
  - `init_mps_abs()`
  - `init_mps_emi()`

#### `init_mps_abs(self)`（L295）

- 流程：
  1. `thermal_mpdm = _init_ground_thermal_state()`
  2. `ket = _init_dipole(thermal_mpdm)`
  3. 若 `expand=True`，则 `_expand_initial_multiset_state(ket)`
  4. `_set_multiset_hamiltonian_offset(self.offset)`
  5. 返回 `(ket.copy(), ket)`

#### `init_mps_emi(self)`（L303）

- 流程：
  1. `thermal_state = _init_excited_thermal_state()`
  2. `ket = _init_dipole(thermal_state)`
  3. 若 `expand=True`，则 `_expand_initial_multiset_state(ket)`
  4. 把 `ket` 绑定到 `self.ms_model`
  5. 计算 excited-state carrier energy
  6. `_set_multiset_hamiltonian_offset(excited_energy + self.offset)`
  7. 返回 `(ket.copy(), ket)`

#### `_init_ground_thermal_state(self)`（L318）

- 若 `thermal_init_method == "imaginary_time_exact"`：
  - 调 `_exact_ground_thermal_mpdm()`
- 若 `thermal_init_method == "imaginary_time_propagate"`：
  - 调 `_propagate_ground_thermal_mpdm()`

#### `_exact_ground_thermal_mpdm(self, model)`（L324）

- 在局域 ground 模型上逐个 SHO 模式构造精确玻尔兹曼权重：
  - `exp(-beta * omega * n / 2)`
- 先生成 thermal `Mps`
- 再 `MpDm.from_mps(thermal_mps)`

#### `_propagate_ground_thermal_mpdm(self, model)`（L344）

- 先构造 `MpDm.max_entangled_gs(model)`
- 然后用 singleset `ThermalProp(..., tdvp_ps)` 沿虚时间传播 `beta/2j`
- 适用于吸收谱里“只有 ground branch 需要热化”的情况

#### `_propagate_excited_thermal_msmpdm(self, state)`（L364）

- 对一个 multiset excited-state MPDM 做虚时间传播。
- 每一步调用：
  - `self.ms_model.evolve_state(state, beta / (2j * insteps), normalize=True)`
- 传播期间临时把 `self.ms_model.evolve_config` 切换成 `self.ievolve_config`
- 用 `try/finally` 确保结束后恢复原配置

#### `_max_entangled_ground_mpdm(self, model, set_evolve_config)`（L382）

- 返回 `MpDm.max_entangled_gs(model)`，并绑定压缩/演化配置。

#### `_init_excited_thermal_state(self)`（L389）

- 这是有限温发射最关键的初态准备步骤。
- 当前逻辑：
  - 对每个 `alpha`，先构造该对角块 `MsModel[alpha][alpha]` 的 max-entangled ground MPDM
  - 再把这些对角块收集成一个 multiset state
  - 最后用 `_propagate_excited_thermal_msmpdm()` 做 multiset 虚时间传播
- 如果用户指定 `imaginary_time_exact`：
  - 代码会明确 `warning`
  - 因为“物理上精确的 excited diagonal thermal initialisation”当前没有直接实现
  - 所以会自动回退到 propagate 路径

#### `_build_multiset_state(self, msmps)`（L409）

- 把一组局域 `Mps/MpDm` 包装成 `MultisetMps`。
- 同时给每个分量绑定：
  - `compress_config`
  - `evolve_config`

#### `_broadcast_local_state(self, local_state, weights)`（L422）

- 把单个局域态复制到所有电子块。
- 可附带不同的缩放权重 `weights`。

#### `_init_dipole(self, state)`（L436）

- 如果传入的是普通局域态：
  - 先 `_broadcast_local_state(state, dipole)`
- 如果传入的是 `MultisetMps`：
  - 对每个已有分量再乘上对应 `dipole[alpha]`

#### `_expand_initial_multiset_state(self, state, coef)`（L450）

- 对有限温光谱的初始 multiset 态做 hint-based bond expansion。
- expansion 后重新给每个分量恢复：
  - `compress_config`
  - `evolve_config`

#### `_set_multiset_hamiltonian_offset(self, energy)`（L459）

- 与 `MultisetChargeDiffusionDynamics` 中的同名函数逻辑相同：
  - 对角块减去统一 offset
  - 非对角块 offset 设为 0
  - 最后刷新 active pair / site template 缓存

#### `evolve_single_step(self, evolve_dt)`（L480）

- `bra, ket = self.latest_mps`
- 若当前步号是奇数：
  - 传播 `ket`
- 若当前步号是偶数：
  - 传播 `bra`，但时间符号取反
- 这是有限温关联函数的双支路演化结构。

#### `_evolve_finite_temperature_branch(self, state, evolve_dt)`（L488）

- 如果 `state` 是 `MultisetMps`：
  1. 先做 `_evolve_ground_multiset(state, -evolve_dt)`
  2. 再做 `self.ms_model.evolve_state(state, evolve_dt, normalize=False)`
- 如果 `state` 只是局域 `MpDm`：
  - 只做 `_evolve_ground_mpdm(state, evolve_dt)`

#### `_evolve_ground_multiset(self, state)`（L495）

- 对 multiset 里每个 `mpdm` 分量分别调用 `_evolve_ground_mpdm()`。

#### `_evolve_ground_mpdm(self, mpdm, evolve_dt)`（L508）

- 先确保 `qn` 是 `numpy array`
- 再构造一个精确局域传播子 `mpo_prop = _exact_local_ground_propagator(-1j * evolve_dt)`
- 最后 `mpdm.apply(mpo_prop, canonicalise=True)`

#### `_exact_local_ground_propagator(self, x)`（L517）

- 为 `init_model` 上的每个 `BasisSHO` 构造一个对角传播子：
  - `diag = exp(x * omega * n)`
- 整个 MPO 只支持 `BasisSHO` 模型。
- 若遇到其他 basis 类型，直接报错。

#### `_ensure_array_qn(mpdm)`（L538）

- 静态工具函数。
- 把 `mpdm.qn` 列表里的元素逐个转成 `np.asarray(qn)`。

#### `process_mps(self, mps)`（L542）

- 对有限温光谱记录关联函数值。
- 若 `bra` 是 multiset：
  - `emi` 用 `_multiset_cross_overlap`
  - `abs` 用 `_multiset_overlap`
- 若 `bra` 不是 multiset：
  - 用 `_state_inner_product`
- 对 `emi`，最后还会整体取复共轭。

#### `stop_evolve_criteria(self)`（L556）

- 固定返回 `False`。
- 光谱作业默认不会提前停止。

#### `autocorr`（L560）

- 返回关联函数时间序列。

#### `bond_dims`（L564）

- 返回每步的 bond 维记录。

#### `get_dump_dict(self)`（L567）

- 保存字段：
  - `temperature`
  - `time series`
  - `time_series`
  - `autocorr`
  - `bond_dims`

#### `_get_dipole(self)`（L576）

- 与 `MultisetSpectraZeroT` 的 `_get_dipole()` 同逻辑。

## 8. `test/` 中 multiset 体系与已获得结果

这一节只整理 `test/` 目录里已经存在的 multiset 脚本与数据。  
我把“脚本层面已经实现了什么”和“仓库里已经留存了什么结果文件”分开写。

### 8.1 总体观察

- 当前 `test/` 下 multiset 结果主要分两类：
  - 动力学人口分布轨迹
  - 光谱自相关函数 `autocorr`
- 动力学数据主要以两种形式存在：
  - 原始 `npz`：包含能量、人口、声子占据、密度矩阵等完整观测量
  - 后处理 `xlsx`：通常只保留每个时间点的电子人口矩阵
- 从所有已检查的 `xlsx` 与动力学 `npz` 看：
  - 每一行电子人口和都基本保持在 1 附近
  - 说明 norm / 总人口在结果文件里整体守恒良好

### 8.2 已实现的 multiset 测试脚本清单

| 脚本 | 体系 | 主要用途 |
| --- | --- | --- |
| `test/test_fmo_0k/multiset/fmo.py` | 7-site FMO, 0 K | multiset 电荷扩散 |
| `test/test_fmo_300k/multiset/fmo.py` | 7-site FMO, 300 K | multiset 有限温电荷扩散 |
| `test/test_2DHolstein25/multiset/2DHolstein25.py` | 5x5 周期二维 Holstein | multiset 电荷扩散 |
| `test/test_Holstein_5/multiset/Holstein5_16bd.py` | 5-site 一维 Holstein | multiset 电荷扩散 |
| `test/test_Holstein_75/multiset/Holstein75_multiset.py` | 75-site 周期一维 Holstein | multiset 电荷扩散 |
| `test/test_Holstein_75/plot_2026_4/multiset/Holstein75_multiset.py` | 75-site 周期一维 Holstein | 带环境变量控制和 startup substeps 的 rerun/敏感性测试 |
| `test/test_spetra_zt_pbi/multiset/PBI.py` | PBI 聚集体零温谱 | multiset 零温吸收/发射关联函数 |
| `test/test_spetra_ft_pbi/multiset/PBI.py` | PBI 聚集体有限温谱 | multiset 有限温吸收/发射关联函数 |

### 8.3 FMO 0 K：`test/test_fmo_0k/multiset/fmo.py`

#### 脚本内容

- 模型：
  - 7-site FMO
  - 使用 `example/fmo_sdf.json` 生成 35 个声子模式
  - 总 Huang-Rhys 因子 `TOTAL_HR = 0.42`
  - 使用固定排列 `mol_arangement = [7, 5, 3, 1, 2, 4, 6] - 1`
- 动力学设置：
  - `max_bonddim = 64`
  - `evolve_dt = 160`
  - `n_snapshots = 250`
  - `stop_at_edge = False`
- 结果导出形式：
  - 脚本本身把 `e_occupations_array` 写成 `xlsx`

#### 仓库中已提交结果

这一组结果主要存放在：

- `test/test_fmo_0k/plot_2026_3/`
- `test/test_fmo_0k/plot_2026_4/`

具体文件与摘要如下：

| 文件 | 形状 | 末态最大人口位置 | 末态最大人口值 | 备注 |
| --- | --- | --- | --- | --- |
| `multiset_32bd.xlsx` | `(250, 7)` | 2 | `0.6371901688` | 标准 32bd 结果 |
| `multiset_64bd.xlsx` | `(250, 7)` | 2 | `0.6482852917` | 标准 64bd 结果 |
| `multiset_128bd.xlsx` | `(250, 7)` | 2 | `0.6509844369` | 更高 bond 维 |
| `multiset_32bd_old.xlsx` | `(250, 7)` | 2 | `0.6372151202` | 老版本/旧跑点 |
| `multiset_64bd_old.xlsx` | `(250, 7)` | 2 | `0.6481591599` | 老版本/旧跑点 |
| `multiset_32bd_speedup.xlsx` | `(250, 7)` | 2 | `0.6370755684` | speedup 版本 |
| `multiset_128bd_speedup.xlsx` | `(250, 7)` | 2 | `0.6509758659` | speedup 版本 |
| `multiset_32bd_coef1.xlsx` | `(250, 7)` | 2 | `0.6194754302` | 改变 expansion 系数后的对照 |
| `multiset_64bd_80dt.xlsx` | `(250, 7)` | 2 | `0.6482893883` | 细时间步对照 |
| `multiset_64bd_320dt.xlsx` | `(125, 7)` | 2 | `0.6464011635` | 更粗时间步 |
| `multiset_64bd_640dt.xlsx` | `(63, 7)` | 2 | `0.6463410010` | 更粗时间步 |
| `multiset_64bd_1280dt.xlsx` | `(32, 7)` | 2 | `0.6462117664` | 最粗时间步 |
| `plot_2026_4/multiset_64bd.xlsx` | `(250, 7)` | 2 | `0.6482975202` | 后续 rerun |

#### 这一组结果说明了什么

- 所有表格第一行和最后一行人口和都约等于 1，说明电子人口守恒是稳定的。
- `64bd` 和 `128bd` 的末态主峰都在第 2 个站点，且主峰值分别约为 `0.6483` 和 `0.6510`，说明在这个例子上随 bond 维增加结果趋于收敛。
- `32bd_coef1.xlsx` 与标准 `32bd` 的末态差异较明显，说明初始 bond expansion 系数会实质影响轨迹。
- `64bd_80dt/320dt/640dt/1280dt` 这组时间步敏感性文件的末态主峰都在站点 2，主峰值从 `0.6483` 变化到 `0.6462`，差异不大，说明该例子对时间步有一定稳健性。

### 8.4 FMO 300 K：`test/test_fmo_300k/multiset/fmo.py`

#### 脚本内容

- 模型仍是 7-site FMO + 每站点 35 个声子模式。
- 温度：
  - `Quantity(300, "K")`
- 初态方法：
  - `method="imaginary_time_propagate"`
- 脚本中当前参数：
  - `max_bonddim = 16`
  - `evolve_dt = 160`
  - `evolve_time = 40000`
  - `stop_at_edge = False`
  - `dump_dir="./"`
  - `job_name="fmo_300K_{max_bonddim}bd"`

#### 仓库中已提交结果

| 文件 | 时间点数 | 电子人口形状 | 声子人口形状 | 末态主峰位置 | 末态主峰值 | `r^2` 末值 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fmo_300K_16bd.npz` | 252 | `(252, 7)` | `(252, 245)` | 4 | `0.3706655943` | `1.0174137237` | 旧 schema，额外包含 `electronic ancilla`、`thermal max bonddim` |
| `fmo_300K_64bd.npz` | 252 | `(252, 7)` | `(252, 245)` | 4 | `0.3064308591` | `1.8618390814` | 64bd 跑点之一 |
| `fmo_300k_64bd.npz` | 252 | `(252, 7)` | `(252, 245)` | 4 | `0.3089965814` | `1.8307890132` | 命名不同，但不是同一数值文件 |

补充数值：

- 三个文件的时间轴都是：
  - `t0 = 0`
  - `t_last = 40160`
- 三个文件的温度都是：
  - `0.0009500434690 a.u.`，即约 300 K
- 能量首末值：
  - `16bd`: `-1.7747555e-05 -> -1.7649884e-05`
  - `64bd`: `-1.7747822e-05 -> -1.7248783e-05`
  - `64bd(lowercase k)`: `-1.7707488e-05 -> -1.7161667e-05`

#### 这一组结果说明了什么

- 与 0 K 的纯人口表不同，这里保留了完整的 `.npz` 原始观测量：
  - 能量
  - `r_square`
  - 电子人口
  - 声子人口
  - 约化密度矩阵
  - coherence / purity / trace
- `phonon_occ_shape = (252, 245)` 正好对应 `7 * 35 = 245` 个声子自由度。
- 末态主峰都在站点 4，但更高 bond 维的 64bd 结果扩散更强：
  - `16bd` 的 `r^2` 末值约 `1.02`
  - `64bd` 的 `r^2` 末值约 `1.83 ~ 1.86`
- `fmo_300K_64bd.npz` 与 `fmo_300k_64bd.npz` 虽然名字接近，但末态能量和 `r^2` 不完全相同，说明它们应是两次独立运行而不是单纯重命名。

### 8.5 二维 Holstein 25 站点：`test/test_2DHolstein25/multiset/2DHolstein25.py`

#### 脚本内容

- 体系：
  - `5 x 5` 周期二维格点
  - 总站点数 `25`
  - 默认初始站点会落在中心 `index = 12`
- 参数：
  - `G = 0.5`
  - `W0 = 1.0`
  - `J = 0.1`
  - `NBOSE = 8`
  - `TMAX = 40`
  - `DT = 0.2`
- 当前脚本中 `max_bonddim = 4`
- 最终输出是电子人口随时间的 `xlsx`

#### 仓库中已提交结果

| 文件 | 形状 | 末态最大人口位置 | 末态最大人口值 |
| --- | --- | --- | --- |
| `2DHolstein25_chi4_multiset.xlsx` | `(201, 25)` | 12 | `0.2182610129` |
| `2DHolstein25_chi8_multiset.xlsx` | `(201, 25)` | 12 | `0.2176012664` |
| `2DHolstein25_chi16_multiset.xlsx` | `(201, 25)` | 12 | `0.2175849769` |
| `2DHolstein25_chi32_multiset.xlsx` | `(201, 25)` | 12 | `0.2175848160` |

#### 这一组结果说明了什么

- 四个结果文件每步人口和都约等于 1。
- 到末态时，最大人口始终还在中心站点 `12`，说明在当前参数和总时间下，激发虽然已经扩散，但仍未离开中心主峰。
- `chi16` 与 `chi32` 的末态峰值已经几乎完全一致，说明这个算例在末态主峰上对 bond 维已经接近收敛。

### 8.6 Holstein 5 站点：`test/test_Holstein_5/multiset/Holstein5_16bd.py`

#### 脚本内容

- 一维 5 站点 Holstein 链。
- 参数：
  - `omega_0 = 1.0`
  - `J = 1.0`
  - `g = 1`
  - `n_phys_dim = 4`
  - 开边界 `periodic=False`
- 当前脚本参数：
  - `max_bonddim = 16`
  - `evolve_dt = 0.1`
  - `n_snapshots = 100`

#### 仓库中已提交结果

| 文件 | 形状 | 末态最大人口位置 | 末态最大人口值 | 说明 |
| --- | --- | --- | --- | --- |
| `Holstein5_16bd_multiset.xlsx` | `(100, 5)` | 2 | `0.3079863295` | 与脚本参数一致 |
| `Holstein5_64bd_multiset.xlsx` | `(500, 5)` | 1 | `0.2402825669` | 更长时间/更高 bond 维的结果文件 |

#### 这一组结果说明了什么

- 两个文件都保持人口和约等于 1。
- `16bd` 文件只有 100 个时间点，末态主峰还在中心站点 2。
- `64bd` 文件有 500 个时间点，末态主峰转移到站点 1，说明更长传播时间下扩散已经显著偏离初始中心。

### 8.7 Holstein 75 站点：`test/test_Holstein_75/multiset/Holstein75_multiset.py`

#### 脚本内容

- 一维 75 站点周期 Holstein 链。
- 参数：
  - `omega_0 = 1.0`
  - `J = 1.0`
  - `g = 1.5`
  - `nu_max = 16`
  - `phonon_dim = 17`
  - `periodic=True`
- `test/test_Holstein_75/multiset/Holstein75_multiset.py` 是基础版本。
- `test/test_Holstein_75/plot_2026_4/multiset/Holstein75_multiset.py` 是更灵活版本，额外支持环境变量：
  - `MAX_BONDDIM`
  - `EVOLVE_DT`
  - `N_SNAPSHOTS`
  - `IF_STARTUP_SUBSTEPS`
  - `STARTUP_SUBSTEPS_N`
  - `OUTPUT_XLSX`

#### 仓库中已提交结果：2026_3 数据

| 文件 | 形状 | 末态最大人口位置 | 末态最大人口值 |
| --- | --- | --- | --- |
| `plot_2026_3/data/Holstein75_8bd_multiset.xlsx` | `(500, 75)` | 39 | `0.0436798216` |
| `plot_2026_3/data/Holstein75_16bd_multiset.xlsx` | `(500, 75)` | 50 | `0.0489426361` |
| `plot_2026_3/data/Holstein75_32bd_multiset.xlsx` | `(500, 75)` | 37 | `0.0365813567` |
| `plot_2026_3/data/Holstein75_64bd_multiset.xlsx` | `(500, 75)` | 51 | `0.0488600826` |

#### 仓库中已提交结果：2026_4 数据

| 文件 | 形状 | 末态最大人口位置 | 末态最大人口值 | 备注 |
| --- | --- | --- | --- | --- |
| `Holstein75_8bd_multiset.xlsx` | `(500, 75)` | 43 | `0.0732242150` | rerun 版本 |
| `Holstein75_16bd_multiset.xlsx` | `(500, 75)` | 37 | `0.0414466021` | rerun 版本 |
| `Holstein75_32bd_multiset.xlsx` | `(500, 75)` | 29 | `0.0536370613` | rerun 版本 |
| `Holstein75_64bd_multiset.xlsx` | `(500, 75)` | 37 | `0.0364563483` | rerun 版本 |
| `Holstein75_8bd_multiset_substep.xlsx` | `(500, 75)` | 39 | `0.0438766972` | startup substeps 对照 |
| `Holstein75_24bd_multiset_substep.xlsx` | `(500, 75)` | 37 | `0.0375903538` | startup substeps 对照 |
| `Holstein75_32bd_multiset_substep.xlsx` | `(500, 75)` | 39 | `0.0766398819` | startup substeps 对照 |

#### 这一组结果说明了什么

- 所有结果文件人口和都维持在 1 附近。
- 这一组算例明显是在做：
  - bond 维敏感性测试
  - startup substeps 对结果的影响测试
- 从末态主峰位置看，不同 bond 维和不同子步设置的结果差异仍然比较大，说明这个 75-site 长链例子在晚时间行为上仍比较敏感，当前文件更像是“扫描与比较数据集”，而不是已经完全收敛到唯一结果的一组数据。

### 8.8 PBI 零温谱：`test/test_spetra_zt_pbi/multiset/PBI.py`

#### 脚本内容

- 模型构造函数 `construct_model(nmols)` 支持：
  - `monomer`
  - `dimer`
  - `hexmer`
- 每个分子含 10 个振动模式。
- 局域激发能设为：
  - `2.13 eV`
- dipole 取 `1.0`
- 电子耦合通过：
  - `HolsteinModel([...], Quantity(-500, "cm-1"))`
- 当前脚本默认：
  - `type_ = "dimer"`
  - `spectratype = "emi"`
  - `spectra_tag = "zt"`
  - `max_bonddim = 4`
  - `expand=True`
  - `evolve_dt = 20`
  - `nsteps = 5000`

#### 仓库中已提交结果

| 文件 | 时间点数 | 时间上限 | `autocorr[0]` | `autocorr[-1]` | bond 维记录 |
| --- | --- | --- | --- | --- | --- |
| `pbi_dimer_zt_abs_multiset.npz` | 5001 | 100000 | `2.0000000004 + 0j` | `-0.3898011406 + 1.0996710886j` | 有 |
| `pbi_dimer_zt_emi_multiset.npz` | 5001 | 100000 | `4.0000000008 + 0j` | `-0.7818651791 + 2.2677827518j` | 有 |

#### 这一组结果说明了什么

- 结果文件是完整的 `.npz`，保存了：
  - `temperature`
  - `time series`
  - `autocorr`
  - `bond_dims`
- 在 unit dipole 设定下：
  - dimer 吸收的 `t=0` 自相关约为 `2`
  - dimer 发射的 `t=0` 自相关约为 `4`
- 这和当前代码里：
  - 吸收使用 `_multiset_overlap`
  - 发射使用 `_multiset_cross_overlap`
 之间的定义差别是吻合的。

### 8.9 PBI 有限温谱：`test/test_spetra_ft_pbi/multiset/PBI.py`

#### 脚本内容

- 模型与零温 PBI 谱相同，也支持 `monomer/dimer/hexmer`。
- 当前脚本默认：
  - `type_ = "dimer"`
  - `spectratype = "emi"`
  - `spectra_tag = "ft"`
  - `temperature = 298 K`
  - `max_bonddim = 32`
  - `thermal_init_method = "imaginary_time_exact"`
  - `insteps = 50`
  - `expand=True`
  - `evolve_dt = 20`
  - `nsteps = 5000`

#### 仓库中已提交结果

| 文件 | 主要字段 | `autocorr[0]` | `autocorr[-1]` | 备注 |
| --- | --- | --- | --- | --- |
| `pbi_dimer_ft_abs_multiset.npz` | `time_series, autocorr` | `2.0000000016 + 0j` | `0.3435666382 + 0.0187859446j` | 精简 schema |
| `pbi_dimer_ft_emi_multiset.npz` | `temperature, time series, time_series, autocorr, bond_dims` | `1.7781810239 - 0j` | `0.3547523558 + 0.0665266505j` | 完整 schema |
| `pbi_hexmer_ft_abs_multiset.npz` | `time_series, autocorr` | `6.0000000040 + 0j` | `-0.2505528523 - 0.0324765866j` | 精简 schema |
| `pbi_hexmer_ft_emi_multiset.npz` | `time_series, autocorr` | `36.0000000240 - 0j` | `-0.8726550126 + 0.1620623209j` | 精简 schema |

补充说明：

- `pbi_dimer_ft_emi_multiset.npz` 明确记录了温度：
  - `0.0009437098459 a.u.`，即约 298 K
- 其它几个有限温结果文件虽然没有存 `temperature` 字段，但从脚本参数看，它们也是 298 K 的 multiset 光谱结果。

#### 这一组结果说明了什么

- 当前仓库里已经有：
  - dimer 有限温吸收
  - dimer 有限温发射
  - hexmer 有限温吸收
  - hexmer 有限温发射
- `autocorr[0]` 的数量级与分子数、以及当前使用的 overlap 定义一致：
  - dimer 吸收约 `2`
  - hexmer 吸收约 `6`
  - hexmer 发射约 `36`
- 结果文件 schema 并不完全统一：
  - 有些是完整 schema
  - 有些只保留了 `time_series + autocorr`
- 这说明这批光谱结果可能来自不同阶段的保存逻辑。

## 9. 可以直接得出的整理结论

### 9.1 当前 `renormalizer/multiset` 已经覆盖的功能

- multiset block Hamiltonian 拆分
- multiset MPS / MPDM 状态容器
- multiset block MPO 与通用 block 算符
- multiset TDVP-PS 演化
- multiset 电荷扩散动力学
- multiset 零温光谱
- multiset 有限温光谱
- 有限温初态构造：
  - exact thermal product
  - thermofield
  - imaginary-time propagation

### 9.2 `test/` 中已经有结果留档的体系

- FMO 0 K
- FMO 300 K
- 2D Holstein 25-site
- Holstein 5-site
- Holstein 75-site
- PBI zero-T dimer spectra
- PBI finite-T dimer / hexmer spectra

### 9.3 数据保存形式上的现状

- 动力学：
  - 原始 `.npz` 主要见于 `FMO 300K`
  - 其它很多体系主要保留了后处理的 `.xlsx` 人口轨迹
- 光谱：
  - 主要保留 `.npz`
  - 但不同文件的字段 schema 有新旧差别

### 9.4 目前代码层最值得记住的三个实现点

- `MultisetModel` 不是简单地“多放几个 MPS”，而是显式构造了 `H^{alpha,beta}` 的 block 结构，并且用 active pair + site template 做批量收缩。
- `MultisetChargeDiffusionDynamics` 和 `MultisetSpectraFiniteT` 都依赖“局域基上的单态热初态 + multiset 包装/传播”的两层流程，这一点和 singleset 的思想一致，但已经扩展到多电子块。
- 光谱里的吸收/发射区别，目前主要体现在 overlap 定义与有限温初态构造方式，而不是简单切换一个参数。
