# CBE-TDVP 当前实现总结

本文档总结当前 Renormalizer 中 CBE-TDVP 的实现，重点覆盖
`renormalizer/mps/mps.py` 与 `renormalizer/mps/cbe.py` 中新增的 CBE 相关类、
函数、流程、量子数处理方式，以及目前在 FMO 压力测试中遇到的问题。

## 1. 算法流程

当前实现目标是保留原有 1TDVP projector-splitting 流程，同时在
`expansion_method="cbe"` 时启用 CBE-TDVP warm-up：

1. 总演化仍由外层 dynamics 调用 `Mps.evolve(mpo, evolve_dt)`。
2. 当 `method == EvolveMethod.tdvp_ps` 且 `expansion_method == "cbe"` 时，
   `Mps.evolve()` 不走原 `_evolve_tdvp_ps()`，而走 `_evolve_cbe_tdvp_ps()`。
3. `_evolve_cbe_tdvp_ps()` 先判断当前是否处于 CBE warm-up：
   - `expansion_method == "cbe"`；
   - `cbe_runtime_disabled == False`；
   - 若 `cbe_disable_after_warmup=True`，则要求 `current_time < cbe_warmup_time`。
4. CBE active 时，每个 TDVP one-site local update 前做局部 expansion：
   - right-to-left sweep：对 bond `imps-1` 做 `A_tr` selection 和左侧 bond expansion；
   - left-to-right sweep：对 bond `imps` 做 `B_tr` selection 和右侧 bond expansion。
5. expansion 不改变当前波函数：
   - right-to-left：`A_ex = [A_l, A_tr]`，`C_ex = [C_{l+1}; 0]`；
   - left-to-right：`B_ex = [B_r; B_tr]`，`C_ex = [C_l, 0]`。
6. 扩展后立即用现有 one-site effective Hamiltonian 做真实时间 TDVP local evolution。
7. local evolution 后，在扩展空间中做 SVD trim：
   - 截断阈值为 `cbe_eps_trim`；
   - 最大维度为 `cbe_Dmax`；
   - 尽量保留本次新增 qn sector，避免刚扩展的新方向在获得权重前被删掉。
8. warm-up 结束后由外层 FMO 脚本设置：
   - `cbe_runtime_disabled=True`；
   - 可选锁定 `compress_config.max_dims = cbe_Dmax`；
   - 后续生产阶段退回普通 fixed-rank 1TDVP。

整体流程是 sweep 内局部 CBE，而不是 t=0 的一次性全局预扩展。

## 2. `renormalizer/mps/cbe.py` 内容

`cbe.py` 是局部 CBE selection 与 local expansion 模块。它不驱动时间演化，
也不直接修改完整 MPS；调用方负责把返回结果写回 MPS 和 environment。

### 数据类

`CBESelectionResult`

- `tensor`：选择出的补空间张量，right-to-left 时是 `A_tr`，left-to-right 时是 `B_tr`。
- `singular_values`：最终 selection 保留方向对应的奇异值。
- `D_expand`：新增 bond dimension 数量。
- `debug_info`：selection 诊断信息，包括原因、预筛奇异值、最终奇异值、正交误差、选中 qn。
- `qn`：新增虚拟 bond sector 的量子数数组。

`CBEExpansionResult`

- `A_ex` / `B_ex`：扩展后的 isometry 张量。
- `C_ex`：扩展后的中心张量，新增 sector 用 0 padding。
- `L_ex` / `R_ex`：若传入 environment 和 MPO tensor，则返回扩展后的 environment。
- `orthogonality_error`：`A^dagger A_tr` 或 `B_tr B^dagger` 的误差。
- `isometry_error`：扩展后 `A_ex` 或 `B_ex` 的 isometry 误差。
- `wavefunction_error`：检查 `A_ex C_ex == A C` 或 `C_ex B_ex == C B`。
- `debug_info`：扩展维度等辅助信息。

### 工具函数

`_array(tensor)`

- 将 Renormalizer `Matrix` 或普通 tensor 转为 NumPy 数组。
- CBE selection 当前主要在 CPU NumPy/SciPy 上执行。

`_select_count(singular_values, eps, max_expand=None, eps_trim=1e-12)`

- 根据相对阈值 `eps`、绝对阈值 `eps_trim` 和 `max_expand` 决定保留多少个奇异方向。

`_orthonormal_columns(mat, eps_trim=1e-12)`

- 对列向量做 QR 正交化，丢掉近零 R 对角元。

`_orthonormal_rows(mat, eps_trim=1e-12)`

- 对行向量做 QR 正交化，本质是对 `mat.conj().T` 做 `_orthonormal_columns()`。

`_hpsi_qn_block(A_l, B_r, W_l, W_r, l_array, r_array, lidx, ridx)`

- 当前 M3000 优化后的关键函数。
- 不再构造完整 two-site `theta`，也不再做 full `theta -> hop(theta)`。
- 只对一个合法 qn block 的行列索引构造 `H|psi>` 小块。
- 计算方式为 factorized contraction：
  - 左侧投影：`L_{l-1} W_l A_l -> left_proj`；
  - 右侧投影：`W_{l+1} R_{l+2} B_{l+1} -> right_proj`；
  - 小块：`block = left_mat @ right_mat.T`。
- 它等价于旧 full two-site `Htheta[a,d,g,l]` 的对应 qn block，但避免了大 tensor 的 GPU OOM。

`_max_new_dim(D_before, cbe_Dmax=None, cbe_max_expand=None)`

- 计算当前 bond 最多还允许新增多少维度。
- 同时受 `cbe_Dmax - D_before` 和 `cbe_max_expand` 限制。

`_qn_key(qn)` / `_qn_mask(qn_array, qn)`

- 将 qn 转为 tuple key，并生成 qn mask。
- 用于按 qn block 切分 selection。

`_candidate_bond_qns(mps, q_left, q_right)`

- 枚举合法的目标 virtual bond qn。
- 条件是左侧可产生 qn，右侧可产生 `mps.qntot - qn`。
- 这保证新增 sector 与总量子数守恒兼容。

`_select_qn_block_columns(...)`

- right-to-left selection 中，把 qn block 内的左奇异向量嵌回完整左空间。
- 再投影掉已有 `A_l` basis 和已接受的同 qn 新方向。
- 返回归一化后的完整左补空间向量。

`_select_qn_block_rows(...)`

- left-to-right selection 中，把 qn block 内的右奇异向量嵌回完整右空间。
- 再投影掉已有 `B_r` basis 和已接受的同 qn 新方向。
- 返回归一化后的完整右补空间向量。

### Shrewd selection

`cbe_shrewd_selection_right_to_left(...)`

- 输入当前 MPS、MPO、bond index、左/右 environment。
- 构造：
  - `q_left = qn[bond_idx] + sigmaqn[bond_idx]`；
  - `q_right = sigmaqn[bond_idx+1] + qn[bond_idx+2]`；
  - 当前 bond qn 为 `mps.qn[bond_idx+1]`。
- 对每个合法目标 qn：
  1. 取对应 `lidx/ridx`；
  2. 用 `_hpsi_qn_block()` 构造该 qn block 的 `H|psi>`；
  3. 从 block 中投影掉已有 `A_l` 和 `B_r` 的当前 bond basis；
  4. 做 `scipy.linalg.svd(block, full_matrices=False)`；
  5. 根据 `cbe_eps_pre` 和 `cbe_eps_final` 两层阈值选择方向；
  6. 用 `_select_qn_block_columns()` 得到完整空间的 `A_tr` 列向量。
- 返回 `A_tr`、奇异值、`D_expand`、新增 qn。

`cbe_shrewd_selection_left_to_right(...)`

- 与 right-to-left 镜像。
- 对每个合法 qn block 做 SVD，使用右奇异向量构造 `B_tr`。
- 返回 `B_tr`、奇异值、`D_expand`、新增 qn。

### Local expansion

`cbe_expand_right_to_left(A_l, C_right, A_tr, l_array=None, W_l=None, bond_idx=None)`

- 若 `D_expand == 0`，直接返回原 tensor copy。
- 否则：
  - `A_ex = concat(A_l, A_tr, axis=2)`；
  - `C_ex = concat(C_right, zeros, axis=0)`。
- 检查：
  - `A_l^dagger A_tr`；
  - `A_ex^dagger A_ex - I`；
  - `A_ex C_ex - A_l C_right`。
- 若提供 `l_array` 和 `W_l`，用 `contract_one_site(..., "L")` 构造 `L_ex`。

`cbe_expand_left_to_right(C_left, B_right, B_tr, r_array=None, W_right=None, bond_idx=None)`

- 若 `D_expand == 0`，直接返回原 tensor copy。
- 否则：
  - `B_ex = concat(B_right, B_tr, axis=0)`；
  - `C_ex = concat(C_left, zeros, axis=2)`。
- 检查：
  - `B_tr B_right^dagger`；
  - `B_ex B_ex^dagger - I`；
  - `C_ex B_ex - C_left B_right`。
- 若提供 `r_array` 和 `W_right`，用 `contract_one_site(..., "R")` 构造 `R_ex`。

## 3. `renormalizer/mps/mps.py` 中的 CBE 内容

### 配置判断和 qn path seed

`_cbe_should_be_active(config)`

- 决定当前 step 是否启用 CBE。
- 逻辑：
  - `expansion_method != "cbe"`：False；
  - `cbe_runtime_disabled=True`：False；
  - `cbe_disable_after_warmup=True` 时要求 `current_time < cbe_warmup_time`；
  - 否则一直 True。

`_cbe_seed_hopping_paths(mps, config)`

- 针对 `HolsteinModel(..., scheme < 4)` 的分立电子自由度。
- 背景：局部 CBE selection 只能在单个 bond 上加 sector；但 scheme=2 下电子 hopping
  需要两个电子 site 之间整条 virtual path 都有合法 qn sector，否则局部新增 sector 会在 qn-aware
  SVD 中断裂或被删掉。
- 当前实现：
  1. 读取 `model.j_matrix`；
  2. 找出所有非零 `J_ij` 的电子自由度；
  3. 构造 ground state；
  4. 对每个 active dof 作用 `a^dagger` 得到一组 one-exciton seed；
  5. 将这些 seed 加和并乘以很小系数 `cbe_path_seed_coef`；
  6. 与当前 MPS 相加并 canonicalise；
  7. 只做一次，标记 `cbe_path_seed_done=True`。
- 这一步不是物理预演化，而是给 scheme=2 的 hopping path 提供 qn scaffold。

### CBE stats 和日志

`_cbe_init_stats(config, cbe_active, bond_dims_before, evolve_dt)`

- 初始化每个 TDVP step 的 CBE 统计，包括：
  - 当前 step/time/dt；
  - CBE active 状态；
  - bond dims before/after；
  - selection 次数、扩展 bond 数；
  - `D_expand` 分布；
  - discarded weights；
  - 新 sector local evolution 后、trim 前的 norm；
  - 正交/isometry/wavefunction preservation 误差；
  - zero expansion 原因。

`_cbe_record_selection(cbe_stats, selection, expansion=None, accepted=True)`

- 记录一次 CBE selection 和 expansion。
- 当 selection 有新增但 expansion 检查失败时，记录为 `nonisometric_expansion`。

`_cbe_record_discarded(cbe_stats, discarded_weight)`

- 记录 SVD trim 的 discarded weight。

`_cbe_record_pretrim(cbe_stats, new_sector_norm, total_norm, singular_values, keep_dim)`

- 记录 local TDVP evolution 后、CBE trim 前：
  - 新 sector norm；
  - 新 sector relative norm；
  - 完整 singular values；
  - trim 后保留维度。

`_cbe_finalize_stats(cbe_stats, mps)`

- 写入最终 bond dims，并计算平均 `D_expand`。

### CBE trim 和 isometry 检查

`_cbe_qn_key(qn)`

- 将 qn 转为 tuple，用于比较。

`_cbe_pick_indices(sigma, qnset, config, required_qn=None)`

- 根据 `cbe_eps_trim` 和 `cbe_Dmax` 从奇异值中选保留索引。
- 若存在 `required_qn`，会尝试至少保留每个新增 qn sector 中的一个代表方向。
- 这是为了避免刚扩展的新 qn sector 因初始权重过小立刻被 trim 删除。

`_cbe_left_isometry_error(tensor)`

- 检查左规范张量的 `A^dagger A - I`。

`_cbe_right_isometry_error(tensor)`

- 检查右规范张量的 `B B^dagger - I`。

`_cbe_svd_trim(mps, mps_t, imps, system, config, required_qn=None)`

- 对 local TDVP evolution 后的扩展 one-site tensor 做 qn-aware SVD。
- 使用 `svd_qn.svd_qn(..., QR=False)`，不是普通 QR。
- 根据 `_cbe_pick_indices()` 选择保留奇异值。
- `system == "L"` 时返回左规范 site tensor 和右侧 transfer；
- `system == "R"` 时返回右规范 site tensor 和左侧 transfer。

### `Mps.evolve()` dispatch 和 summary

`Mps.evolve(mpo, evolve_dt, normalize=True)`

- 原逻辑中 `EvolveMethod.tdvp_ps` 默认调用 `_evolve_tdvp_ps()`。
- 当前新增分支：
  - 若 `method == tdvp_ps` 且 `expansion_method == "cbe"`，调用 `_evolve_cbe_tdvp_ps()`；
  - 否则保持原方法映射。
- 演化后若存在 `cbe_last_stats`，打印 step summary：
  - step index/time/dt；
  - warm-up 或 production；
  - CBE 是否 active；
  - bond dims before/after；
  - CBE calls、expanded bonds、max/avg D_expand；
  - trim 前新 sector norm；
  - second singular value；
  - discarded weight；
  - norm 和 occupation finite 检查；
  - e occupations；
  - zero expansion reasons。

### `_evolve_cbe_tdvp_ps()`

这是 CBE-TDVP 的 sweep 主体。它保留原 `_evolve_tdvp_ps()` 的 one-site TDVP projector-splitting
结构，但在每个 one-site local evolution 前插入 CBE expansion。

整体结构：

1. 根据 `evolve_dt` 类型准备复数 MPS 和非 Krylov solver 的系数。
2. 判断 `cbe_active`。
3. 若 active，执行 `_cbe_seed_hopping_paths()`。
4. 构造 `Environ(mps, mpo)`。
5. 两轮 sweep：
   - 一轮 right-to-left；
   - 一轮 left-to-right；
   - 每轮后 `mps._switch_direction()`。

right-to-left 分支：

1. 条件：`cbe_active and (not mps.to_right) and imps != 0`。
2. bond index 为 `imps - 1`。
3. 先检查左侧 `A_l` 是否 left-isometry。
4. 调用 `cbe_shrewd_selection_right_to_left()` 得到 `A_tr` 和 `qn_new`。
5. 调用 `cbe_expand_right_to_left()` 得到 `A_ex, C_ex, L_ex`。
6. 检查 orthogonality/isometry/wavefunction preservation。
7. 若通过：
   - `mps[bond_idx] = A_ex`；
   - `mps[imps] = C_ex`；
   - `mps.qn[imps] = concat(old_qn, qn_new)`；
   - 写回 `environ.write("L", bond_idx, L_ex)`；
   - 标记 `use_cbe_trim=True`。
8. 构造扩展后的 one-site effective Hamiltonian：
   - `hop_expr(l_array, r_array, [mpo[imps]], shape)`。
9. 用原有 local TDVP evolution 演化 `mps[imps]`。
10. CBE 模式下用 `_cbe_svd_trim()` 替代普通 QR-SVD。
11. 写回右规范 tensor、更新右 environment，并做 zero-site backward evolution。

left-to-right 分支：

1. 条件：`cbe_active and mps.to_right and imps != len(mps)-1`。
2. bond index 为 `imps`。
3. 检查右侧 `B_r` 是否 right-isometry。
4. 调用 `cbe_shrewd_selection_left_to_right()` 得到 `B_tr` 和 `qn_new`。
5. 调用 `cbe_expand_left_to_right()` 得到 `C_ex, B_ex, R_ex`。
6. 若检查通过：
   - `mps[imps] = C_ex`；
   - `mps[bond_idx+1] = B_ex`；
   - `mps.qn[imps+1] = concat(old_qn, qn_new)`；
   - 写回 `environ.write("R", imps+1, R_ex)`；
   - 标记 `use_cbe_trim=True`。
7. 构造扩展后的 one-site effective Hamiltonian。
8. local TDVP evolution。
9. `_cbe_svd_trim()`。
10. 写回左规范 tensor、更新左 environment，并做 zero-site backward evolution。

CBE inactive 时，`_evolve_cbe_tdvp_ps()` 基本退化为普通 `_evolve_tdvp_ps()`：

- 不调用 selection/expansion；
- 使用原来的 `svd_qn.svd_qn(..., QR=True)`；
- 保持 fixed-rank 1TDVP。

## 4. 相较于原 1TDVP 的改动

原 `_evolve_tdvp_ps()` 是 fixed-rank one-site TDVP：

1. 读 left/right environment。
2. 构造 one-site effective Hamiltonian。
3. local evolution。
4. 用 qn-aware QR/SVD 分解。
5. 更新 environment。
6. 做 zero-site backward evolution。

CBE-TDVP 的新增点：

1. 新增 `expansion_method = "none" | "krylov" | "cbe"`。
2. 保留原 `_evolve_tdvp_ps()`，新增 `_evolve_cbe_tdvp_ps()`。
3. `Mps.evolve()` 根据 `expansion_method=="cbe"` dispatch 到 CBE 版本。
4. 在每个 local one-site TDVP update 前插入：
   - shrewd selection；
   - qn-block selection；
   - local bond expansion；
   - expanded environment 写回。
5. 在 local evolution 后，CBE active 时用 SVD trim 替代普通 QR 形式的分解。
6. 增加 warm-up/prod 阶段控制。
7. 增加 scheme=2 下电子 hopping path seed。
8. 增加大量 CBE 诊断日志。

这不是 Krylov 全局预扩展。Krylov expander 仍由旧逻辑保留；CBE 是 sweep 内局部动态扩展。

## 5. 量子数处理

CBE-TDVP 当前实现对 qn 的处理是核心部分。

### selection 阶段

对 bond `l|l+1`，当前总 qn 为 `mps.qntot`。

right-to-left 与 left-to-right 都构造：

- 左侧候选 qn：
  `q_left = qn[l] + sigmaqn[l]`
- 右侧候选 qn：
  `q_right = sigmaqn[l+1] + qn[l+2]`

只有满足：

```text
q_right == mps.qntot - q_left
```

的 qn sector 才允许参与 CBE selection。

这样每个 qn block 的 selection 都在总量子数守恒的子空间内进行。`A_tr` 或 `B_tr`
不是先 dense SVD 后再分配 qn，而是先决定合法 qn block，再在这个 block 内 selection。

### 当前 bond basis 投影

每个 qn block 内会取当前 bond qn：

```text
bond_qn = mps.qn[bond_idx + 1]
```

然后从 selection block 中投影掉已有的 `A_l`/`B_r` 当前 qn basis。这样新增方向是当前 bond space
的补空间，而不是重复已有 sector。

### 新增 qn 写回

selection 返回 `qn_new`。扩展成功后：

- right-to-left：
  `mps.qn[imps] = concat(old_qn, qn_new)`
- left-to-right：
  `mps.qn[imps+1] = concat(old_qn, qn_new)`

这使新增 virtual bond dimension 与 MPS qn bookkeeping 同步。

### SVD trim 阶段

`_cbe_svd_trim()` 使用 Renormalizer 原有 `svd_qn.svd_qn()`，因此分解本身仍是 qn-aware。
随后 `_cbe_pick_indices()` 根据奇异值和 `cbe_Dmax` 选择保留方向，并尝试保留 `required_qn`
中新增 sector 的至少一个方向。

### scheme=2 的额外处理

FMO scheme=2 使用分立电子自由度，电子 site 与声子 site 交错。单个 local bond 上加入 qn
sector 不一定足以形成完整的电子 hopping path。因此 `_cbe_seed_hopping_paths()` 在 warm-up
开始时根据 MPO hopping channel 对整条 path 做极小幅度 seed。

这一步的目的不是改变物理初态，而是提供 qn scaffold，让后续 CBE selection 和 TDVP 实时间演化
能在合法 qn sector 中传播激子占据。

## 6. 当前遇到的案例困难

### FMO scheme=2 初期困难

早期在 `HolsteinModel(..., scheme=2)` 下，激子 occupation 不转移，而 `scheme=4`
把电子自由度合并进 `BasisMultiElectronVac` 后结果正常。这说明问题不主要是 TDVP 时间演化本身，
而是分立电子自由度下 virtual bond qn sector 没有被正确铺开。

当前通过两部分缓解：

1. selection 改为 qn-block 先验选择；
2. 增加 hopping path seed，保证 scheme=2 的电子 hopping path 有完整 qn scaffold。

### 旧 two-site `H|psi>` 路径的 GPU OOM

M3000 压力测试中，早期 CBE selection 会构造完整 two-site `theta` 并调用 two-site
`hop_expr`，导致类似 10 GB 级临时 tensor，并在 CuPy `tensordot` 中 OOM。

当前已改为 `_hpsi_qn_block()`：

- 只构造合法 qn block；
- 通过 factorized contraction 得到 block；
- 避免 full `theta -> hop(theta)`。

### 当前 M3000 压力测试瓶颈

任务 `89511` 的日志显示：

```text
step 1:   36.8 s, max_bond_after=11
step 2:   39.3 s, max_bond_after=20
step 3:   79.6 s, max_bond_after=40
step 4:  322.6 s, max_bond_after=83
step 5: 1849.8 s, max_bond_after=160
step 6: 12909.6 s, max_bond_after=303
step 7: 长时间未完成
```

这说明当前瓶颈已经从 GPU OOM 转移到 CPU dense SVD 的计算复杂度。`cbe.py` 中仍有：

```python
scipy.linalg.svd(block, full_matrices=False)
```

当 qn block 随 bond dimension 增大时，full SVD 成本快速增长，接近三次复杂度。

### 建议后续优化

1. 将 qn block full SVD 改为 partial SVD 或 randomized SVD。
2. 利用 `_hpsi_qn_block()` 的 factorized 结构，不显式构造 full block，而是实现
   `LinearOperator`：
   - `block @ x = left_mat @ (right_mat.T @ x)`；
   - `block.T @ y = right_mat @ (left_mat.T @ y)`。
3. 根据 `cbe_max_expand` 只求 top-k singular vectors。
4. 增加 per-qn-block 计时日志：
   - bond；
   - qn；
   - `len(lidx), len(ridx)`；
   - block shape；
   - block construction time；
   - SVD time。
5. 给 M3000 先设置更保守的限制：
   - `CBE_MAX_EXPAND=4` 或 `8`；
   - `CBE_DMAX=256` 或 `512`；
   - 避免 warm-up 中 bond dimension 几何式增长。

## 7. 相关配置参数

虽然配置类在 `renormalizer/utils/configs.py`，但与当前 CBE-TDVP 直接相关：

- `expansion_method`：`"none" | "krylov" | "cbe"`。
- `cbe_Dmax`：CBE trim 后最大 bond dimension。
- `cbe_eps_pre`：selection 预筛阈值。
- `cbe_eps_final`：selection 最终阈值。
- `cbe_eps_trim`：local evolution 后 SVD trim 阈值。
- `cbe_max_expand`：每个 bond 每次最多新增维数。
- `cbe_warmup_time`：CBE warm-up 总时间。
- `cbe_warmup_substeps`：warm-up 子步数。
- `cbe_disable_after_warmup`：warm-up 后是否关闭 CBE。
- `cbe_lock_after_warmup`：warm-up 后是否锁定最大 bond dimension。
- `cbe_path_seed`：是否启用 scheme=2 hopping qn path seed。
- `cbe_path_seed_coef`：path seed 的小扰动系数。
- `cbe_isometry_tol`：扩展后 isometry/wavefunction preservation 检查阈值。

