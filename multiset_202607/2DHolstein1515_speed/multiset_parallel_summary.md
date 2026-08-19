# multiset 4-GPU (MPI) 并行方法分析总结

分析对象：`multiset_202607/2DHolstein1515_speed/multiset_parallel/`
基准算例：2D Holstein 15×15，ω₀=1.0，J=1.0，g=0.5，ν_max=8，M=64，dt=0.1，500 步
硬件：单节点 4×V100，OpenMPI 4.1.8（**非 CUDA-aware**）+ CuPy
作业记录：`multiset_parallel_runs/H1515_m64_local_500steps_118009.log`

---

## 0. 结论速览

| 指标 | 结果 |
|---|---|
| 数值正确性 | ✅ 与单卡串行前 100 步 population 对比，前 3 步 max\|Δ\| ~1e-16…1e-11，第 100 步 9.6e-6（浮点重结合导致的轨迹漂移，非 bug） |
| 4×V100 相对 1×V100 加速比 | 步 1–10: **1.24×**；步 11–50: **1.19×**；步 51–100: **1.09×**（并行效率 27%，且随步数下降） |
| 被并行化部分占总墙钟时间 | **6.4%**（compute 2.1% + comm 4.2%） |
| 通信 / 计算 比 | **2.2**（通信比它换来的计算更贵） |
| 真正的热点 | `Environ.GetLR` 环境张量更新，**完全串行且在 4 个 rank 上复制** |

**一句话结论：实现是正确的、设计是自洽的，但并行化对准了一个只占 6% 的热点，因此 Amdahl 上限就锁死在 1.07× 左右。**

---

## 1. 代码组织：三层 monkey patch

全部实现在算例目录内，**不修改 `renormalizer/` 包**，运行时替换三个函数：

| 文件 | 替换的目标 | 并行维度 | 通信原语 |
|---|---|---|---|
| [mpi_expand_patch.py](multiset_parallel/mpi_expand_patch.py) | `MultisetModel.expand_bond_dimension_multiset` | 初始键维扩张的 α 循环（round-robin） | `comm.bcast`（pickle 整条 MPS） |
| [mpi_krylov_patch.py](multiset_parallel/mpi_krylov_patch.py) | `renormalizer.lib.krylov.expm_krylov` | Lanczos 基矢 / 向量代数 | `allreduce`(标量) + `Allgatherv` |
| [mpi_apply_hop_patch.py](multiset_parallel/mpi_apply_hop_patch.py) | `MultisetModel._apply_hop_batched` | 有效 Hamiltonian 的 (α,β) pair | `Alltoallv`(halo) 或 `Reduce_scatter` |

启动顺序在 [Holstein1515.py](multiset_parallel/Holstein1515.py) 中**必须**是：
`_configure_rank_gpu()` 设 `RENO_GPU` → `import renormalizer` → `install_patch()` ×3。
原因：[backend.py:26](../../renormalizer/mps/backend.py#L26) 在 import 时刻读取 `RENO_GPU` 并绑定设备。

另有 [independent_worker.py](multiset_parallel/independent_worker.py)：**粗粒度**并行入口（一卡跑一个完整独立算例，用于参数扫描），不 import mpi4py，与上述细粒度方案是两条互不相干的路线。

---

## 2. 并行方法的基本理论

### 2.1 可并行的维度来自 multiset 的块结构

multiset 表示 |Ψ⟩ = Σ_α |α⟩ ⊗ |φ_α⟩，α = 0…N_e−1（此例 N_e = 225）。TDVP-PS 单点方程的有效 Hamiltonian 是一个 N_e×N_e 的块矩阵：

```
(H_eff Y)_α  =  Σ_{β : (α,β) ∈ P}  L^{αβ} · W^{αβ} · R^{αβ} · Y_β
```

`P` = active pairs，由 [`_active_mpo_select_grouping`](../../renormalizer/multiset/multiset_model.py#L154) 构造。
15×15 开边界最近邻：对角 225 + 非对角 2×420 = **1065 个 pair**。

于是有两条自然的切分：**按 pair 切**（负载最均衡，但输出要归约）和 **按电子行 α 切**（输出无需归约，但输入要 halo）。两条都实现了。

### 2.2 布局 A：`reduce_scatter`（兼容/正确性基线）

`_legacy_pair_sharded_action`（[mpi_apply_hop_patch.py:179](multiset_parallel/mpi_apply_hop_patch.py#L179)）

1. 每次矩阵作用前 `Allgatherv` 出完整的 Y（225×32768 complex128 ≈ 118 MB）；
2. 各 rank 用 `_pair_slice` 取连续的 pair 区间，算出**完整**的 Y_out；
3. `Reduce_scatter` 求和并把结果切片发回各 rank。

优点：与串行完全等价、负载天然均衡。缺点：每次 Lanczos 迭代两次全向量集合通信。

### 2.3 布局 B：`local`（生产路径，`RENO_MPI_KRYLOV_MODE=local`）

`_local_electronic_action`（[mpi_apply_hop_patch.py:424](multiset_parallel/mpi_apply_hop_patch.py#L424)）

rank r 拥有**连续的电子行区间** [d_r, d_r+c_r)（此例每 rank 56 或 57 行）：

1. `_build_local_plan`（[:246](multiset_parallel/mpi_apply_hop_patch.py#L246)）离线算出：本 rank 需要哪些远端 β 行（halo/ghost rows）、要发给谁、收到后放在拼接缓冲区的哪个位置（`available_lookup`）；
2. `_exchange_halo`（[:361](multiset_parallel/mpi_apply_hop_patch.py#L361)）一次 `Alltoallv` 把这些行取回，拼成 `Y_available = [本地行 | halo 行]`；
3. 只计算 α ∈ 本地区间的 pair，散射矩阵也只取本地行 `S[alpha_start:alpha_stop, selector]`；
4. **没有全局 gather，没有输出归约** —— 输出天然就是本 rank 拥有的那 56 行。

这就是典型的 **domain decomposition + ghost/halo exchange**，与有限差分/晶格 QCD 的做法同构：电子指标 α 就是"格点"，active pair 的 (α,β) 关系就是"stencil"。

### 2.4 分布式 Lanczos

`expm_krylov_distributed`（[mpi_krylov_patch.py:262](multiset_parallel/mpi_krylov_patch.py#L262)）

- `V_local` 形状 (m, local_count)：**每个 rank 只存 1/P 的 Krylov 基矢**（显存 1/P）；
- α_j = Σ_r ⟨w_r, v_r⟩ 由 `_distributed_vdot_real` 的 `allreduce(SUM)` 拼出；
- β_j = √(Σ_r ‖w_r‖²) 由 `_distributed_norm` 拼出；
- 三对角矩阵（≤50×50）在每个 rank 上**冗余对角化**，保证系数逐位一致；
- 收敛/退出判据用 `_all_true`（`allreduce MIN`）/ `_any_true`（`MAX`）做集合表决 —— 这是**防死锁的关键**：绝不允许某个 rank 单方面提前 return；
- 返回前做一次 `Allgatherv` 把完整向量交还给**完全未修改的** TDVP 调用方。

### 2.5 唯一的包内钩子：向量分块元数据

```python
ivp_eq._reno_vector_block_count = self.N_electron
```
[multiset_model.py:537](../../renormalizer/multiset/multiset_model.py#L537) / [:588](../../renormalizer/multiset/multiset_model.py#L588) / [:629](../../renormalizer/multiset/multiset_model.py#L629)

它告诉 Krylov patch"这个向量可以按 N_electron 个等长块切分"，从而能按电子行而不是按裸元素切。缺失时自动 fallback 到复制模式（计入 `missing_metadata_fallbacks`，本次运行为 0）。这是一个设计得很干净的契约。

### 2.6 halo 的表面/体积比（解释实测通信量）

row-major 编号下 56 个连续 α ≈ 3.7 个晶格行，边界是完整的两条晶格行：
- 内部 rank（1,2）halo = 2×15 = **30 行**
- 端点 rank（0,3）halo = 1×15 = **15 行**

日志实测完全吻合：`halo_rows_sent / halo_exchange_calls` = 135453360/4515112 = **30**（rank 1,2），67726680/4515112 = **15**（rank 0,3）。
即内部 rank 每次 Lanczos 迭代都要额外搬运 30/56 = **54%** 的数据量。

---

## 3. 实测性能数据（job 118009）

```
总 evolve 墙钟          255,862 s (71 h)
expand_bond_dimension   361 s（其中 bcast 45–49 s）
Krylov 求解次数         449,000 /rank ≈ 898 /步
平均 Lanczos 迭代       10.06 次/求解
hop 作用次数            4,515,112 /rank
compute_s               5,260 – 5,509 s   →  2.1%
comm_s                  10,007 – 11,875 s →  4.2%
                        ────────────────────────
被并行化的部分合计                          6.4%
halo 通信量（内部 rank） 39.4 TB 发 + 39.4 TB 收
单次 halo               8.7 MB / 2.6 ms → 3.3 GB/s（host staging 的 PCIe + 共享内存瓶颈）
负载不均衡              local_pairs 1.16e9 vs 1.23e9 → 3.4%（可忽略）
```

与单卡（job 115340，同参数 100 步，63,063 s）逐步对照：

| 步区间 | 1×V100 | 4×V100 | 加速比 |
|---|---|---|---|
| 1–10 | 774.9 s/步 | 624.4 s/步 | 1.24× |
| 11–50 | 682.1 s/步 | 571.6 s/步 | 1.19× |
| 51–100 | 560.6 s/步 | 515.4 s/步 | **1.09×** |
| 400–500 | — | 506.1 s/步 | — |

由 Amdahl 反推可并行份额 p = (1−1/S)/(1−1/P) ≈ **0.11**，与直接测得的 6.4%（+ expand）一致。**即使把 hop 的通信开销降到零，总加速比上限也只有约 1.07×。**

---

## 4. 真正的瓶颈在哪里（profile 证据）

在 6×6 / M=16 / ν_max=4 上做 cProfile（跳过首步，取稳态两步，共 19.07 s）：

| 部分 | 累计时间 | 占比 |
|---|---|---|
| `Environ.GetLR`（环境张量更新） | 8.95 s | **47%** |
| `expm_krylov` 总计 | 5.52 s | 29% |
| └ `_apply_hop_batched` | 4.39 s | 23% |
| CuPy H2D（`cupy._core.core.array`，tottime） | 3.88 s | **20%** |
| CuPy D2H（`.get()`，tottime） | 2.70 s | **14%** |

在 15×15 上 GetLR 的比重进一步压倒一切，因为两者的调用次数规模完全不同：

```
每步 GetLR 调用数 = 1065 pairs × 450 site-visit  = 479,250   （6×6 只有 10,920，44 倍）
每步 hop  调用数 =                                    9,030
```

`479,250 × ~1 ms ≈ 480 s/步`，恰好填满 505 s/步中 hop 之外的全部空白。

**三个根因：**

1. **环境更新没有批量化。** [multiset_model.py:567-581](../../renormalizer/multiset/multiset_model.py#L567-L581) 与 [:610-623](../../renormalizer/multiset/multiset_model.py#L610-L623) 是 `for alpha: for pair_id:` 的 Python 双重循环，每次 `GetLR` → `contract_one_site` → 3 次小 `tensordot`。而 `_apply_hop_batched` 早就做了按 W 形状分组的 batched einsum —— 两者的工程水平差了一个数量级。
2. **MPS 张量常驻主机内存。** `Matrix.array` 恒为 numpy（[matrix.py:26](../../renormalizer/mps/matrix.py#L26)），`tensordot` 每次调用 `asxp` 做 H2D（[matrix.py:210-211](../../renormalizer/mps/matrix.py#L210-L211)）。每步数百 GB 的 PCIe 往返，小算例里就已占 34%。
3. **这部分在 4 个 rank 上完全复制**，不但没有任何并行收益，还让 4 张卡各占一份完整环境显存。

---

## 5. 改进建议（按性价比排序）

### P0 —— 单卡就能拿到数倍收益，优先级远高于任何并行调优

1. **批量化环境更新（最大的一笔）。**
   仿照 `_build_site_batched_data` / `_apply_hop_batched`，把 1065 个 pair 的 `contract_one_site` 合并成按 W 形状分组的 batched einsum，一组一次调用：
   `L'_n = einsum(L_n, conj(A_{α_n}), W_n, A_{β_n})`。
   `_site_group_templates` 里已经 stack 好了 `W`、`alpha_idx`、`beta_idx`，直接复用即可。
   落点：新增 `MultisetModel._update_environ_batched()`，替换 [multiset_model.py:567-581](../../renormalizer/multiset/multiset_model.py#L567-L581) 与 [:610-623](../../renormalizer/multiset/multiset_model.py#L610-L623)。
   预期：单卡 5–10×。

2. **让 site 张量在一次 site-visit 内常驻显存。**
   把 225 个 α 在 `[imps]` 的张量一次 stack 成 (225, a, d, b) 的 device 数组上传，供 1065 个 pair 复用，而不是每个 pair 各自 `asxp`。可与第 1 条一起做。

3. **把 `xp.matmul(S, out_flat)` 换成 scatter-add。**
   `S` 是 (225 × 1065) 的 0/1 散射矩阵，用稠密 matmul 实现散射浪费了 N_e 倍 FLOP（O(N_e·n_pairs·dim) vs O(n_pairs·dim)）。改用 `cupyx.scatter_add` / 按 `alpha_idx` 的 index-add。
   落点：[multiset_model.py:797](../../renormalizer/multiset/multiset_model.py#L797)、[mpi_apply_hop_patch.py:214](multiset_parallel/mpi_apply_hop_patch.py#L214)、[:466](multiset_parallel/mpi_apply_hop_patch.py#L466)。

4. **修 `_build_local_plan` 的缓存失效。**
   cache 挂在 `batched_groups[0]` 上（[mpi_apply_hop_patch.py:257](multiset_parallel/mpi_apply_hop_patch.py#L257)），而 `batched_groups` 每次 site-visit 都由 `_build_site_batched_data` 新建 → **跨 Krylov 求解必然 miss**，449,000 次重复构建 halo 计划（其中含 O(P × n_pairs) 的 Python set 运算），且这段开销落在计时器之外、完全不可见。
   应改为挂在 `self._site_group_templates[imps]` 上，或用模型级 dict 以 `(imps, counts, displs)` 为 key。

### P1 —— 并行结构：把并行化重新对准热点

5. **按 α 分布环境张量（架构上最大的一笔）。**
   让 rank r 只持有 α ∈ [d_r, d_r+c_r) 的那些 pair 的 `Environ`，只更新自己那 ~266 个 pair。
   关键洞察：**这一步不需要任何额外通信** —— 每个 rank 的 Krylov 本来就只需要自己那些行的 L/W/R，而 β 侧的 MPS 已经全复制。顺带把环境显存降到 1/P。
   这能把可并行份额从 6% 提到 ~95%，是当前架构下收益最大的改动。
   落点：`_build_environ_list` / `_get_or_build_environ_list`（[multiset_model.py:246-268](../../renormalizer/multiset/multiset_model.py#L246-L268)）加 owner 过滤，环境更新循环用同一套过滤。

6. **用 NCCL 替代 host-staged MPI。**
   当前 3.3 GB/s 完全是 D2H + 共享内存造成的。单节点 4×V100 用 `cupy.cuda.nccl` 做 device-to-device AllGather/AlltoAll 可到 10–25 GB/s，且**彻底绕开"Curie 的 OpenMPI 不是 CUDA-aware"这个限制**（不需要换 MPI 实现）。halo 时间预期降 3–5×。

7. **通信/计算重叠。**
   把 pair 分成「纯本地 pair（α、β 都本地）」与「halo pair」两组，先发 `Ialltoallv`，算本地组，再算 halo 组。当前 comm = 2.2×compute，单靠重叠最多省掉 compute 那一份，需与第 6 条配合。

8. **二维块分解。**
   把 α 的划分从"连续区间"改成 15×15 晶格上的 2×2 块（每块 ~8×8）：halo 从 30 行降到 ~16 行，通信减 ~45%。
   需要把 `_partition_counts`（[mpi_krylov_patch.py:81](multiset_parallel/mpi_krylov_patch.py#L81)）和 `electron_counts/displs` 从"连续 range"泛化成任意 owner 映射 + 电子指标置换。
   **这是继续扩大 rank 数的前提**：1D 划分下 halo/本地 = 2L·P/N_e，P=4 已经 54%，P=8 就超过 100%，纯 1D 方案在 8 卡上必然负加速。

### P2 —— 算法与数值

9. **Krylov 收敛判据。** 现在每个偶数 j 都重新 `eigh_tridiagonal` 并**重建整条解向量**做 `allclose`（`_expm_krylov_local`，[mpi_krylov_patch.py:188](multiset_parallel/mpi_krylov_patch.py#L188)），代价 O(j·local_len)。换成标准的 `|β_j · u_{j,0}|` 残差估计几乎免费。平均 10 次迭代里要做 3 次这种检查。

10. **单精度 complex64。** 通信量与显存减半、V100 上 einsum 吞吐翻倍。当前 4-GPU vs 1-GPU 在 100 步已有 1e-5 的轨迹漂移，说明这个量级的扰动对 population 结论是可接受的 —— 值得先跑一个短程对照。

11. **明确区分两条并行路线。** 若目标是参数扫描（不同 g / J / 初始位点）而非单条长轨迹，`independent_worker.py` 的一卡一任务能拿到接近 4× 的**吞吐**，性价比远高于当前细粒度方案。细粒度方案的价值在于降低**单条轨迹的时延**和突破单卡显存 —— 后者目前还没实现（MPS 与环境仍全复制），建议作为第 5 条的附带目标一起达成。

### P3 —— 可观测性（避免再次优化错地方）

12. **补齐分段计时。** 目前只有 hop 有计时器，所以 GetLR 这个 90% 的热点在日志里完全不可见。建议在 `_ms_evolve_tdvp_ps` 中加：环境更新 / QR / Krylov(hop, 向量代数, 通信) 四段计时。

13. **修正异步计时失真。** CuPy 是异步的，当前 `compute_seconds` 只测到 kernel launch；真正的执行时间被记进了下一次 `_exchange_halo` 里 D2H 的同步等待。计时点前后需 `xp.cuda.Stream.null.synchronize()`。（这不影响"6.4%"这个结论 —— compute+comm 之和仍是上界，且与 Amdahl 反推的 0.11 相互印证。）

14. **打印每个 rank 的 GPU UUID / PCI bus id。** 现在日志里 4 个 rank 都显示 `backend_gpu=0`（因为 `--gpu-bind=single:1` 让每个 task 只看到一张卡，这次是对的）。但一旦绑定配置退化成 4 rank 共用 1 卡，现象与"并行效率低"**完全无法区分**。
