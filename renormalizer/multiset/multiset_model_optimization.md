# multiset_model.py 优化说明

本文档记录本次对 `renormalizer/model/multiset_model.py` 的全部性能优化修改，包括：

- 修改的目标
- 每一项优化对应的方法
- 对应代码位置
- 关键代码片段
- 已完成的验证与当前结论

本次优化严格只修改了一个源码文件：

- `renormalizer/model/multiset_model.py`

新增本说明文件：

- `renormalizer/model/multiset_model_optimization.md`

## 1. 优化目标

当前 multiset-tdvp / multiset-mps 的瓶颈主要来自以下几个方面：

1. `MultisetModel` 中有 `N` 个 MPS，`N x N` 个 MPO block，很多循环都按满 `N^2` 规模执行。
2. `_ms_evolve_tdvp_ps` 是最核心的热点函数，会被反复调用。
3. 在 sweep 中，不仅主站点演化频繁进入 Python 层循环，而且每个 `alpha` 的 QR/SVD 也会反复单独调用。
4. `_apply_hop_batched` 已有并行思路，但仍然存在分组构造、scatter 组装、空 block 遍历等额外开销。
5. 在 GPU 后端下，真正应该尽量保留的是大批量张量运算；应该尽可能减少 Python 调度和无贡献 block 的处理。

本次优化的总原则是：

- 不改变算法功能
- 不改变数值流程的核心形式
- 优先减少空工作
- 尽量把批量线性代数保留在 `xp` 后端上
- 让 GPU 更多地处理大张量，而不是让 CPU 调度大量细碎对象

## 2. 本次修改总览

| 修改项 | 方法 | 主要代码位置 | 主要收益 |
| --- | --- | --- | --- |
| 稀疏化 `MPO` block 遍历 | 只缓存非空 `(alpha, beta)` 对 | `__init__`, `_refresh_mpo_cache` | 避免大量空 `MPO` 的环境构造与循环 |
| 预构造站点分组模板 | 预缓存每个站点的 `W` 分组和 scatter 结构 | `_refresh_mpo_cache`, `_build_site_batched_data` | 避免每个 sweep/site 反复重建分组 |
| 量子数分块 batched QR | 缓存 QN block 结构后做 batched QR/RQ | `_get_qr_qn_plan`, `_batched_qr_qn` | 降低 `N` 个 MPS 独立 QR 的 Python 开销 |
| 演化主循环稀疏化 | `_ms_evolve_tdvp_ps` 只处理活跃 pair | `_ms_evolve_tdvp_ps` | 显著降低 `Environ` 与反向环境构造成本 |
| 反向 U/C 环境批处理 | reverse step 也改成活跃 pair 分组 | `_build_reverse_batched_data` | 进一步减少 Python 列表和空块处理 |
| 保留 batched einsum 核心 | 继续用 grouped einsum，改进 scatter | `_apply_hop_batched` | 保持 GPU 上的大批量收缩 |
| `expand` / `Hamiltonian` 稀疏化 | 只对活跃 pair 计算 | `expand_bond_dimension_multiset`, `Hamiltonian` | 减少初始化与观测量计算的无效循环 |
| 小修正 | 去除 `a^\dagger` 字符串的 `SyntaxWarning` | `_reset_all_MsOp` | 清理告警，不影响性能 |

## 3. 修改 1：引入活跃 `(alpha, beta)` 稀疏缓存

### 3.1 问题

原实现中，很多地方直接按完整的 `N x N` 双重循环处理所有 `MPO block`。  
但对 Holstein / FMO 这类模型来说，大量 `H^{alpha,beta}` 实际上是空的。

原来的代价包括：

- 为空 `MPO` 也构造 `Environ`
- 主 sweep 时对空 block 做判断
- reverse U/C 时仍然按 `N^2` 结构组织
- `Hamiltonian` 和 `expand_bond_dimension_multiset` 中也存在相同问题

### 3.2 方法

增加以下缓存结构：

- `self._active_pairs`
- `self._active_alpha`
- `self._active_beta`
- `self._active_pair_mpos`
- `self._active_pairs_by_alpha`
- `self._site_group_templates`

这些结构由 `_refresh_mpo_cache()` 统一构造。

### 3.3 代码位置

- `multiset_model.py:234-242`
- `multiset_model.py:312-373`

### 3.4 核心代码

```python
self.MsMpo = MultisetMpo(self.MsModel,self.N_electron)
self._active_pairs: List[Tuple[int, int]] = []
self._active_alpha: List[int] = []
self._active_beta: List[int] = []
self._active_pair_mpos: List[Mpo] = []
self._active_pairs_by_alpha: List[List[int]] = [[] for _ in range(self.N_electron)]
self._site_group_templates = []
self._qr_qn_plan_cache = {}
self._refresh_mpo_cache()
```

```python
for alpha in range(self.N_electron):
    for beta in range(self.N_electron):
        mpo = self.MsMpo.msmpo[alpha][beta]
        if len(mpo) == 0:
            continue
        pair_id = len(active_pairs)
        active_pairs.append((alpha, beta))
        active_pair_mpos.append(mpo)
        active_pairs_by_alpha[alpha].append(pair_id)
```

### 3.5 效果

这一步直接把很多地方的复杂度从“按 `N^2` 逻辑遍历”变成“只遍历有贡献的 pair”。  
对于大体系，这是本次提速中最直接的一项。

## 4. 修改 2：预构造每个站点的分组模板

### 4.1 问题

原 `_build_batched_data` 会在每个站点、每次 sweep 中：

- 重新扫描所有 `(alpha, beta)`
- 重新按 `W.shape` 分组
- 重新 `stack` `W`
- 重新生成 scatter 结构

这些操作虽然单次不大，但在 `_ms_evolve_tdvp_ps` 的双 sweep 和 Krylov matvec 下会被放大很多次。

### 4.2 方法

把每个站点的 `W` 分组模板提前缓存到 `self._site_group_templates` 中。  
这样在实际演化时只需要：

- 读取当前站点的 `L/R`
- 按模板做一次 `stack`
- 直接进入 `_apply_hop_batched`

同时缓存 scatter matrix `S`，避免每次重复构造。

### 4.3 代码位置

- `multiset_model.py:331-373`
- `multiset_model.py:375-379`
- `multiset_model.py:460-476`

### 4.4 核心代码

```python
for imps, local_mpo in enumerate(mpo):
    W = asxp(local_mpo.array)
    key = tuple(W.shape)
    bucket = site_group_dicts[imps].setdefault(
        key,
        {
            "pair_ids": [],
            "alpha_idx": [],
            "beta_idx": [],
            "w_tensors": [],
        },
    )
    bucket["pair_ids"].append(pair_id)
    bucket["alpha_idx"].append(alpha)
    bucket["beta_idx"].append(beta)
    bucket["w_tensors"].append(W)
```

```python
templates.append(
    {
        "pair_ids": tuple(bucket["pair_ids"]),
        "alpha_idx": xp.asarray(bucket["alpha_idx"], dtype=np.int64),
        "beta_idx": xp.asarray(bucket["beta_idx"], dtype=np.int64),
        "S": self._build_scatter_matrix(bucket["alpha_idx"], bucket["w_tensors"][0].real.dtype),
        "W": xp.stack(bucket["w_tensors"]),
        "nsite": 1,
        "n_pairs": len(bucket["pair_ids"]),
    }
)
```

### 4.5 效果

这一步减少了热点路径中的：

- Python 分组逻辑
- 重复 `xp.stack(W)`
- 重复 scatter 结构构造

尤其适合 `N` 较大、站点数也较多时的重复 sweep。

## 5. 修改 3：缓存量子数分块结构并实现 batched QR/RQ

### 5.1 问题

原实现中，`_ms_evolve_tdvp_ps` 对每个 `alpha` 都调用一次：

```python
svd_qn.svd_qn(..., QR=True, ...)
```

也就是说，即使所有 multiset 元素共享同一个：

- `qnbigl`
- `qnbigr`
- `qntot`

量子数分块结构仍然会被反复在 Python/NumPy 层重新分析，并逐个做 QR/RQ。

当 `N=75` 时，这会造成非常显著的 Python 调度和 CPU 参与。

### 5.2 方法

拆成两步：

1. `_get_qr_qn_plan()`  
   只根据 `qnbigl/qnbigr/qntot` 生成一次量子数分块计划，并缓存。

2. `_batched_qr_qn()`  
   对每个 QN block 一次性抽出所有 `alpha` 的 block，使用 `xp.linalg.qr` 做 batched QR/RQ，再恢复回完整矩阵。

这里：

- `system == "L"` 对应 batched QR
- `system == "R"` 通过转置后的 QR 实现 batched RQ

### 5.3 代码位置

- `multiset_model.py:381-417`
- `multiset_model.py:419-458`

### 5.4 核心代码

```python
def _get_qr_qn_plan(self, qnbigl, qnbigr, qntot):
    cache_key = (
        qnbigl.tobytes(),
        qnbigr.tobytes(),
        np.asarray(qntot).tobytes(),
    )
    if cache_key in self._qr_qn_plan_cache:
        return self._qr_qn_plan_cache[cache_key]
```

```python
for lset, rset, nl, nr in self._get_qr_qn_plan(qnbigl, qnbigr, qntot):
    block = coef_matrix[:, lset][:, :, rset]

    if system == "L":
        u_block, vt_block = xp.linalg.qr(block, mode="reduced")
    else:
        q_t, r_t = xp.linalg.qr(xp.swapaxes(block, -1, -2), mode="reduced")
        u_block = xp.swapaxes(r_t, -1, -2)
        vt_block = xp.swapaxes(q_t, -1, -2)
```

### 5.5 效果

这是本次最重要的“减少 `N` 个 MPS 之间 Python 开销”的优化之一。

它保留了：

- 原本的量子数分块逻辑
- 原本的 QR / RQ 数学流程

但把原来“每个 `alpha` 分开调用”的模式，改成了：

- 同一分块结构下的一次 batched 线性代数操作

这显著增强了 GPU 后端的利用率。

## 6. 修改 4：重写 `_ms_evolve_tdvp_ps` 的热点数据流

### 6.1 问题

`_ms_evolve_tdvp_ps` 是整个程序最核心、调用最频繁的函数。  
原版本中的主要热点有：

- 为所有 `N^2` pair 构造 `Environ`
- 每个站点都重建 `l_array_ab/r_array_ab/w_array_ab`
- 每个 `alpha` 单独做 QR
- reverse U/C 环境仍按 `N^2` 逻辑组织

### 6.2 方法

新的 `_ms_evolve_tdvp_ps` 做了以下改动：

1. `Environ_list` 只为活跃 pair 构造
2. 当前站点的 `L/R` 只读取活跃 pair
3. 主站点 batched 数据直接由 `_build_site_batched_data()` 构造
4. QR 部分改为 `_batched_qr_qn()`
5. reverse U/C 的环境只对活跃 pair 更新
6. `Y0/U0/C0` 都改为按 `xp.stack(...).reshape(-1)` 形成连续批量向量

### 6.3 代码位置

- `multiset_model.py:531-719`

### 6.4 核心代码

```python
conj_mps = [mps_alpha.conj() for mps_alpha in ms_mps.msmps]
Environ_list = [
    Environ(
        ms_mps.msmps[self._active_beta[pair_id]],
        self._active_pair_mpos[pair_id],
        mps_conj=conj_mps[self._active_alpha[pair_id]],
    )
    for pair_id in range(len(self._active_pairs))
]
```

```python
l_array_ab = [environ.read("L", imps - 1) for environ in Environ_list]
r_array_ab = [environ.read("R", imps + 1) for environ in Environ_list]
batched_data = self._build_site_batched_data(imps, l_array_ab, r_array_ab)
Y0 = xp.stack(
    [asxp(ms_mps.msmps[a][imps].array).reshape(dim) for a in range(self.N_electron)]
).reshape(-1)
```

```python
u_batch, qnlset, vt_batch, qnrset = self._batched_qr_qn(
    mps_t,
    qnbigl,
    qnbigr,
    ms_mps.msmps[0].qntot,
    system,
)
```

```python
for pair_id in self._active_pairs_by_alpha[alpha]:
    beta = self._active_beta[pair_id]
    r_array_u[pair_id] = Environ_list[pair_id].GetLR(
        "R", imps, ms_mps.msmps[beta], self._active_pair_mpos[pair_id],
        itensor=r_array_ab[pair_id], method="System", mps_conj=mps_conj_alpha
    )
```

### 6.5 效果

该修改把整个 TDVP 主流程从“满 `N^2` 组织 + 每个 `alpha` 单独 QR”改成了：

- 活跃 pair 稀疏遍历
- batched 环境组装
- batched QN-aware QR

这就是本次性能提升的主要来源。

## 7. 修改 5：保留 batched einsum 主体，但改进 `_apply_hop_batched`

### 7.1 问题

原 `_apply_hop_batched` 已经是比较正确的 GPU 友好写法，但 scatter 部分原先依赖每组动态构造的 `S`。  
后续尝试里也发现 `cupy.add.at` 对复数支持不理想，因此最终保留了更稳定的矩阵乘法 scatter。

### 7.2 方法

对于每个 group：

1. `Y[beta_idx]` 先 gather 出对应输入
2. 用 batched einsum 完成张量收缩
3. 用缓存的 scatter matrix `S` 做：

```python
Y_out += xp.matmul(S, out_flat)
```

### 7.3 代码位置

- `multiset_model.py:901-967`

### 7.4 核心代码

```python
Y_exp = Y[beta_idx]
```

```python
if len(shape) == 3:
    temp = xp.einsum('ncek,nlfk->ncelf', Y_exp, R_all)
    temp2 = xp.einsum('ncelf,nbdef->ncdlb', temp, W_all)
    out = xp.einsum('ncdlb,nabc->nadl', temp2, L_all)
elif len(shape) == 4:
    temp = xp.einsum('ncegk,nlfk->nceglf', Y_exp, R_all)
    temp2 = xp.einsum('nceglf,nbdef->ncglbd', temp, W_all)
    out = xp.einsum('ncglbd,nabc->nadgl', temp2, L_all)
```

```python
out_flat = out.reshape(n_pairs, dim)
Y_out += xp.matmul(S, out_flat)
```

### 7.5 效果

这一步保持了原算法的 batched 张量收缩形式，同时让 scatter 过程：

- 更稳定
- 支持复数
- 避免运行时重新造矩阵

## 8. 修改 6：`expand_bond_dimension_multiset` 稀疏化

### 8.1 问题

原实现中，构造

- `Σ_{beta != alpha} H^{alpha,beta} |Psi_beta>`

时，仍按完整 `N` 遍历并检查空 `MPO`。

### 8.2 方法

直接改为只遍历 `self._active_pairs_by_alpha[alpha]`。

### 8.3 代码位置

- `multiset_model.py:735-807`

### 8.4 核心代码

```python
for pair_id in self._active_pairs_by_alpha[alpha]:
    beta = self._active_beta[pair_id]
    if beta == alpha:
        continue
    driven = self._active_pair_mpos[pair_id].apply(original_mps[beta])
    cross_states.append(driven)
```

### 8.5 效果

减少初始化阶段的无效 `MPO.apply(...)` 检查和空遍历。

## 9. 修改 7：`Hamiltonian` 稀疏化

### 9.1 问题

原 `Hamiltonian()` 同样按 `N^2` 遍历全部 `(alpha, beta)`，即使多数 block 是空的。

### 9.2 方法

直接对 `self._active_pairs` 求和，并且预先缓存 bra：

```python
bras = [self.MsMps.msmps[a].conj() for a in range(self.N_electron)]
```

### 9.3 代码位置

- `multiset_model.py:864-874`

### 9.4 核心代码

```python
num = 0.0
bras = [self.MsMps.msmps[a].conj() for a in range(self.N_electron)]
for pair_id, (alpha, beta) in enumerate(self._active_pairs):
    num += self.MsMps.msmps[beta].expectation(
        mpo=self._active_pair_mpos[pair_id],
        self_conj=bras[alpha]
    )
den = sum(bras[a].dot(self.MsMps.msmps[a]) for a in range(self.N_electron))
```

### 9.5 效果

减少初始化时计算 `E0` 的无效循环。

## 10. 修改 8：MPO 更新后刷新缓存

### 10.1 问题

`cdd_init_mps()` 里会重新构造带 offset 的 MPO。  
如果此时不刷新缓存，后面使用的活跃 pair / 分组模板就不是最新的。

### 10.2 方法

在 MPO 重建之后显式调用：

```python
self._refresh_mpo_cache()
```

### 10.3 代码位置

- `multiset_model.py:827-845`

### 10.4 效果

确保：

- 活跃 pair 信息与当前 MPO 一致
- 站点分组模板与当前 MPO 一致

## 11. 修改 9：清理 `SyntaxWarning`

### 11.1 内容

原 `_reset_all_MsOp()` 中字符串含有：

- `"a^\dagger a "`
- `"a^\dagger a"`

Python 会把 `\d` 视作无效转义并报 `SyntaxWarning`。  
已修改为：

- `"a^\\dagger a "`
- `"a^\\dagger a"`

### 11.2 代码位置

- `multiset_model.py:281-284`

### 11.3 影响

仅清理告警，不影响数值结果与性能逻辑。

## 12. 结果与验证

### 12.1 语法检查

已通过：

```bash
python3 -m py_compile renormalizer/model/multiset_model.py
```

### 12.2 功能 smoke test

已验证以下情况可以正常运行：

1. 两站点 Holstein，`T=0`
2. 两站点 Holstein，有限温度 `T=300K`
3. 有限温度测试确认本次改动仍可走到 `MpDm` / 4 维局域张量路径

验证到的现象：

- 演化后总 population 保持为 `1`
- `ivp_calls` / `matvec_calls` 正常增长
- 0K 与 finite-T 两条路径都可以完成至少一个时间步

### 12.3 Holstein-75 实测

本地测试条件：

- 模型：Holstein-75
- `max_bonddim = 16`
- `evolve_dt = 0.1`
- GPU 后端：CuPy

得到：

- `active_pairs = 223`
- `site_num = 75`
- `evolve_s = 58.074 s`

与你原先日志中的量级对比：

- 原来约 `180 s / step`
- 优化后约 `58 s / step`

即单步时间大约下降到原来的三分之一左右。

## 13. 为什么这次优化有效

本次提速并不是靠改变算法本身，而是主要来自以下三点：

1. 从满 `N^2` 组织改成只处理有贡献的 `MPO block`
2. 把原本按 `alpha` 逐个执行的 QR/RQ 改成 batched 线性代数
3. 让站点级分组模板、scatter 结构、QN block 结构都能缓存复用

也就是说，本次优化的实质是：

- 减少 Python 层管理成本
- 减少 CPU 参与
- 增强 GPU 上大批量张量运算的占比

这和 multiset 模型在大 `N` 下的主要瓶颈是完全对应的。

## 14. 当前状态

当前可以确认的是：

- 代码已实现
- 速度提升已经确认
- 小规模与有限温度 smoke test 已通过

当前仍建议继续做的工作是：

1. 用你已有的 FMO / Holstein-75 正确性基准继续做长时间轨迹验证
2. 检查 population、energy、reduced density matrix 等观测量是否与旧版本一致
3. 如需继续提速，下一步应优先考虑把更多 multiset 数据结构直接 batched 化，而不是保留 `N` 个独立 `Mps` 对象

## 15. 结论

本次对 `multiset_model.py` 的优化，主要完成了：

- 稀疏化 `N x N` block 处理
- 预缓存站点分组模板
- batched QN-aware QR/RQ
- 演化主循环的活跃 pair 重构
- `expand` / `Hamiltonian` 的稀疏化

在不改变算法功能和数值流程的前提下，已经显著降低了大体系下的时间开销，并且更充分发挥了 GPU 后端的优势。
