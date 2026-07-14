# Entanglement Analysis for Multiset MPS/TTN

## English Report

### Research Question

The focal paper, Li, Ren, and Yan, "A Further Comparison of MPS and TTNS for Nonadiabatic Dynamics of Exciton Dissociation", JCTC 2026, asks why singleset MPS and TTNS calculations for the P3HT:PCBM exciton-dissociation model can show persistent long-time discrepancies even though both ansatze are exact at sufficiently large bond dimension. The paper resolves the discrepancy by showing that it mainly comes from insufficient and unevenly distributed bond dimensions, and then uses bond entanglement entropy to redesign the tensor-network topology.

For the current multiset work, the analogous question is: why can multiset MPS/TTN achieve accurate dynamics with much smaller effective internal bond entropy than singleset MPS/TTN, and how should the observed maximum bond entropy curves be interpreted?

Evidence locations:
- Li 2026, Introduction: discrepancy attributed to insufficient bond dimension and related to entanglement patterns.
- Li 2026, Section 2.2: von Neumann entropy from singular values, and the bound that a bond with entropy `S` needs an effective bond dimension of at least `exp(S)`.
- Li 2026, Figures 5 and 7: entropy distribution and time evolution of maximum bond entropy guide topology design.
- Local code: `renormalizer/mps/mps.py::calc_bond_entropy`, `renormalizer/tn/tree.py::calc_bond_entropy`, `renormalizer/multiset/multiset_tdjob.py::_calc_bond_entropy_multiset`, `renormalizer/tn/multiset_ttn.py::calc_bond_entropy_summary`.

### Method

Li 2026 uses three linked diagnostics.

1. Compute bond von Neumann entropy from the singular values across each virtual bond:

   ```text
   S = - sum_l s_l^2 log(s_l^2)
   ```

2. Interpret the maximum bond entropy as the bottleneck under fixed uniform bond dimension. If the maximum entropy is `Smax`, a rough lower bound for the required Schmidt rank is `exp(Smax)`.

3. Use the spatial and temporal distribution of high-entropy bonds to redesign the topology. In their P3HT:PCBM model, low-frequency oligothiophene modes carry the largest entropy; TreeX distributes this entropy across top subtrees, while ChainX fails because simply putting a highly entangled unit in the middle of a chain does not preserve the interaction/order structure.

For the current multiset data, the same logic applies, with one important adjustment: current multiset `S_maxbond` is the maximum over independent conditional nuclear tensor networks:

```text
|Psi(t)> = sum_alpha |alpha> |psi_alpha(t)>
S_maxbond = max_alpha max_bond S(|psi_alpha(t)>)
```

It is not the entanglement entropy of the full electron+nuclear wavefunction across a singleset bond. Electronic-vibrational entanglement is separately captured by `S_el` from the electronic reduced density matrix.

### Data

Local data examined:

- 1D long-range Holstein data:
  - `multiset_202606/1DHolstein_nu16/plot.ipynb`
  - `multiset_202606/1DHolstein_nu16/Holstein31_alpha*/{multiset,singleset}/*.npz`
  - observables: `S_maxbond`, `S_el`, `e_occupations`, RMSD reconstructed from populations.

- P3HT:PCBM multiset TTN/MPS entropy data:
  - `multiset_ttn/P3HT:PCBM/plot/plot_entropy.ipynb`
  - `multiset_ttn/P3HT:PCBM/multiset_treeX/p3ht_ms_ttn_treeX_{16,32,64}.npz`
  - `multiset_ttn/P3HT:PCBM/multiset_chain/p3ht_ms_ttn_chain_16.npz`
  - `multiset_202605/P3HT:PCBM/multiset/p3ht_multiset_{16,32,64}.npz`
  - singleset references: `multiset_ttn/P3HT:PCBM/2026Further_plot/figures/MPS.npz`, `Tree4.npz`.

Local code findings:

- `calc_vn_entropy` normalizes the input probabilities before computing entropy. Therefore, scaling a multiset component by its electronic population does not change its internal bond entropy.
- A conda `reno` environment check confirmed that multiplying the same Schmidt values by factors 0.1, 0.5, 1, or 2 gives identical entropy.
- Current 1D `.npz` files only store scalar `S_maxbond`, not full `S_all` or `S_maxbond_eachset`; those arrays are supported by the code and should be saved for publication-grade attribution of jumps.

### Main Results

#### 1. Why multiset MPS has lower maximum bond entropy

In a singleset MPS, one tensor network must encode all electronic configurations and all corresponding nuclear wavepackets in the same chain. A cut through the chain can therefore see both:

- nuclear entanglement within each conditional wavepacket, and
- electronic-state-conditioned differences between nuclear wavepackets.

In a multiset MPS, the second part is moved outside each vibrational MPS into the explicit electronic sum. Each `|psi_alpha>` only needs to represent the nuclear wavepacket conditional on electronic state `alpha`. Therefore, the internal bond entropy is closer to the shape complexity of a single conditional nuclear wavepacket, not the entropy of the union of many incompatible wavepackets.

This explains the key numerical pattern in the 1D Holstein data. Comparing best available multiset curves with singleset `M=64`, the singleset `S_maxbond` is typically larger by `Delta S ~ 0.4-0.9`, corresponding to an effective Schmidt-rank ratio `exp(Delta S) ~ 1.5-2.5`. The reduction is not caused by small electronic populations: entropy normalization removes the population scale.

#### 2. Why `S_el` can be large while multiset `S_maxbond` remains modest

In the 1D Holstein data, `S_el` can be close to `2.8-3.1`, while multiset internal `S_maxbond` often stays around `1.2-1.8` at late times. This is not a contradiction. It means the full wavefunction has strong electronic-vibrational correlation, but the correlation is represented by the explicit sum over electronic states rather than by forcing a single vibrational chain to carry it.

For a JCTC paper, this separation should be presented as a central advantage:

```text
S_el diagnoses electronic-vibrational branching.
S_maxbond diagnoses the internal tensor complexity of each conditional nuclear branch.
Multiset works well when S_el is large but conditional branch entropies remain moderate.
```

#### 3. Why oscillations appear in maximum bond entropy

There are two sources.

Physical source:
- In Holstein dynamics, Franck-Condon wavepackets breathe and revisit resonance regions with roughly vibrational-period structure.
- As the electronic packet spreads and partially refocuses, the conditional nuclear wavepackets alternately become more and less distinguishable across spatial cuts.
- Therefore, bond entropy oscillates with the same coherent vibronic mechanism that produces population/RMSD oscillations.

Numerical/diagnostic source:
- `S_maxbond` is a maximum over bonds and, in multiset, over electronic components.
- The identity of the maximizer can change in time. When the bottleneck bond or bottleneck electronic component switches, the scalar maximum can show cusp-like changes even if every individual bond entropy is smooth.
- This is directly analogous to Li 2026, where the maximum entropy location shifts across Chain/Tree structures and determines the bottleneck.

#### 4. Why abrupt changes or spikes occur

For singleset MPS/TTN, abrupt changes in the maximum entropy curve often reflect a moving entanglement front. In the P3HT singleset data, the saved full profiles show about 8-9 switches of the maximum-entropy bond over 202 time points. Late-time maximum bonds become localized near specific bottleneck cuts, matching Li 2026's explanation.

For multiset MPS/TTN, there is an additional caveat. Because current `S_maxbond` ignores the population weight of each electronic component, a tiny-population component with a complex internal singular spectrum can dominate the scalar maximum. This is especially visible in some P3HT multiset TreeX curves, where early-time `S_maxbond` spikes are not proportional to the electronic entropy or population spread. These spikes should be described as worst-case conditional branch entropies, not as population-weighted physical entropy of the whole wavepacket.

#### 5. Why multiset TTN differs from singleset TTN/MPS

The same decomposition advantage applies to TTN. In singleset TTN, topology must simultaneously handle electronic delocalization, LE/CS branching, and vibrational correlations. In multiset TTN, each electronic state has its own nuclear TTN. This reduces the need for a single tree bond to carry incompatible nuclear branches.

The P3HT data show:

- multiset chain TTN `M=16`: `S_maxbond` grows to about `4.5`, indicating a poor topology and a strong bottleneck.
- multiset TreeX `M=16`: peak around `2.29`, much lower than multiset chain.
- multiset TreeX `M=32/64`: late-time values around `2.8-3.1`, comparable to or lower than converged singleset Tree/MPS references.
- multiset MPS from `multiset_202605`: late-time `S_maxbond` increases with `M`, indicating that smaller `M` suppresses entropy by truncation; the `M=64` late-time value around `3.6` is close to the singleset MPS `M=512` maximum entropy scale in Li 2026 data.

This supports a nuanced conclusion: multiset reduces electronic-branching entropy, while topology still matters strongly for the nuclear conditional wavepacket. Multiset is not automatically low-entanglement; it is low-entanglement only when the conditional nuclear topology matches the physical correlations.

### Limitations

1. Current scalar `S_maxbond` is not enough to locate the bottleneck.
   - Save `S_all` and `S_maxbond_eachset` for multiset MPS and TTN.
   - Also save the argmax electronic component and argmax bond index.

2. Current `S_maxbond` is unweighted by electronic population.
   - Add population-thresholded diagnostics, e.g. only include components with population `p_alpha > 1e-4` or `1e-3`.
   - Add population-weighted summaries, such as `sum_alpha p_alpha max_bond S_alpha`.

3. Singleset and multiset entropies are not the same mathematical observable.
   - Singleset bond entropy is a full-state bipartite entropy for a specific ordering/topology.
   - Multiset `S_maxbond` is a conditional branch entropy. The paper should explicitly state this to avoid overclaiming.

4. For publication, connect entropy with accuracy.
   - Plot `S_maxbond`, population error, and RMSD error on the same parameter sweep.
   - Use Li 2026's logic: maximum entropy is a stricter convergence criterion than population alone.

### What It Inspires for My Work

The multiset paper should include a dedicated "Entanglement structure and efficiency" section. Suggested claims:

1. Multiset does not merely reduce bond dimension empirically; it changes where the electron-vibrational entanglement is stored.
2. Large `S_el` with moderate conditional `S_maxbond` is the signature regime where multiset is most efficient.
3. Oscillatory and abrupt `S_maxbond` features should be interpreted as coherent vibronic breathing plus switching of the bottleneck branch/bond.
4. Multiset TTN confirms that once electronic branching is separated, remaining efficiency depends on nuclear topology; TreeX-like structures reduce top-layer bottlenecks.
5. Add population-aware entropy diagnostics to make the analysis robust enough for JCTC.

### Key Cited Papers

1. Li, Ren, and Yan, JCTC 2026, "A Further Comparison of MPS and TTNS..."
   - Central reference for using bond entropy to diagnose tensor-network bottlenecks and redesign topology.

2. Lindoy et al., 2025, pyTTN paper.
   - Important competing framework and direct P3HT:PCBM multiset/singleset MPS/TTN comparison.

3. Kloss et al., PRL 2019, multiset MPS Holstein paper.
   - Foundational multiset MPS ansatz and TDVP motivation for strong Holstein-type coupling.

4. Meyer, Manthe, Cederbaum, 1990; Wang and Thoss, 2003 ML-MCTDH lineage.
   - Historical origin of multi-set wavefunction representations and tree tensor-network interpretation.

5. Li et al., 2025 MS-VQD paper.
   - Useful physical explanation of multiset ansatz: different electronic surfaces should use separate nuclear ansatz blocks.

### Cross-Paper Synthesis

Kloss 2019 shows that multiset MPS is powerful for strong vibronic coupling because different electronic sites require different nuclear wavepackets. Lindoy 2025 and Li 2026 show that for P3HT:PCBM, tensor-network topology and bond entropy patterns determine whether MPS/TTN calculations converge efficiently. The present work can combine these two threads: multiset separates electronic branching, while MPS/TTN topology handles residual nuclear entanglement inside each branch.

This is stronger than saying "multiset has lower bond dimension." The mechanism is:

```text
singleset bottleneck = electronic branching + conditional nuclear entanglement + topology mismatch
multiset bottleneck = max conditional nuclear entanglement + topology mismatch
```

The removed term is precisely why multiset has superior performance in regimes with strong electron-phonon coupling and electronically dependent nuclear motion.

### Concrete Data Suggestions

For the JCTC manuscript, add the following diagnostics:

1. Save full entropy tensors:
   - `S_all[t, alpha, bond]`
   - `S_maxbond_eachset[t, alpha]`
   - `argmax_alpha[t]`
   - `argmax_bond[t]`

2. Add population-aware entropies:
   - thresholded max: `max_{p_alpha > eps, bond} S_alpha,bond`
   - weighted mean: `sum_alpha p_alpha max_bond S_alpha`
   - weighted distribution plots.

3. Add figure panels:
   - `S_el(t)` vs `S_maxbond(t)` for 1D Holstein.
   - `S_maxbond(t)` plus argmax component/bond color strip.
   - 2D heat map: alpha/electron-phonon coupling vs `Delta S = S_single - S_multi`.
   - P3HT:PCBM TreeX vs chain: maximum entropy and population error.

4. Add a convergence statement:
   - population/RMSD may look converged before `S_maxbond` is converged.
   - entropy should be used as a stricter diagnostic, following Li 2026.

## 中文报告

### 研究问题

Li、Ren、Yan 的 2026 JCTC 论文研究的是：为什么 P3HT:PCBM 激子解离模型中，singleset MPS 和 TTNS 在长时间极限会出现小但持续的差异。它的结论是，这个差异主要来自 bond dimension 不足，以及 bond dimension 在不同键上的分配不合理。文章进一步用键纠缠熵来指导 tensor-network 结构设计。

对你现在的 multiset 工作，真正对应的问题是：为什么 multiset MPS/TTN 可以用更小的内部键纠缠熵得到高精度动力学；以及目前看到的最大键纠缠熵曲线应该如何解释。

证据来源：
- Li 2026 Introduction：差异来自 bond dimension 不足，并且和纠缠熵分布有关。
- Li 2026 Section 2.2：用 singular values 定义 von Neumann entropy，并给出 `M >= exp(S)` 的经验下界。
- Li 2026 Figures 5 和 7：用最大键熵的空间分布和时间演化来设计 TreeX。
- 本地代码：`renormalizer/mps/mps.py::calc_bond_entropy`、`renormalizer/tn/tree.py::calc_bond_entropy`、`renormalizer/multiset/multiset_tdjob.py::_calc_bond_entropy_multiset`、`renormalizer/tn/multiset_ttn.py::calc_bond_entropy_summary`。

### 方法

Li 2026 的分析方法可以概括为三步。

1. 从每条虚拟键的 Schmidt singular values 计算键熵：

   ```text
   S = - sum_l s_l^2 log(s_l^2)
   ```

2. 把最大键熵理解成固定均匀 bond dimension 下的瓶颈。如果最大键熵是 `Smax`，那么有效 Schmidt rank 至少要达到 `exp(Smax)` 的量级。

3. 看最大键熵分布在哪些自由度之间，并据此改 tensor-network 拓扑。Li 2026 中，低频 OT modes 的熵最大，所以 TreeX 把这些 modes 更均匀地分配到顶层子树中。

对 multiset 数据，要额外注意一个定义差异：

```text
|Psi(t)> = sum_alpha |alpha> |psi_alpha(t)>
S_maxbond = max_alpha max_bond S(|psi_alpha(t)>)
```

也就是说，你现在保存的 multiset `S_maxbond` 不是整个电子+振动总波函数在某条 singleset 键上的纠缠熵，而是所有电子态条件振动态中最大的内部键熵。电子-振动纠缠本身由 `S_el` 单独记录。

### 数据

检查的数据包括：

- 1D long-range Holstein:
  - `multiset_202606/1DHolstein_nu16/plot.ipynb`
  - `multiset_202606/1DHolstein_nu16/Holstein31_alpha*/{multiset,singleset}/*.npz`
  - 主要指标：`S_maxbond`、`S_el`、`e_occupations`、由 population 重建的 RMSD。

- P3HT:PCBM multiset TTN/MPS:
  - `multiset_ttn/P3HT:PCBM/plot/plot_entropy.ipynb`
  - `multiset_ttn/P3HT:PCBM/multiset_treeX/p3ht_ms_ttn_treeX_{16,32,64}.npz`
  - `multiset_ttn/P3HT:PCBM/multiset_chain/p3ht_ms_ttn_chain_16.npz`
  - `multiset_202605/P3HT:PCBM/multiset/p3ht_multiset_{16,32,64}.npz`
  - singleset 参考：`multiset_ttn/P3HT:PCBM/2026Further_plot/figures/MPS.npz`、`Tree4.npz`。

代码结论：

- `calc_vn_entropy` 会先把 `sigma^2` 归一化，所以整体范数或 electronic population 不会影响单个分量的内部 bond entropy。
- 我用 conda `reno` 环境做了最小测试：同一组 Schmidt values 乘以 0.1、0.5、1、2 后，计算出的 entropy 完全相同。
- 当前 1D `.npz` 只存了 scalar `S_maxbond`，没有存 `S_all` 和 `S_maxbond_eachset`。代码已经支持这些量，后续建议保存，用于判断最大熵到底来自哪一个电子分量、哪一条键。

### 主要结果

#### 1. 为什么 multiset MPS 的最大键熵更低

singleset MPS 必须用一个张量网络同时表示所有电子态，以及这些电子态对应的不同振动波包。因此，一条链上的切分可能同时看到两类纠缠：

- 每个条件振动波包内部的核自由度纠缠；
- 不同电子态对应不同振动波包所造成的电子态条件分支纠缠。

multiset MPS 把第二部分从单个振动 MPS 中移出，显式写成电子态求和。每个 `|psi_alpha>` 只需要表示在电子态 `alpha` 条件下的振动波包。因此它的内部键熵更接近“单个条件振动波包的形状复杂度”，而不是“多个彼此不兼容的振动波包的总复杂度”。

这正好解释了 1D Holstein 数据：把最佳可用 multiset 曲线和 singleset `M=64` 对比，singleset 的 `S_maxbond` 通常高出 `Delta S ~ 0.4-0.9`，对应有效 Schmidt rank 比值 `exp(Delta S) ~ 1.5-2.5`。这个差异不是因为 multiset 某些电子分量 population 小，因为 entropy 计算已经消除了整体范数。

#### 2. 为什么 `S_el` 很大而 multiset `S_maxbond` 仍然不高

1D Holstein 中，`S_el` 可以达到 `2.8-3.1`，但 multiset 内部 `S_maxbond` 后期常在 `1.2-1.8` 附近。这不是矛盾。它说明总波函数有很强的电子-振动关联，但这种关联被显式电子态求和承载，而不是强迫某一条振动 MPS 键来承载。

论文中可以把这点写成核心机制：

```text
S_el 诊断电子-振动分支强度。
S_maxbond 诊断每个条件核波包的内部张量复杂度。
当 S_el 很大但条件分支 S_maxbond 不大时，multiset 最有优势。
```

#### 3. 为什么最大键熵会振荡

有两个来源。

物理来源：
- Holstein Franck-Condon 波包会随局域振动发生 breathing，并周期性回到共振区域。
- 电子 population 扩散、反射、再聚焦时，条件振动波包在空间切分两侧的可区分性会周期性增强或减弱。
- 因此 `S_maxbond` 会和 population/RMSD 一样出现振荡。

数值/诊断来源：
- `S_maxbond` 是对所有键取最大值；在 multiset 中还要对所有电子分量取最大值。
- 最大值对应的键或电子分量会随时间切换。即使每一条单独的键熵都是平滑的，取最大后的 scalar 曲线也可能出现 cusp 或突变。
- 这与 Li 2026 的解释一致：最大熵键的位置本身就是算法瓶颈的位置。

#### 4. 为什么会有突变或尖峰

对 singleset MPS/TTN，突变通常来自纠缠前沿移动。P3HT singleset 的完整 entropy profile 显示，在 202 个时间点中，最大熵键大约切换 8-9 次；后期最大熵稳定在特定瓶颈切分附近。

对 multiset MPS/TTN，还要额外小心：当前 `S_maxbond` 不按 electronic population 加权。因此，一个 population 很小但内部 singular spectrum 较复杂的电子分量，也可能支配全局 scalar `S_maxbond`。这可以解释 P3HT multiset TreeX 某些早期尖峰。这类尖峰更适合被称为“worst-case conditional branch entropy”，不应直接解释成主波包物理纠缠突然暴涨。

#### 5. 为什么 multiset TTN 和 singleset 的最大键熵差别很大

TTN 中也有同样的 multiset 分解优势。singleset TTN 必须让同一个树结构同时处理电子 delocalization、LE/CS branching 和核振动关联；multiset TTN 则让每个电子态有自己的核 TTN，所以电子分支不再强行压到同一棵树的某条键上。

P3HT 数据说明：

- multiset chain TTN `M=16` 的 `S_maxbond` 增长到约 `4.5`，说明链式结构很差，存在强瓶颈。
- multiset TreeX `M=16` 峰值约 `2.29`，远低于 chain。
- multiset TreeX `M=32/64` 后期约 `2.8-3.1`，与 singleset Tree/MPS 的收敛参考相当或更低。
- `multiset_202605` 中的 multiset MPS 后期 `S_maxbond` 随 `M` 上升，说明小 `M` 曲线会因为截断而人为压低熵；`M=64` 后期约 `3.6`，接近 Li 2026 singleset MPS `M=512` 的最大键熵尺度。

所以结论要写得细致：multiset 降低的是电子分支造成的纠缠压力；剩余的条件核波包纠缠仍然依赖 MPS ordering 或 TTN topology。multiset 不是自动低熵，只有当条件核波包的 topology 合理时才低熵。

### 局限

1. 当前 scalar `S_maxbond` 不能定位瓶颈。
   - 建议保存 `S_all`、`S_maxbond_eachset`、`argmax_alpha`、`argmax_bond`。

2. 当前 `S_maxbond` 不按 population 加权。
   - 建议增加 thresholded max，例如只统计 `p_alpha > 1e-4` 或 `1e-3` 的分量。
   - 建议增加 weighted mean，例如 `sum_alpha p_alpha max_bond S_alpha`。

3. singleset 和 multiset 的 bond entropy 不是同一个数学量。
   - singleset bond entropy 是完整总波函数在某种 ordering/topology 下的 bipartite entropy。
   - multiset `S_maxbond` 是条件分支内部 entropy。
   - 论文中必须明说，否则容易被审稿人质疑。

4. 需要把 entropy 和 accuracy 连接起来。
   - 建议把 `S_maxbond`、population error、RMSD error 放到同一组参数扫描里。
   - 按 Li 2026 的逻辑，最大键熵通常比 population 本身更严格地反映收敛性。

### 对你论文的启发

JCTC 主文中建议单独加入一节：

```text
Entanglement structure and efficiency of the multiset ansatz
```

可以提出以下观点：

1. multiset 不是经验性降低 bond dimension，而是改变了电子-振动纠缠的存储位置。
2. `S_el` 大、但条件分支 `S_maxbond` 中等，是 multiset 最有效的特征信号。
3. `S_maxbond` 的振荡来自 coherent vibronic breathing；突变主要来自最大键/最大电子分量切换。
4. multiset TTN 说明：电子分支分离之后，剩下的效率仍取决于核自由度 topology。
5. 为达到 JCTC 级别严谨性，需要补充 population-aware entropy diagnostics。

### 关键参考文献

1. Li, Ren, Yan, JCTC 2026.
   - 用 bond entropy 诊断 tensor-network 瓶颈并设计 topology，是你的熵分析直接模板。

2. Lindoy et al., 2025 pyTTN.
   - 竞争框架，并且直接讨论 P3HT:PCBM 中 multiset/singleset MPS/TTN。

3. Kloss et al., PRL 2019.
   - multiset MPS 和强 Holstein coupling 动力学的基础文献。

4. Meyer/Manthe/Cederbaum 以及 Wang/Thoss 的 MCTDH/ML-MCTDH 传统。
   - multiset wavefunction 和 tree tensor-network 解释的历史来源。

5. Li et al., 2025 MS-VQD.
   - 对 multiset ansatz 的物理解释很清楚：不同电子面应该使用不同核波包 ansatz。

### 跨文献综合

Kloss 2019 说明，在强电声耦合下，不同电子位置对应不同振动波包，multiset MPS 因此很有效。Li 2026 和 Lindoy 2025 说明，在 P3HT:PCBM 这类复杂模型中，tensor-network topology 和键熵分布决定 MPS/TTN 的收敛效率。你的工作可以把这两条线统一起来：

```text
singleset 瓶颈 = 电子分支纠缠 + 条件核波包内部纠缠 + topology mismatch
multiset 瓶颈 = 条件核波包内部最大纠缠 + topology mismatch
```

multiset 去掉的正是电子态分支这部分张量复杂度。这就是为什么它在强电声耦合、短程电子耦合、电子态依赖核波包差异显著的区域表现好。

### 具体数据补充建议

1. 保存完整 entropy tensor：
   - `S_all[t, alpha, bond]`
   - `S_maxbond_eachset[t, alpha]`
   - `argmax_alpha[t]`
   - `argmax_bond[t]`

2. 增加 population-aware 熵：
   - thresholded max：`max_{p_alpha > eps, bond} S_alpha,bond`
   - weighted mean：`sum_alpha p_alpha max_bond S_alpha`
   - 分布图：population vs conditional entropy。

3. 主文图建议：
   - 1D Holstein: `S_el(t)` 与 `S_maxbond(t)` 同图。
   - 1D Holstein: `Delta S = S_single - S_multi` 参数图。
   - P3HT: chain vs TreeX 的 `S_maxbond` 与 population error 对照。
   - `argmax_alpha/bond` 色条，解释突变来自瓶颈切换。

4. 收敛性陈述：
   - population/RMSD 看似收敛，不代表 bond entropy 已经收敛。
   - 最大键熵应作为比 population 更严格的 convergence diagnostic。

