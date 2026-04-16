# 有限温初态制备原理说明

本文档说明当前 `test/test_fmo_300k/` 下有限温初态的物理思路，以及 `singleset` 与 `multiset` 两套实现目前的差异。

## 1. 总体目标

有限温计算的目标不是直接传播零温纯态，而是传播一个能够表示热平衡密度矩阵

\[
\rho(\beta)=\frac{e^{-\beta H}}{Z}
\]

的纯化态（purified state）。这里

- \(\beta = 1/(k_B T)\)
- \(Z = \mathrm{Tr}(e^{-\beta H})\)

在热场双态/纯化表述中，通常引入一个辅助空间，定义无穷温最大纠缠态

\[
|I\rangle = \sum_n |n\rangle \otimes |\tilde n\rangle
\]

然后通过

\[
|\Psi_\beta\rangle \propto e^{-\beta H/2}|I\rangle
\]

得到有限温纯化态。对辅助自由度做迹操作后，就恢复有限温密度矩阵。

## 2. `singleset` 的实现

`singleset` 的实现路径在 [dynamics.py](../../renormalizer/transport/dynamics.py#L173)。

它的流程是：

1. 用 `MpDm.max_entangled_gs(self.model)` 构造无穷温最大纠缠态。
2. 调用 `ThermalProp(gs_mp, exact=True, space="GS")`。
3. 通过虚时间传播

   \[
   e^{-\beta H/2}
   \]

   将最大纠缠态冷却到目标温度。
4. 得到有限温基态振动态后，再施加电子激发，进入后续 TDVP 演化。

也就是说，`singleset` 走的是“先无穷温，再显式虚时间冷却”的数值流程。

## 3. `multiset` 当前的参考初态模型

当前 `multiset` 会先从完整模型中抽取一个仅用于热平衡初态制备的参考模型 `init_model`，对应代码在 [multiset_model.py](../../renormalizer/model/multiset_model.py#L263)。

这个 `init_model` 的构造方式是：

1. 从原始 Holstein Hamiltonian 中只保留 `len(op.dofs) == 1` 的项。
2. 对含有 `a^\dagger a` 的项，用 `_reset_all_MsOp` 将电子投影替换为单位算符。
3. 最终只留下纯振动的局域项。

因此，当前 `multiset` 的热平衡参考面本质上是“一组彼此独立的局域谐振子”。

## 4. `multiset` 当前有限温初态的实现

当前 `multiset` 的有限温初态在 [multiset_model.py](../../renormalizer/model/multiset_model.py#L95) 构造。

它没有显式调用：

- `MpDm.max_entangled_gs(...)`
- `ThermalProp(...).evolve(...)`

而是直接写出了虚时间冷却之后的解析结果。

具体来说，对于单个谐振子

\[
H_\lambda = \omega_\lambda b^\dagger_\lambda b_\lambda
\]

从最大纠缠态

\[
|I_\lambda\rangle = \sum_n |n\rangle \otimes |\tilde n\rangle
\]

经过虚时间冷却后得到

\[
|\Psi_{\beta,\lambda}\rangle \propto
\sum_n e^{-\beta \omega_\lambda n /2}|n\rangle \otimes |\tilde n\rangle
\]

因此局域系数就是

\[
c_n \propto e^{-\beta \omega_\lambda n/2}
\]

代码中正是按这个公式构造：

```python
weights = np.exp(-0.5 * beta * basis.omega * np.arange(basis.nbas, dtype=float))
weights /= np.linalg.norm(weights)
```

然后将这些局域系数作为 Hartree-product 纯化态写入 `Mps.hartree_product_state(...)`，最后再通过 `MpDm.from_mps(...)` 转成有限温初态。

## 5. 为什么当前实现与“虚时间冷却”在这里等价

关键原因是：当前 `multiset` 的 `init_model` 是纯局域、可分离的谐振子哈密顿量。

对这种模型，

\[
e^{-\beta H/2}|I\rangle
\]

可以逐模分解，而且每个模的结果都能解析写出。因此：

- `singleset` 中“先构造最大纠缠态，再虚时间冷却”
- `multiset` 当前“直接写出冷却后的局域热权重”

在当前 `init_model` 下是数学等价的。

换句话说，`multiset` 现在不是换了另一套物理，而是把虚时间演化的终态直接解析写出来了。

## 6. 为什么没有直接复用 `ThermalProp(exact=True)`

之前尝试过沿用 `singleset` 的接口，但 `multiset` 的 `init_model` 是普通 `Model`，而 `ThermalProp(exact=True)` 内部会走 `Mpo.exact_propagator(...)`，这个实现当前是按 `HolsteinModel` 写的，默认模型：

- 可按分子迭代
- 保留电子站点结构

这与 `multiset` 当前的纯 phonon `Model` 不兼容。

因此目前 `multiset` 采用了“直接构造解析终态”的方式来避免该接口不匹配。

## 7. 当前实现的适用范围

当前实现成立的前提是：

1. 有限温参考哈密顿量是纯局域振动模型。
2. 各模之间没有需要在初态中显式处理的非局域耦合。
3. 每个模都可以视为独立谐振子。

因此它对当前 FMO `multiset` 初态制备是合理的。

但如果未来 `init_model` 不再是纯局域谐振子，例如：

- 出现模间耦合
- 出现更一般的非对角振动项
- 希望严格复刻 `singleset` 的数值流程

那么更自然的做法仍然是实现真正的：

1. 无穷温最大纠缠态构造
2. 虚时间传播冷却

并让这条路径能直接支持 `multiset` 的纯 phonon `Model`。

## 8. 与后续 `multiset` 动力学的衔接

得到有限温初态后，`multiset` 仍然按原有流程继续：

1. 在 `cdd_init_mps()` 中做 Franck-Condon 激发。
2. 计算初始能量并重设 MPO offset。
3. 扩展 bond dimension。
4. 进入 `multiset` TDVP 演化。

由于有限温初态是 `MpDm` 形式，因此当前 `multiset` 的 batched hop 也已经额外支持了四阶局域张量。

## 9. 一句话总结

`singleset` 当前是“显式虚时间冷却”；  
`multiset` 当前是“在纯局域谐振子参考模型上，直接写出虚时间冷却后的解析纯化态”。

对当前 FMO 的 `multiset` 初态模型，这两种做法在物理上是等价的。
