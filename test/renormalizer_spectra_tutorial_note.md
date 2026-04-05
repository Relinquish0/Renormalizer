# Renormalizer 光谱计算方法教程式笔记

## 1. 引言

这份笔记面向希望结合代码与理论理解 `renormalizer` 中光谱计算实现的读者，重点分析：

- `renormalizer/spectra` 中的三类核心 case
- `renormalizer/cv` 中的 correction vector 光谱方法
- 其他与“光谱”密切相关的模块，如 `transport/spectral_function.py` 和 `mps/tda.py`

本文尽量按“先理论、再代码、再数值流程”的顺序展开，让读者既能知道这些类在算什么，也能知道代码为什么这样组织。

---

## 2. 光谱计算的总框架

在分子振动-电子耦合模型中，吸收谱和发射谱通常都可以写成某种二阶偶极响应函数。对 `renormalizer` 来说，最核心的想法有两条路线：

### 2.1 时间域路线

先构造一个被偶极算符作用后的初态，再做实时间传播，采样自相关函数：

\[
C(t) = \langle \Psi | \hat\mu^\dagger e^{-i\hat H t} \hat\mu | \Psi \rangle
\]

或者对应的发射版本。随后再对 `C(t)` 做傅里叶变换得到频谱。

这条路线在 `renormalizer/spectra` 中实现，是一个典型的 TD-DMRG / TD-MPS 光谱工作流。

### 2.2 频域路线

不显式做长时间传播，而是直接在每个频率点 `\omega` 上解一个校正矢量方程，例如：

\[
\left[(\hat H - E_0 - \omega)^2 + \eta^2\right] |X(\omega)\rangle
=
-\eta \hat\mu |\Psi_0\rangle
\]

其中 `\eta` 是 Lorentzian 展宽参数。

这条路线在 `renormalizer/cv` 中实现，对应 DDMRG / correction vector 思路。

### 2.3 两类路线的物理差别

- 时间域法更像“先得到动力学，再由动力学恢复频谱”。
- 频域法更像“直接针对目标频率求响应”。

二者求的是同一类线性响应信息，只是数值组织方式不同。

---

## 3. 代码公共骨架：光谱任务如何被组织

在理解 `spectra/` 和 `cv/` 之前，先看几个底层公共类。

### 3.1 `TdMpsJob`：时间演化任务模板

文件：[`renormalizer/utils/tdmps.py`](/curie-home/zengjj/Renormalizer/renormalizer/utils/tdmps.py#L19)

`TdMpsJob` 是一个很关键的抽象基类。它定义了时间演化任务的统一流程：

1. `init_mps()`：准备初态
2. `evolve_single_step(dt)`：做一步传播
3. `process_mps(mps)`：对新态做观测量采样
4. `evolve(...)`：把上面三件事按时间循环组织起来

也就是说，`spectra/` 里的类并不需要自己重写整个时间循环，它们只需要定义：

- 初态怎么构造
- 一步怎么传播
- 每一步记录什么量

这就是 `spectra` 子模块代码能保持简洁的原因。

### 3.2 `BraKetPair`：把相关函数写成 bra/ket 重叠

文件：[`renormalizer/mps/mps.py`](/curie-home/zengjj/Renormalizer/renormalizer/mps/mps.py#L2061)

`BraKetPair` 保存三件事：

- `bra_mps`
- `ket_mps`
- `ft`

其中 `ft = calc_ft()`，本质上就是每一步时间点的相关函数值。默认情形下：

\[
f(t) = \langle \mathrm{bra}(t) | \mathrm{ket}(t) \rangle
\]

如果传入了 `mpo`，则也可以是带算符的期望值。

所以在 `spectra` 的实现里，所谓“算自相关函数”，实际上就是不断更新 bra/ket，然后读取 `BraKetPair.ft`。

### 3.3 `SpectraTdMpsJobBase`：把 `ft` 收集成自相关函数

文件：[`renormalizer/spectra/base.py`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/base.py#L11)

`SpectraTdMpsJobBase` 做的事情非常直接：

- 构造 Hamiltonian MPO：`self.h_mpo = Mpo(model, offset=offset)`
- 根据 `spectratype` 决定是吸收还是发射
- 在 `process_mps()` 中把 `braket_pair.ft` 追加到 `_autocorr`

因此 `autocorr` 就是整个计算最直接的输出。

### 3.4 `ThermalProp`：有限温初态的虚时传播器

文件：[`renormalizer/mps/thermalprop.py`](/curie-home/zengjj/Renormalizer/renormalizer/mps/thermalprop.py#L13)

有限温光谱的关键在于如何构造热态。`renormalizer` 采用纯化/热场思路：

1. 先构造一个最大纠缠的 MpDm
2. 再做虚时间传播 `e^{-\beta H / 2}`
3. 从而得到表示有限温密度矩阵的纯化态

这一步统一由 `ThermalProp` 完成。它支持：

- `exact=True`：在局域可精确传播的情形下直接用精确传播子
- `exact=False`：走一般的 MPS/MPO 虚时演化

这对后面理解 `SpectraFiniteT` 和 `SpectraFtCV` 很关键。

---

## 4. `renormalizer/spectra`：时间域光谱方法逐步分析

### 4.1 模块总览

文件：[`renormalizer/spectra/__init__.py`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/__init__.py)

对外暴露的类是：

- `SpectraExact`
- `SpectraFiniteT`
- `SpectraOneWayPropZeroT`
- `SpectraTwoWayPropZeroT`

如果按方法学归类，可以整理成三类：

1. 零温 TD-DMRG：`SpectraOneWayPropZeroT` / `SpectraTwoWayPropZeroT`
2. 有限温 TD-DMRG：`SpectraFiniteT`
3. 精确传播参考实现：`SpectraExact`

下面逐个分析。

---

## 5. 零温时间域光谱：`SpectraOneWayPropZeroT` 与 `SpectraTwoWayPropZeroT`

文件：[`renormalizer/spectra/zerot.py`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/zerot.py#L15)

### 5.1 物理目标

零温吸收谱和发射谱都可由零温相关函数得到。

对于吸收，一种典型写法是：

\[
C_{\mathrm{abs}}(t) = \langle \Psi_0^{\mathrm{GS}} |
\hat\mu^\dagger e^{-i\hat H t}\hat\mu
| \Psi_0^{\mathrm{GS}} \rangle
\]

对于发射，则参考态通常在激发态势能面上：

\[
C_{\mathrm{emi}}(t) = \langle \Psi_0^{\mathrm{EX}} |
\hat\mu e^{-i\hat H t}\hat\mu^\dagger
| \Psi_0^{\mathrm{EX}} \rangle
\]

代码中并没有把这些公式直接写出来，而是通过“构造偶极作用后的 MPS 并传播”来实现。

### 5.2 先确定参考态属于哪个激子子空间

在 [`SpectraTdMpsJobBase.__init__`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/base.py#L12) 中：

- `emi` 对应 `self.nexciton = 1`
- `abs` 对应 `self.nexciton = 0`

这一步非常重要。它说明：

- 零温发射是从单激子子空间中的参考态出发
- 零温吸收是从零激子子空间中的基态出发

### 5.3 如何构造参考态

`SpectraZeroT.get_imps()` 在 [`zerot.py:60`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/zerot.py#L60)：

1. 用 `Mps.random(...)` 生成一个随机初猜
2. 调用 `gs.optimize_mps(i_mps, self.h_mpo)` 做 DMRG 基态优化

这里的“基态”不是整个全空间无条件的最低能态，而是对应 `self.nexciton` 所指定粒子数/激子数子空间内的最低态。

因此：

- 吸收时得到的是 GS 子空间基态
- 发射时得到的是 EX 子空间基态

### 5.4 如何构造被偶极算符作用后的态

`SpectraZeroT.init_mps()` 在 [`zerot.py:46`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/zerot.py#L46)：

- 发射时使用 `operator = "a"`
- 吸收时使用 `operator = r"a^\dagger"`

然后：

```python
dipole_mpo = Mpo.onsite(self.model, operator, dipole=True)
a_ket_mps = dipole_mpo.apply(self.get_imps(), canonicalise=True)
```

这一步对应的就是理论里的 `\hat\mu |\Psi\rangle`。

之后代码把 `a_ket_mps` 归一化，并复制出一个 `a_bra_mps`，组成初始 `BraKetPair`。

### 5.5 `OneWay` 与 `TwoWay` 的区别

#### 5.5.1 `SpectraOneWayPropZeroT`

文件位置：[`zerot.py:68`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/zerot.py#L68)

每一步只传播 ket：

```python
latest_ket_mps = latest_ket_mps.evolve(self.h_mpo, evolve_dt)
```

而 bra 固定不动。于是相关函数的采样形式近似是：

\[
C(t) \sim \langle \phi(0) | \phi(t) \rangle
\]

这就是最直接的一侧传播方案。

#### 5.5.2 `SpectraTwoWayPropZeroT`

文件位置：[`zerot.py:75`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/zerot.py#L75)

这里 bra 和 ket 交替传播：

- 奇数步：推进 ket 的 `+dt`
- 偶数步：推进 bra 的 `-dt`

这样做的直观理解是：把时间相位平均分摊到 bra 和 ket 两边，可以减轻单边传播时某些相位震荡和误差累积问题。它是一种对称的相关函数采样方式。

### 5.6 数值层面看，这一类方法在做什么

把代码串起来，流程非常清楚：

1. 在指定激子子空间求参考态
2. 作用偶极算符构造 `\mu|\Psi\rangle`
3. 在 Hamiltonian 下做实时间传播
4. 每一步计算 bra/ket 重叠，记录到 `autocorr`
5. 用户在外部再对 `autocorr(t)` 做傅里叶变换得到谱线

### 5.7 测试说明了什么

测试文件：[`renormalizer/spectra/tests/test_spectra.py`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/tests/test_spectra.py#L37)

可以看到：

- `test_zero_t_abs()` 同时测试 one-way 和 two-way
- `test_zero_t_emi()` 也同时测试两种传播
- 都是把 `autocorr` 与事先保存的标准结果比较

这再次说明 `spectra/` 的核心目标是生成正确的时间域相关函数。

---

## 6. 有限温时间域光谱：`SpectraFiniteT`

文件：[`renormalizer/spectra/finitet.py`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/finitet.py#L25)

### 6.1 理论背景

有限温光谱的核心对象不是单一波函数，而是热密度矩阵：

\[
\rho_\beta = e^{-\beta \hat H}
\]

于是相关函数变成热平均：

\[
C(t)=\mathrm{Tr}\left[
\rho_\beta \hat\mu(t)\hat\mu(0)
\right]
\]

在张量网络里，常见做法是把热密度矩阵纯化为一个更大的纯态，然后在这个纯化态上做传播。`renormalizer` 正是这样做的。

### 6.2 发射情形：如何构造有限温初态

`init_mps_emi()` 位于 [`finitet.py:67`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/finitet.py#L67)。

它的步骤可以翻译成：

1. 构造 EX 子空间最大纠缠初态 `MpDm.max_entangled_ex(self.model)`
2. 用 `ThermalProp` 做半个逆温 `\beta/2` 的虚时传播
3. 得到 `e^{-\beta H/2}` 作用后的纯化热态
4. 再施加偶极算符的共轭转置 `dipole_mpo_dagger`

代码里最关键的片段是：

```python
tp.evolve(None, self.insteps, self.temperature.to_beta() / 2j)
ket_mpo = tp.latest_mps
a_ket_mpo = ket_mpo.apply(dipole_mpo_dagger, canonicalise=True)
```

这里的 `/ 2j` 看起来像复数写法，实际上是用统一的演化接口把虚时传播写进 `TdMpsJob` 框架中。

### 6.3 吸收情形：如何构造有限温初态

`init_mps_abs()` 位于 [`finitet.py:124`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/finitet.py#L124)。

过程是：

1. 构造 GS 子空间最大纠缠初态 `MpDm.max_entangled_gs`
2. 在 `space="GS"` 下精确虚时传播
3. 再作用 `a^\dagger`

也就是说：

- 有限温吸收从 GS 热分布出发
- 有限温发射从 EX 热分布出发

这在物理上是很自然的。

### 6.4 为什么吸收和发射对应的 `BraKetPair` 不一样

在 [`finitet.py:16`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/finitet.py#L16) 中有两个子类：

- `BraKetPairEmiFiniteT`
- `BraKetPairAbsFiniteT`

其中发射版本重写了 `calc_ft()`，返回共轭：

```python
return np.conj(super(...).calc_ft())
```

这说明有限温发射在当前实现下，其 bra/ket 组织方式会导致相关函数自然带一个共轭关系，代码用这个小包装把物理上需要的最终方向修正回来。

### 6.5 实时间传播做了什么

`evolve_single_step()` 位于 [`finitet.py:140`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/finitet.py#L140)。

它并不是简单地只传播 ket，而是：

- 一侧做 `evolve_exact(..., "GS")`
- 一侧做一般 `evolve(self.h_mpo, dt)`
- 并在奇偶步之间切换 bra/ket

这可以理解成：把局域、易处理的那部分 GS 相位因子单独精确提出，以减少相关函数中纯相位震荡，从而改善数值稳定性和谱线后处理质量。

### 6.6 `stop_evolve_criteria()` 的含义

在 [`finitet.py:116`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/finitet.py#L116)，代码设定：

- 如果最后 10 个点的平均值和波动都远小于初值
- 就认为相关函数已经衰减到足够小，可以停止传播

这是一种很实用的有限温时间域截断准则。

### 6.7 小结

`SpectraFiniteT` 的思想可以概括为：

1. 用纯化态表示有限温密度矩阵
2. 用虚时传播生成热初态
3. 用偶极算符激发该热态
4. 用实时间传播采样热相关函数

这就是有限温 TD-DMRG 光谱算法的典型实现。

---

## 7. 精确传播版本：`SpectraExact`

文件：[`renormalizer/spectra/exact.py`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/exact.py#L15)

### 7.1 这个类的定位

`SpectraExact` 不是一般大体系的主力算法，而更像是：

- 在某些局域可精确传播的模型上提供基准
- 在简单模型上作为 TD-DMRG 结果的参考

代码注释已经写明：

- 对单分子情形，某些 EX/GS 空间传播是局域且可精确的
- bra 侧某些相位可忽略，以减少振荡

### 7.2 初态构造仍然延续同一逻辑

`init_mps()` 在 [`exact.py:65`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/exact.py#L65)：

1. 先在对应子空间求参考态
2. 再根据吸收/发射施加 `a^\dagger` 或 `a`

因此从物理流程看，它与 `SpectraZeroT` 是一致的，差别只在“传播器如何实现”。

### 7.3 时间传播为何叫 exact

`evolve_single_step()` 在 [`exact.py:97`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/exact.py#L97)：

```python
latest_ket_mps = latest_ket_mps.evolve_exact(self.h_mpo, evolve_dt, self.space2)
```

也就是说，它直接调用 `evolve_exact` 而不是通用 TDVP / Trotter 风格传播。

这只有在 Hamiltonian 结构足够特殊时才可行，因此它的适用范围比一般 TD-DMRG 要窄。

### 7.4 一个需要注意的实现事实

文档字符串提到它支持某些有限温情况，但当前构造函数中有：

```python
assert temperature == 0
```

位置见 [`exact.py:40`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/exact.py#L40)。

因此从当前代码实际行为看，它的可靠使用场景应理解为“以零温、尤其简单体系的 exact 传播参考为主”。

### 7.5 测试中的角色

在测试 [`test_zero_exact_emi()`](/curie-home/zengjj/Renormalizer/renormalizer/spectra/tests/test_spectra.py#L19) 中，
`SpectraExact` 被用于验证零温发射相关函数。这说明它在项目中承担了重要的基准作用。

---

## 8. `renormalizer/cv`：频域 correction vector / DDMRG 方法

### 8.1 总体思想

`cv` 模块不从时间相关函数出发，而是直接针对每个频率 `\omega` 求响应。

这类方法的优点是：

- 不需要很长时间传播
- 对窄展宽和高分辨定点频率响应很有吸引力

代价是：

- 每个频率点都要解一个大线性方程
- 如果频率网格很密，总成本也会很高

### 8.2 公共骨架 `SpectraCv`

文件：[`renormalizer/cv/spectra_cv.py`](/curie-home/zengjj/Renormalizer/renormalizer/cv/spectra_cv.py#L52)

`SpectraCv` 抽象出 correction vector 方法的通用流程：

1. 构造右端项 `b_mps`
2. 构造 correction vector 初猜 `cv_mps`
3. 对某个 `\omega` 调用 `oper_prepare(omega)`
4. 进行 sweep 式局域优化 `optimize_cv(...)`
5. 从泛函值 `l_value` 恢复频谱强度

在 [`cv_solve()`](/curie-home/zengjj/Renormalizer/renormalizer/cv/spectra_cv.py#L121) 中，最终响应写成：

\[
I(\omega) = -\frac{1}{\pi \eta} L
\]

这里 `L` 是 sweep 中最优的泛函值。

---

## 9. 零温 correction vector：`SpectraZtCV`

文件：[`renormalizer/cv/zerot.py`](/curie-home/zengjj/Renormalizer/renormalizer/cv/zerot.py#L25)

### 9.1 右端项 `b` 如何构造

`init_b_mps()` 位于 [`zerot.py:80`](/curie-home/zengjj/Renormalizer/renormalizer/cv/zerot.py#L80)。

它先：

1. 在指定激子子空间求参考态
2. 记录基态能量 `e0`
3. 构造

\[
b = -\eta \hat\mu |\Psi_0\rangle
\]

代码里正是：

```python
b_mps = dipole_mpo.apply(mps.scale(-self.eta))
```

### 9.2 频率相关算符如何构造

`oper_prepare()` 位于 [`zerot.py:124`](/curie-home/zengjj/Renormalizer/renormalizer/cv/zerot.py#L124)。

代码设置：

\[
A = H - E_0 - \omega
\]

即：

```python
identity = Mpo.identity(self.model).scale(-self.e0 - omega)
self.a_oper = self.h_mpo.add(identity)
```

### 9.3 局域优化的物理意义

`optimize_cv()` 的注释非常关键，位置在 [`zerot.py:129`](/curie-home/zengjj/Renormalizer/renormalizer/cv/zerot.py#L129)。

它说明这里优化的是一个工作泛函 `L`，从而把 correction vector 方程转化为一个 sweep 过程中的局域线性代数问题。实现上可以理解为：

\[
\left(A^2 + \eta^2\right)X = b
\]

然后通过共轭梯度 `scipy.sparse.linalg.cg` 解局域化后的线性方程。

### 9.4 预条件和局域更新

该类实现里还有两个重要数值细节：

- 从 `A^2 + \eta^2` 的对角部分构造 preconditioner
- 每个局域步更新 correction vector 的张量，再通过 SVD 截断保持 bond dimension

所以它在算法结构上和普通 DMRG sweep 很像，只不过优化对象不再是基态能量，而是响应方程的解。

### 9.5 `1site` 与 `2site`

`SpectraZtCV` 支持：

- `method="1site"`
- `method="2site"`

其中 `2site` 需要更复杂的张量收缩，代码里还专门用 `opt_einsum` 做路径缓存。  
从实现风格看，作者希望保留 `2site` 的灵活性，但更常规、更稳的用法大概仍是 `1site`。

---

## 10. 有限温 correction vector：`SpectraFtCV`

文件：[`renormalizer/cv/finitet.py`](/curie-home/zengjj/Renormalizer/renormalizer/cv/finitet.py#L30)

### 10.1 与零温 CV 的本质区别

零温 CV 里的对象还是普通 MPS；  
有限温 CV 里，correction vector 和右端项都变成了 MPO/纯化空间对象。

所以这里实际上是在“李维尔空间”或纯化空间里做频域响应。

### 10.2 correction vector 初猜如何构造

`init_cv_mpo()` 位于 [`finitet.py:104`](/curie-home/zengjj/Renormalizer/renormalizer/cv/finitet.py#L104)。

它调用：

```python
Mpo.finiteT_cv(self.model, 1, self.m_max, self.spectratype, percent=1.0)
```

这说明有限温 correction vector 已经不再是普通 ket，而是为有限温响应问题专门设计的 MPO 结构。

### 10.3 右端项热态的构造

`init_b_mpo()` 位于 [`finitet.py:111`](/curie-home/zengjj/Renormalizer/renormalizer/cv/finitet.py#L111)。

它和 `SpectraFiniteT` 的思路非常像：

- 吸收：从 GS 热态出发
- 发射：从 EX 热态出发
- 都通过 `ThermalProp` 构造热初态
- 再乘上 `-\eta \mu`

因此可以说：

- `SpectraFiniteT` 是“有限温时间域法”
- `SpectraFtCV` 是“有限温频域法”

二者共享同样的热态准备思想。

### 10.4 量子数块的物理意义

`optimize_cv()` 在 [`finitet.py:165`](/curie-home/zengjj/Renormalizer/renormalizer/cv/finitet.py#L165) 明确区分：

- 吸收对应 `|1><0|`
- 发射对应 `|0><1|`

这不是一个纯数值细节，而是非常有物理意义的实现：

- 有限温吸收对应“从零激发热分布被激发到单激发块”
- 有限温发射对应“从单激发热分布退激发回零激发块”

所以这里的量子数筛选就是把正确的跃迁块抽取出来。

### 10.5 为什么作者不推荐 2-site 有限温 CV

在 [`update_LR()`](/curie-home/zengjj/Renormalizer/renormalizer/cv/finitet.py#L349) 后面直接写了：

```python
# 2site for finite temperature is too expensive, so I drop it
raise NotImplementedError
```

这非常能说明问题：有限温 correction vector 在张量维数和收缩复杂度上都明显更重，因此目前实际支持的主路线是 `1site`。

---

## 11. `cv` 测试反映了怎样的使用方式

测试文件：

- [`renormalizer/cv/tests/test_abs.py`](/curie-home/zengjj/Renormalizer/renormalizer/cv/tests/test_abs.py#L18)
- [`renormalizer/cv/tests/test_emi.py`](/curie-home/zengjj/Renormalizer/renormalizer/cv/tests/test_emi.py#L16)

可以看到一个典型工作流：

1. 准备一个频率网格 `freq_reg`
2. 选取若干代表性频率点
3. 调用 `batch_run(test_freq, cores, spectra)`

这说明 `cv` 模块的典型使用方式就是扫频。

从数值习惯上也能看到：

- 零温常用很小的 `eta`
- 有限温时常为了稳定性使用稍大的 `eta`
- 某些测试会对 Hamiltonian 做能量 offset，以帮助 CG 收敛

这也是频域方法的典型经验。

---

## 12. 其他与光谱密切相关的方法

虽然用户这次重点是 `spectra` 和 `cv`，但库里还有两个相关度很高的模块。

### 12.1 `transport/spectral_function.py`

文件：[`renormalizer/transport/spectral_function.py`](/curie-home/zengjj/Renormalizer/renormalizer/transport/spectral_function.py#L15)

这个模块算的是零温单粒子 retarded Green's function：

\[
iG_{ij}(t)=\langle 0| c_i(t)c_j^\dagger |0\rangle
\]

然后再转到 `k` 空间：

\[
A(k,\omega) = -\frac{1}{\pi}\mathrm{Im}\int_0^\infty dt\, e^{i\omega t} G_k(t)
\]

它与 `spectra/` 的共同点非常强：

1. 先构造 `c^\dagger |0\rangle`
2. 再实时间传播
3. 然后采样重叠或期望值
4. 最后由外部做频域变换

所以从算法家族上看，`SpectralFunctionZT` 可以被视为“时间域相关函数法”在输运谱函数中的对应版本。

### 12.2 `mps/tda.py`

文件：[`renormalizer/mps/tda.py`](/curie-home/zengjj/Renormalizer/renormalizer/mps/tda.py#L15)

`TDA` 严格说不直接输出完整谱线，而是求激发态能量和切空间激发向量。  
它在光谱研究中的意义在于：

- 可以给出吸收峰位置的离散近似
- 可以作为频谱分析时的激发态参照

如果说：

- `spectra/` 和 `cv/` 更关注“响应函数与线形”
- 那么 `TDA` 更关注“激发态能量结构”

二者互补，但不等价。

---

## 13. 统一视角：这些方法之间到底是什么关系

现在把前面的内容统一起来，可以得到一个很清晰的图景。

### 13.1 `spectra/` 与 `cv/` 的关系

两者都在求线性光谱响应，只是数值路径不同：

- `spectra/`：先求 `C(t)`，再傅里叶变换
- `cv/`：直接求 `I(\omega)`

所以它们不是两类不同物理问题，而是同一问题的两种数值实现。

### 13.2 零温与有限温的关系

区别不在于“后续传播是否完全不同”，而主要在于：

- 零温：参考对象是单个参考态
- 有限温：参考对象是热密度矩阵或其纯化表示

于是有限温代码里就多出了：

- `MpDm`
- `ThermalProp`
- 热态缓存与虚时传播

### 13.3 `Exact` 与一般 TD-DMRG 的关系

`SpectraExact` 是在特定结构下用精确传播子取代一般时间演化器。  
它更像是：

- 特殊模型上的高精度参考
- 或者小体系上的 benchmark

而不是通用大体系首选方案。

---

## 14. 如何选择这些方法

### 14.1 如果你想看完整时间衰减和谱形来源

优先考虑 `renormalizer/spectra`。

原因：

- 直接得到 `autocorr(t)`
- 容易观察去相干、振荡、衰减时间尺度
- 更适合做“动力学到频谱”的分析

### 14.2 如果你只关心某些频率点上的高分辨响应

优先考虑 `renormalizer/cv`。

原因：

- 不需要长时间传播
- 对窄展宽、局部频率窗口更自然
- 可以直接扫频

### 14.3 如果模型简单且存在局域精确传播结构

可以考虑 `SpectraExact` 作为参考。

### 14.4 如果关注的是单粒子谱函数而不是光学偶极谱

看 `transport/spectral_function.py` 更合适。

### 14.5 如果你关注离散激发态而非连续线形

可以参考 `mps/tda.py`。

---

## 15. 总结

`renormalizer` 中与光谱相关的实现可以总结为下面这张“概念地图”：

- `renormalizer/spectra`
  - 目标：时间域偶极相关函数
  - 方法：TD-DMRG / TD-MPS
  - 代表类：
    - `SpectraOneWayPropZeroT`
    - `SpectraTwoWayPropZeroT`
    - `SpectraFiniteT`
    - `SpectraExact`

- `renormalizer/cv`
  - 目标：频域响应函数
  - 方法：Correction Vector / DDMRG
  - 代表类：
    - `SpectraZtCV`
    - `SpectraFtCV`

- `renormalizer/transport/spectral_function.py`
  - 目标：单粒子 Green's function 与谱函数
  - 方法：时间域传播

- `renormalizer/mps/tda.py`
  - 目标：激发态能量与切空间激发
  - 方法：TDA/CIS 风格张量网络激发态算法

如果从最核心的思想来概括，那么这个库里的光谱方法基本都围绕一句话展开：

> 先把“被探测算符作用后的量子态或密度矩阵”构造出来，再用张量网络方法传播或求解响应方程，最后把响应信息转写为可观测谱线。

---

## 16. 后续可继续深入的方向

如果要在这份教程基础上继续深化，建议下一步从以下几个方向继续：

1. 补上每个方法对应的最终傅里叶变换、展宽和单位换算公式
2. 追踪 `Mpo.onsite(..., dipole=True)` 在模型中的偶极定义
3. 分析 `evolve_config`、`compress_config` 对谱线精度的影响
4. 在一个具体 Holstein 模型例子上，把输入参数、时间相关函数和最终谱线完整串起来
5. 对比 `spectra` 与 `cv` 在同一模型上的数值误差与成本

这几步做完后，就可以从“读代码综述”自然过渡到“可复现实战笔记”。
