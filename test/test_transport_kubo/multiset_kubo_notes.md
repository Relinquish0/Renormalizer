# Multiset 有限温 Kubo 设计笔记

这份笔记用于整理当前对 `renormalizer/transport/kubo.py`、`example/transport_kubo.py` 以及 multiset 版本有限温迁移率实现的理解、疑问和下一步设计方向。

## 1. `kubo.py` 做了什么，`transport_kubo.py` 的流程是什么

### 1.1 `TransportKubo` 的物理目标

`renormalizer/transport/kubo.py` 里的 `TransportKubo` 类通过 Green-Kubo 公式计算迁移率：

\[
\mu = \frac{1}{k_B T}\int_0^\infty dt \langle \hat j(t)\hat j(0)\rangle
\]

它真正要算的是电流自相关函数

\[
C(t)=\langle \hat j(t)\hat j(0)\rangle
\]

再对时间积分得到迁移率。

### 1.2 原版 `TransportKubo` 的实现思路

原版实现把有限温 Kubo 分成两段：

1. 先准备有限温“半热态”

\[
e^{-\beta H / 2}
\]

2. 再做实时间传播，计算

\[
\mathrm{Tr}\left[e^{-\beta H / 2} e^{iHt}\hat j e^{-iHt}\hat j e^{-\beta H / 2}\right]
\]

代码层面对应的流程是：

1. 构造哈密顿量 `h_mpo`
2. 从原始模型中扫描并构造电流算符 `j_oper`，必要时还会构造 `j_oper2`
3. 先从无限温态开始，通过虚时演化得到 `e^{-\beta H / 2}`
4. 将电流算符作用到热态上，得到 `J e^{-\beta H / 2}`
5. 同时传播 bra 和 ket
6. 每一步计算电流自相关函数 `C(t)`
7. 对 `C(t)` 的实部做时间积分，得到 mobility

### 1.3 `example/transport_kubo.py` 的案例流程

`example/transport_kubo.py` 本身很薄，主要是在驱动 `TransportKubo`：

1. 读 YAML 参数
2. 用 `load_from_dict(param, 3, False)` 构造 Holstein 模型和温度
3. 配置压缩参数和两类演化参数
   - `ievolve_config`：虚时演化
   - `evolve_config`：实时间演化
4. 实例化 `TransportKubo`
5. 调用 `ct.evolve(...)`
6. 最终输出自相关函数、时间序列和 mobility

可以理解为：

- `example/transport_kubo.py` 只负责“案例输入和任务调度”
- `renormalizer/transport/kubo.py` 负责“有限温 Kubo 的具体物理与数值实现”

## 2. 原版 renormalizer 中电流算符如何构建，`j_1/j_2` 如何拆分，关联函数如何得到

### 2.1 电流算符如何从哈密顿量构建

原版 `TransportKubo` 不是手写电流算符，而是从 `model.ham_terms` 里扫描：

1. 找到包含两个电子算符的项
2. 识别它们对应的电子跃迁方向
3. 乘上距离因子 `(R_m - R_n)`
4. 将该项翻译成电流算符中的一项

其核心依赖是：

- 模型里必须显式有电子 basis
- Hamiltonian 里必须显式保留 `a^\dagger` 和 `a`

### 2.2 为什么会拆成 `j_1` 和 `j_2`

在 Holstein-Peierls 模型里，电流算符自然分成两部分：

1. `j_1`：纯电子跃迁导致的流
2. `j_2`：含声子辅助项的流

即

\[
\hat j = \hat j_1 + \hat j_2
\]

其中：

\[
\hat j_1 \sim \sum_{mn}(R_m-R_n)\epsilon_{mn}a_m^\dagger a_n
\]

\[
\hat j_2 \sim \sum_{mn}(R_m-R_n)V_{mn}(Q)a_m^\dagger a_n
\]

原版代码里：

- 如果没有声子辅助跃迁项，则只构造 `j_oper`
- 如果有 Peierls 型项，则还会构造 `j_oper2`

### 2.3 关联函数如何从流算符得到

如果只有一部分电流算符，则直接计算

\[
C(t)=\langle j(t)j(0)\rangle
\]

如果拆成两部分，则

\[
C(t)=\langle (j_1(t)+j_2(t))(j_1(0)+j_2(0))\rangle
\]

于是分成四项：

1. \(\langle j_1(t)j_1(0)\rangle\)
2. \(\langle j_1(t)j_2(0)\rangle\)
3. \(\langle j_2(t)j_1(0)\rangle\)
4. \(\langle j_2(t)j_2(0)\rangle\)

原版 `kubo.py` 正是这样做的，并把这四项保存为 decomposition。

## 3. multiset 情况下，为什么原版电流算符构造不能直接用，以及应如何构造

### 3.1 问题出在哪里

在 multiset 里，电子态和振动态被拆开了。

当前 `MultisetModel` 的做法是：

1. 把总哈密顿量拆成 block 形式 `H^{alpha,beta}`
2. 在每个 block 里，把电子算符 `a^\dagger a` 从振动模型中剥离出去
3. 最终 `MsModel[alpha][beta]` 只保留振动空间里的算符

因此：

- multiset 的 `MsModel` 已经没有显式电子 basis
- multiset 的 block MPO 里也已经没有原始 `a^\dagger / a`

这意味着原版 `TransportKubo._construct_current_operator()` 不能直接复用。因为它依赖：

1. `basis[site].is_electron`
2. `ham_terms` 中显式的 `a^\dagger` 与 `a`

而这两点在 multiset block 模型中都不成立。

### 3.2 multiset 下正确的电流算符形式

在 multiset 表示里，总态写成

\[
|\Psi\rangle=\sum_\alpha |\alpha\rangle\otimes |\chi_\alpha\rangle
\]

此时电流算符应写成电子 block 形式：

\[
\hat J = \sum_{\alpha\beta} |\alpha\rangle \langle \beta| \otimes \hat J^{\alpha\beta}
\]

也就是：

- 电子部分只负责 `beta -> alpha` 的跳转
- 每个 block 上再挂一个振动空间算符

### 3.3 multiset 下 `j_1` 和 `j_2` 的构造

对于 Holstein / Holstein-Peierls，可以写成：

#### `j_1`：纯电子跃迁部分

\[
\hat J_1^{\alpha\beta} = (R_\alpha - R_\beta)\,\epsilon_{\alpha\beta}\,\hat I_{\mathrm{vib}}
\]

这里振动部分只是单位算符。

#### `j_2`：声子辅助跃迁部分

\[
\hat J_2^{\alpha\beta} = (R_\alpha - R_\beta)\,\hat V_{\alpha\beta}(Q)
\]

这里 \(\hat V_{\alpha\beta}(Q)\) 是跃迁依赖的振动算符，例如线性耦合项 \((b^\dagger+b)\)。

### 3.4 multiset 下流算符对态的作用

在 multiset 中，电流算符作用后应满足

\[
(J\chi)_\alpha = \sum_\beta J^{\alpha\beta}\chi_\beta
\]

这件事在结构上与 multiset Hamiltonian 的 block 作用是同型的。

因此一个关键结论是：

- multiset 版 Kubo 不需要重新发明一套完全不同的传播框架
- 需要新加的是“电流算符的 block 构造与作用”
- 不是复用原版基于电子 basis 扫描 `ham_terms` 的 `j_oper` 构造器

## 4. multiset 版本 `transport_kubo` 应如何实现

### 4.1 总体思路

multiset 版 finite-T Kubo 可以沿用原版 `TransportKubo` 的物理结构，但对象要换成 multiset 版本：

1. 准备有限温初态
2. 构造 multiset block 电流算符
3. 对热态施加 `J_1` / `J_2`
4. 同时传播 bra / ket
5. 逐步计算自相关函数
6. 对时间积分得到 mobility

### 4.2 初态准备应保留两个选项

当前比较自然的设计是保留两个初始化分支：

1. `imaginary_time`
2. `thermofield`

即：

- `imaginary_time`：代表严格参考路线
- `thermofield`：代表基于 Bogoliubov / thermal field 的替代路线

这里不额外引入第三种 bridge 模式。

### 4.3 对 `imaginary_time` 与 `thermofield` 关系的当前判断

当前更稳妥的判断是：

1. 在解析、可对角化且 Bogoliubov 模与目标哈密顿量一致的参考问题里，`thermofield` 与 imaginary time purification 可以等价
2. 但对完整耦合的 Holstein / Peierls 单电子问题，不能在未 benchmark 前直接宣称两者严格等价
3. 因此在 multiset 版 Kubo 的实现中，应把它们视为两个可选的有限温态制备模式，而不是默认完全相同

也就是说：

- `thermofield` 很可能是一个非常有价值、也可能数值上更优的路线
- 但它在项目中仍需通过 benchmark 来确认与严格 Kubo 热态的偏差

### 4.4 时间演化核是否需要改变

当前倾向判断：

- multiset 下时间演化的物理目标没有变
- 实时传播和虚时传播都仍然是把总态投影到 multiset 流形上
- 因而 `multiset-tdvp-ps` 仍然可以作为基础传播器

真正新增的不是新的 TDVP 核，而是：

1. 一个 multiset 版的 Kubo job
2. 一个 multiset 版的 current operator
3. 对多个 multiset 态的统一管理

### 4.5 建议的数据/对象分层

建议把 multiset finite-T Kubo 拆成几层：

#### A. `MultisetCurrentOperator`

负责：

- 从原始 `model.ham_terms` 构造 block 形式的 `J_1` / `J_2`
- 提供对 `MultisetMps` 的作用

#### B. `MultisetTransportKubo`

负责：

- 选择 `imaginary_time` 或 `thermofield`
- 准备 `rho_half`
- 构造 `J_1 rho_half`、`J_2 rho_half`
- 管理实时间传播
- 保存 `auto_corr` 与 decomposition

#### C. 传播内核

继续使用当前已有的：

- `MultisetModel`
- `MultisetTdJob`
- `ms_evolve_tdvp_ps`

### 4.6 第一版实现建议

第一版建议聚焦在：

1. 先只做 Holstein finite-T Kubo
2. 支持 `imaginary_time` / `thermofield` 两种 init mode
3. 先构造 `J_1`
4. 在 benchmark 后再扩展到 `J_2` 和 Peierls decomposition

如果从一开始就按完整结构设计，也可以直接保留 `J_1/J_2` 的接口，但在功能上先让 `J_2` 对 Holstein 模型为空。

## 5. 当前结论与待确认问题

### 5.1 当前较确定的结论

1. 原版 `kubo.py` 的流程已经清楚：热态准备 -> 电流算符作用 -> 双态传播 -> 相关函数 -> mobility
2. 原版电流算符构造依赖显式电子 basis，不能直接套到 multiset block 模型上
3. multiset 下应以 block 形式重新构造电流算符
4. `multiset-tdvp-ps` 大概率仍可继续用作 Kubo 的基础传播方法
5. `imaginary_time` 和 `thermofield` 都值得作为 finite-T Kubo 的初始化选项

### 5.2 仍需 benchmark/验证的问题

1. 在具体 Holstein benchmark 中，`thermofield` 与 `imaginary_time` 的误差到底有多大
2. 在 multiset 有限温态下，哪种初始化方式对后续 Kubo 相关函数更稳定
3. `J_2` 的声子辅助流在 multiset block 形式下，怎样组织数据结构最自然
4. multiset 版 Kubo 的多态传播，是否需要专门优化 checkpoint 和 dump 结构

## 6. 一句话总结

multiset 版 finite-T Kubo 的关键，不是“另写一套全新的时间传播核”，而是：

1. 用 multiset 兼容的方式准备有限温态
2. 用 block 形式重写电流算符
3. 在已有 `multiset-tdvp-ps` 基础上完成 Kubo 的多态传播与相关函数计算
