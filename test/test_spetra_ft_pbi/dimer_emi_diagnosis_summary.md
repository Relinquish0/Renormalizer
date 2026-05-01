# Dimer finite-T EMI 诊断记录

## 背景

本记录整理了关于 `Dimer - emi offset sensitivity` 的讨论，涉及：

- `test/test_spetra_ft_pbi/plot/plot_dimer.ipynb`
- `renormalizer/multiset/multisetspectra.py`

目标是判断当前 `multiset` 的 finite-temperature `emi` 谱异常，究竟来自画图后处理，还是来自 spectra 本身的状态表示、含时演化与关联函数定义。

## 关于 offset sensitivity 的结论

### 现象

在 `plot_dimer.ipynb` 中观察到：

- 当 `offset = 2.13 - 0.086 eV` 时，`multiset emi` 明显错误，主峰落在错误波段。
- 当 `offset = 5 eV` 时，`multiset emi` 在 `cm^-1` 和 `nm` 坐标下都呈现出对称/镜像式结果。

### 解释

这不表示 `offset = 5` 时算法变正确，而表示当前 `multiset finite-T emi` 的原始关联函数中带有一个错误的高频载波。

若原始关联函数近似为：

`C(t) ~ A(t) exp(-i Ω_wrong t)`

则 FFT 后的相对频率峰位在 `Ω_wrong` 附近。再加入后处理中的 `offset` 后，得到：

`ω_abs = ω_offset ± ω`

因此：

- 当 `offset` 较小时，一侧可能落到负频率区，被过滤掉，只剩单支错误谱。
- 当 `offset` 足够大时，两侧都落在正频率区，就会表现为围绕 `offset` 中心的双边镜像。

### 数值判断

此前讨论中的诊断显示：

- `multiset finite-T emi` 的 raw autocorr 快相位量级约为 `3.28 eV`
- `singleset finite-T emi` 的残余相位仅约为 `0.04 eV`

因此 `offset=5 eV` 看到的镜像，实质上是错误 carrier 被平移后的结果，而不是正确物理谱。

## 归一化与画图程序的责任

### 归一化不是主因

`plot_dimer.ipynb` 中的归一化是峰值归一化：

```python
max_intensity = np.max(intensities)
if max_intensity != 0:
    intensities = intensities / max_intensity
```

它不是按 `xlim` 内积分归一化，也不是面积归一化。

### plot 程序的错误性质

`plot_dimer.ipynb` 有过一个次要兼容性问题：

- `time series`
- `time_series`

两个 key 的使用不统一。

这个问题会导致 notebook 某些单元读文件失败，但它**不是**导致 `multiset finite-T emi` 主峰落错波段、或 `offset=5` 出现镜像的根因。

### 主结论

主错误不在 `plot_dimer.ipynb`，而在 `renormalizer/multiset/multisetspectra.py`。

## `multisetspectra.py` 中的问题属于哪一部分

更准确地说：

- **主问题不在 `process_mps()` 的最终重叠计算本身**
- **主问题在 finite-T `emi` 这条链路里“什么对象被拿去做含时演化”**

也就是：

- 初态构造
- ground / excited 两支传播如何组织
- dipole 作用后对象是否真正进入了正确的 transition space

而不是单纯某个 overlap 公式正负号写错。

## 明确拆解成两个子问题

### 子问题 1：当前 `init_mps_emi()` 在物理上到底构造了什么对象

当前 finite-T `emi` 初始化流程是：

1. 用 `_init_excited_thermal_state()` 为每个电子通道构造 excited thermal 纯化对象
2. 再在 `_init_dipole()` 中给每个通道乘以 dipole 权重
3. 但结果仍然保留为按 `alpha` 分块的 `MultisetMps`

所以现在得到的对象不是：

- 一个公共 ground transition object

而是：

- 一个带 dipole 权重的 excited-sector multiset superposition

问题在于，对于 finite-T `emi`，dipole 之后物理上应当进入公共的 transition space；但当前代码只是改了各通道权重，并没有真正把对象变成那个 transition object。

### 子问题 2：正确的 finite-T `emi` hybrid 表示应如何组织

一个物理上更合理的 finite-T `emi` hybrid 表示应当分成两支：

#### 1. ground branch

- 单个 ground-side transition object
- 不再保留电子通道分块
- 负责 `e^{+i H_g t}` 传播

#### 2. excited branch

- 保留为 `MultisetMps`
- 负责 excited thermal state 在 multiset 表示下的传播
- 负责 `e^{-i H_e t}` 传播

因此正确的 finite-T `emi` 不应继续沿用当前这种：

- `bra = MultisetMps`
- `ket = MultisetMps`
- 两边都交替走同一套 finite-T branch 演化

而应当更像：

- `bra =` 单个 ground transition object
- `ket =` excited thermal `MultisetMps`

然后：

- `bra` 只按 ground Hamiltonian 演化
- `ket` 只按 excited multiset Hamiltonian 演化
- 最后再做 ground-side object 与 excited multiset components 之间的矩阵元收缩

## 已经确认的结论

1. `abs` 路径与后处理基本正常，`multiset` 和 `singleset` 一致性较好。
2. `finite-T emi` 的 `multiset` 路径存在结构性问题。
3. 当前问题的主根因在 `multisetspectra.py`，不是 `plot_dimer.ipynb`。
4. 当前异常更接近“对象表示/传播链路错误”，而不是“最终重叠公式局部写错”。
5. 若要真正修复 finite-T `emi`，需要重构为 ground-branch / excited-branch 分离的 hybrid 表示。

## 备注

此次讨论中，`plot_dimer.ipynb` 已被改成可以同时输出：

- `Wavelength (nm)`
- `Wavenumber (cm^-1)`

两套图，用于更直观地区分：

- 只是 `offset` 引起的整体平移
- 还是内部频率结构本身已经错误
