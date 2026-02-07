#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试矩阵优化方法的性能对比脚本

使用方法:
    python test_matrix_optimization.py
"""

import numpy as np
import time
from renormalizer.model.multiset_model import MultisetModel

def test_performance(enable_perf_test=True):
    """
    运行你的 FMO 或其他测试，对比性能
    """
    # 导入你的模型设置
    # 例如从 test/fmo.py 导入
    # from test.fmo import setup_model

    print("=" * 70)
    print("矩阵优化方法性能测试")
    print("=" * 70)

    # TODO: 替换成你的实际模型初始化代码
    # model = setup_model()
    # ms_model = MultisetModel(model, max_bonddim=100, enable_performance_test=enable_perf_test)

    print("\n提示: 请在你的实际代码中:")
    print("1. 初始化 MultisetModel 时添加参数: enable_performance_test=True")
    print("   例如: ms_model = MultisetModel(model, max_bonddim=100, enable_performance_test=True)")
    print("\n2. 运行完 evolve 后调用:")
    print("   ms_model.print_performance_stats()")
    print("\n3. 对比输出的时间统计")

    # 示例性能对比输出
    print("\n" + "=" * 70)
    print("预期性能改进:")
    print("  - 原始方法 (for 循环):          ~60 ms/call")
    print("  - 矩阵方法 (构建):              ~100-200 ms (一次性)")
    print("  - 矩阵方法 (应用):              ~5-10 ms/call")
    print("  - 如果 expm_krylov 调用 10 次:")
    print("    * 原始: 60ms × 10 = 600ms")
    print("    * 矩阵: 150ms + 7ms × 10 = 220ms")
    print("    * 加速比: ~2.7x")
    print("=" * 70)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    test_performance()
