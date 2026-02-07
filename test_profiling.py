#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
性能剖析脚本：找出 _apply_block_operator 的真正瓶颈

使用方法:
    1. 修改你的代码，初始化时添加 enable_profiling=True:
       ms_model = MultisetModel(model, max_bonddim=100, enable_profiling=True)

    2. 运行你的测试:
       python test/fmo.py  # 或其他测试脚本

    3. 查看输出的剖析信息
"""

import logging
import numpy as np

def main():
    """
    示例：如何在你的代码中启用剖析
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("性能剖析使用说明")
    print("=" * 70)
    print()
    print("1. 在你的代码中初始化 MultisetModel 时添加:")
    print("   ms_model = MultisetModel(model, max_bonddim=100, enable_profiling=True)")
    print()
    print("2. 运行你的测试代码")
    print()
    print("3. 查看类似如下的输出:")
    print()
    print("   [INFO] [Profile] === Call #1 (imps=5, dim=15000) ===")
    print("   [INFO] [Profile] Total: 58.23 ms")
    print("   [INFO] [Profile]   - Reshape: 0.15 ms (0.3%)")
    print("   [INFO] [Profile]   - Op calls (49x): 56.80 ms (97.5%)")
    print("   [INFO] [Profile]     * Avg per call: 1.16 ms")
    print("   [INFO] [Profile]     * Min/Max: 0.85 / 1.45 ms")
    print("   [INFO] [Profile]   - Stack+Sum: 1.28 ms (2.2%)")
    print()
    print("=" * 70)
    print("解读:")
    print("  - 如果 'Op calls' 占 95%+ → 瓶颈在张量收缩本身")
    print("    * 优化方向: 降低 block_dim, 使用更小的 bond dimension")
    print("    * 或优化 hop_expr/张量收缩的实现")
    print()
    print("  - 如果 'Stack+Sum' 占比高 → 可以优化累加逻辑")
    print("  - 如果 'Reshape' 占比高 → 可以优化数据布局")
    print("=" * 70)

if __name__ == "__main__":
    main()
