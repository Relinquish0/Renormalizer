# -*- coding: utf-8 -*-

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from holstein31_ft_common import run_multiset


if __name__ == "__main__":
    run_multiset(0.5)
