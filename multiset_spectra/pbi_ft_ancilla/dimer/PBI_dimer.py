from pathlib import Path
import sys

PBI_FT_ANCILLA_ROOT = Path(__file__).resolve().parents[1]
if str(PBI_FT_ANCILLA_ROOT) not in sys.path:
    sys.path.insert(0, str(PBI_FT_ANCILLA_ROOT))

from pbi_multiset_ancilla_runner import run_multiset_ancilla


if __name__ == "__main__":
    run_multiset_ancilla(__file__, model_name="dimer")
