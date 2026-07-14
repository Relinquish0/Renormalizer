from pathlib import Path
import sys

PBI_FT_ROOT = Path(__file__).resolve().parents[3]
if str(PBI_FT_ROOT) not in sys.path:
    sys.path.insert(0, str(PBI_FT_ROOT))

from pbi_singleset_runner import run_singleset


if __name__ == "__main__":
    run_singleset(__file__, model_name="hexamer", spectratype="emi")
