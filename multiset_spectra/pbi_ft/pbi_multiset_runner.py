from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from renormalizer.multiset import MsEvolveMethod, MultisetSpectraFiniteT
from renormalizer.utils import (
    CompressConfig,
    CompressCriteria,
    EvolveConfig,
    Quantity,
)

from pbi_model import construct_pbi_model, display_model_name, normalize_model_name


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def run_multiset(script_file: str, model_name: str, spectratype: str) -> None:
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    model_name = normalize_model_name(model_name)
    spectratype = spectratype.strip().lower()
    if spectratype not in {"abs", "emi"}:
        raise ValueError("spectratype must be 'abs' or 'emi'")

    max_bonddim = _env_int("MAX_BONDDIM", 32)
    evolve_dt = _env_float("EVOLVE_DT", 20.0)
    nsteps = _env_int("NSTEPS", 5000)
    insteps = _env_int("INSTEPS", 50)
    temperature_k = _env_float("TEMPERATURE_K", 298.0)
    threshold = _env_float("COMPRESS_THRESHOLD", 1e-8)
    expand = _env_bool("EXPAND", True)
    thermal_init_method = os.environ.get("THERMAL_INIT_METHOD", "imaginary_time_propagate")

    dump_dir = Path(script_file).resolve().parent
    job_name = f"{display_model_name(model_name)}_{spectratype}_m{max_bonddim}"

    model = construct_pbi_model(model_name)
    evolve_config = EvolveConfig(method=MsEvolveMethod.ms_evolve_tdvp_ps, adaptive=False)
    compress_config = CompressConfig(
        CompressCriteria.both,
        threshold=threshold,
        max_bonddim=max_bonddim,
    )

    spectra = MultisetSpectraFiniteT(
        model=model,
        spectratype=spectratype,
        temperature=Quantity(temperature_k, "K"),
        max_bonddim=max_bonddim,
        thermal_init_method=thermal_init_method,
        insteps=insteps,
        offset=Quantity(0),
        evolve_config=evolve_config,
        compress_config=compress_config,
        expand=expand,
        dump_dir=str(dump_dir),
        job_name=job_name,
    )

    spectra.evolve(evolve_dt=evolve_dt, nsteps=nsteps)
