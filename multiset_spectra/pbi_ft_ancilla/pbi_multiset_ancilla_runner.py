from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PBI_FT_ROOT = REPO_ROOT / "multiset_spectra" / "pbi_ft"
for path in (REPO_ROOT, PBI_FT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pbi_model import construct_pbi_model, display_model_name, normalize_model_name
from renormalizer.multiset import MsEvolveMethod, MultisetSpectraFiniteT
from renormalizer.utils import (
    CompressConfig,
    CompressCriteria,
    EvolveConfig,
    Quantity,
)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def run_multiset_ancilla(script_file: str, model_name: str) -> None:
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    model_name = normalize_model_name(model_name)
    spectratype = "emi"
    max_bonddim = _env_int("MAX_BONDDIM", 32)
    evolve_dt = _env_float("EVOLVE_DT", 20.0)
    nsteps = _env_int("NSTEPS", 5000)
    insteps = _env_int("INSTEPS", 50)
    temperature_k = _env_float("TEMPERATURE_K", 298.0)
    threshold = _env_float("COMPRESS_THRESHOLD", 1e-8)
    expand = _env_bool("EXPAND", True)
    thermal_init_method = os.environ.get("THERMAL_INIT_METHOD", "imaginary_time_propagate")

    if thermal_init_method != "imaginary_time_propagate":
        raise ValueError("Electronic-ancilla finite-T emission requires imaginary_time_propagate.")

    dump_dir = Path(script_file).resolve().parent
    job_name = f"{display_model_name(model_name)}_{spectratype}_ma{max_bonddim}"

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
        use_electronic_ancilla=True,
        dump_dir=str(dump_dir),
        job_name=job_name,
    )

    spectra.evolve(evolve_dt=evolve_dt, nsteps=nsteps)

    output_path = dump_dir / f"{job_name}.npz"
    np.savez(
        output_path,
        model=model_name,
        spectratype=spectratype,
        method="multiset_electronic_ancilla",
        temperature=Quantity(temperature_k, "K").as_au(),
        temperature_k=temperature_k,
        max_bonddim=max_bonddim,
        evolve_dt=evolve_dt,
        nsteps=nsteps,
        insteps=insteps,
        compress_threshold=threshold,
        expand=expand,
        thermal_init_method=thermal_init_method,
        use_electronic_ancilla=True,
        carrier_energy=getattr(spectra, "carrier_energy", 0.0),
        time_series=np.asarray(spectra.evolve_times),
        **{"time series": np.asarray(spectra.evolve_times)},
        autocorr=spectra.autocorr,
        autocorr_components=spectra.autocorr_components,
    )
