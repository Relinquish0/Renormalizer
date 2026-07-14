from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from renormalizer.spectra import SpectraFiniteT
from renormalizer.utils import (
    CompressConfig,
    CompressCriteria,
    EvolveConfig,
    EvolveMethod,
    Quantity,
)

from pbi_model import construct_pbi_model, display_model_name, normalize_model_name


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def run_singleset(script_file: str, model_name: str, spectratype: str) -> None:
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

    dump_dir = Path(script_file).resolve().parent
    job_name = f"{display_model_name(model_name)}_{spectratype}_s{max_bonddim}"

    model = construct_pbi_model(model_name)
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, adaptive=False)
    compress_config = CompressConfig(
        CompressCriteria.fixed,
        max_bonddim=max_bonddim,
    )

    spectra = SpectraFiniteT(
        model=model,
        spectratype=spectratype,
        temperature=Quantity(temperature_k, "K"),
        evolve_config=evolve_config,
        insteps=insteps,
        offset=Quantity(0),
        compress_config=compress_config,
        icompress_config=compress_config,
        dump_dir=str(dump_dir),
        job_name=job_name,
    )

    spectra.evolve(evolve_dt=evolve_dt, nsteps=nsteps)

    output_path = dump_dir / f"{job_name}.npz"
    np.savez(
        output_path,
        model=model_name,
        spectratype=spectratype,
        method="singleset",
        temperature=Quantity(temperature_k, "K").as_au(),
        temperature_k=temperature_k,
        max_bonddim=max_bonddim,
        evolve_dt=evolve_dt,
        nsteps=nsteps,
        time_series=np.asarray(spectra.evolve_times),
        **{"time series": np.asarray(spectra.evolve_times)},
        autocorr=spectra.autocorr,
    )
