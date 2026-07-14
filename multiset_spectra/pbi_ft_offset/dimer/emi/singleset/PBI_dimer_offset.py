from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
PBI_FT_ROOT = REPO_ROOT / "multiset_spectra" / "pbi_ft"
for path in (REPO_ROOT, PBI_FT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from renormalizer.mps import Mpo, Mps, gs
from renormalizer.spectra import SpectraFiniteT
from renormalizer.utils import (
    CompressConfig,
    CompressCriteria,
    EvolveConfig,
    EvolveMethod,
    OptimizeConfig,
    Quantity,
    constant,
)

from pbi_model import construct_pbi_model, display_model_name, normalize_model_name


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def model_zpe_ha(model) -> float:
    if hasattr(model, "gs_zpe"):
        return float(model.gs_zpe)
    return sum(0.5 * float(ph.omega[0]) for mol in model.mol_list for ph in mol.ph_list)


def zero_t_one_exciton_energy_ha(model) -> float:
    opt = OptimizeConfig()
    np.random.seed(0)
    h_mpo = Mpo(model, offset=Quantity(0))
    i_mps = Mps.random(model, 1, opt.procedure[0][0], 1)
    i_mps.optimize_config = opt
    energies, _ = gs.optimize_mps(i_mps, h_mpo)
    return float(np.min(energies))


def offset_energy_ha(mode: str, e_ex0_ha: float, e00_ha: float) -> float:
    mode = mode.upper()
    if mode == "EEX0":
        return e_ex0_ha
    if mode == "E00":
        return e00_ha
    raise ValueError("OFFSET_MODE must be EEX0 or E00")


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    model_name = normalize_model_name(_env_str("MODEL_NAME", "dimer"))
    spectratype = "emi"
    max_bonddim = _env_int("MAX_BONDDIM", 64)
    evolve_dt = _env_float("EVOLVE_DT", 20.0)
    nsteps = _env_int("NSTEPS", 5000)
    insteps = _env_int("INSTEPS", 50)
    temperature_k = _env_float("TEMPERATURE_K", 298.0)
    offset_mode = _env_str("OFFSET_MODE", "EEX0").upper()

    dump_dir = Path(__file__).resolve().parent
    model = construct_pbi_model(model_name)

    gs_zpe_ha = model_zpe_ha(model)
    e_ex0_ha = zero_t_one_exciton_energy_ha(model)
    e00_ha = e_ex0_ha - gs_zpe_ha
    carrier_ha = offset_energy_ha(offset_mode, e_ex0_ha, e00_ha)

    display = display_model_name(model_name)
    job_name = f"{display}_{spectratype}_s{max_bonddim}_offset_{offset_mode}"

    print(f"Model = {model_name}")
    print(f"MAX_BONDDIM = {max_bonddim}")
    print(f"OFFSET_MODE = {offset_mode}")
    print(f"gs_zpe = {gs_zpe_ha:.12f} Ha = {gs_zpe_ha * constant.au2ev:.9f} eV")
    print(f"E_EX0 = {e_ex0_ha:.12f} Ha = {e_ex0_ha * constant.au2ev:.9f} eV")
    print(f"E00 = {e00_ha:.12f} Ha = {e00_ha * constant.au2ev:.9f} eV")
    print(f"runtime offset/carrier = {carrier_ha:.12f} Ha = {carrier_ha * constant.au2ev:.9f} eV")

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
        offset=Quantity(carrier_ha),
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
        method="singleset_offset",
        offset_mode=offset_mode,
        temperature=Quantity(temperature_k, "K").as_au(),
        temperature_k=temperature_k,
        max_bonddim=max_bonddim,
        evolve_dt=evolve_dt,
        nsteps=nsteps,
        insteps=insteps,
        carrier_energy=carrier_ha,
        e_ex0_energy=e_ex0_ha,
        e00_energy=e00_ha,
        gs_zpe_energy=gs_zpe_ha,
        time_series=np.asarray(spectra.evolve_times),
        **{"time series": np.asarray(spectra.evolve_times)},
        autocorr=spectra.autocorr,
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
