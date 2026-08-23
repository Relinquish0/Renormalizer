# -*- coding: utf-8 -*-

"""2D Holstein model builder for the 4xV100 multiset scaling runs.

Parameters follow ``multiset_202607/2DHolstein1515_speed`` exactly so the new
sharded code can be compared against the published 15x15 numbers:
omega0 = J = 1, g = 0.5, nu_max = 8, open boundaries, T = 0 K.

Everything is an environment variable so one driver covers the whole size ladder.
"""

import os

import numpy as np

from renormalizer.model import HolsteinModel, Mol, Phonon
from renormalizer.utils import Quantity


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


class HolsteinConfig:
    """Resolved run configuration, read once from the environment."""

    def __init__(self):
        self.nrow = _env_int("NROW", 15)
        self.ncol = _env_int("NCOL", self.nrow)
        self.omega_0 = _env_float("OMEGA_0", 1.0)
        self.j = _env_float("J", 1.0)
        self.g = _env_float("G", 0.5)
        self.nu_max = _env_int("NU_MAX", 8)
        self.max_bonddim = _env_int("MAX_BONDDIM", 64)
        self.evolve_dt = _env_float("EVOLVE_DT", 0.1)
        self.nsteps = _env_int("NSTEPS", 5)
        self.periodic = _env_flag("PERIODIC", False)
        initial = os.environ.get("INITIAL_SITE")
        self.initial_site = (
            site_index(self.nrow // 2, self.ncol // 2, self.ncol)
            if initial is None
            else int(initial)
        )
        if self.nrow < 1 or self.ncol < 1:
            raise ValueError("NROW and NCOL must both be positive")
        if not 0 <= self.initial_site < self.nrow * self.ncol:
            raise ValueError(
                f"INITIAL_SITE must lie in [0, {self.nrow * self.ncol})"
            )

    @property
    def n_sites(self):
        return self.nrow * self.ncol

    @property
    def tag(self):
        return f"{self.nrow}x{self.ncol}"

    def as_dict(self):
        return {
            "nrow": self.nrow,
            "ncol": self.ncol,
            "n_sites": self.n_sites,
            "omega_0": self.omega_0,
            "J": self.j,
            "g": self.g,
            "nu_max": self.nu_max,
            "max_bonddim": self.max_bonddim,
            "evolve_dt": self.evolve_dt,
            "nsteps": self.nsteps,
            "initial_site": self.initial_site,
            "periodic": self.periodic,
        }


def site_index(ix, iy, ncol):
    return ix * ncol + iy


def build_2d_j_matrix(nrow, ncol, j, periodic=False):
    """Nearest-neighbour transfer integrals on a rectangular lattice.

    Row-major site numbering means a coupling reaches alpha +- 1 (within a row)
    or alpha +- ncol (between rows); that bound is what the halo width in
    ``mpi_shard_patch`` relies on.
    """
    j_matrix = np.zeros((nrow * ncol, nrow * ncol))
    for ix in range(nrow):
        for iy in range(ncol):
            current = site_index(ix, iy, ncol)
            if periodic or ix + 1 < nrow:
                neighbor = site_index((ix + 1) % nrow, iy, ncol)
                if neighbor != current:
                    j_matrix[current, neighbor] = j
                    j_matrix[neighbor, current] = j
            if periodic or iy + 1 < ncol:
                neighbor = site_index(ix, (iy + 1) % ncol, ncol)
                if neighbor != current:
                    j_matrix[current, neighbor] = j
                    j_matrix[neighbor, current] = j
    return j_matrix


def build_model(cfg: HolsteinConfig):
    lam = cfg.g**2 * cfg.omega_0
    displacement = np.sqrt(2.0 * lam) / cfg.omega_0
    ph = Phonon.simple_phonon(
        Quantity(cfg.omega_0), Quantity(displacement), cfg.nu_max
    )
    mol = Mol(Quantity(0), [ph])
    j_matrix = build_2d_j_matrix(cfg.nrow, cfg.ncol, cfg.j, periodic=cfg.periodic)
    return HolsteinModel([mol] * cfg.n_sites, j_matrix, scheme=2)


LIGHTWEIGHT_OBSERVABLES = {
    "energy": False,
    "r_square": False,
    "e_occupations": True,
    "ph_occupations": False,
    "S_all": False,
    "S_maxbond_eachset": False,
    "S_maxbond": False,
    "S_maxbond_normed": False,
    "S_maxbond_unnormed": False,
    "rho": False,
    "coherent_length": False,
    "trace": False,
    "purity": False,
}
