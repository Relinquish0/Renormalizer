# -*- coding: utf-8 -*-
"""
MultisetMpo: MPO manager for N² operator matrices.

Stores H^{α,β} MPO for each pair of electronic states (α, β).
"""

import logging
from typing import List
import numpy as np

from renormalizer.mps.mpo import Mpo
from renormalizer.model import Model

logger = logging.getLogger(__name__)


class MultisetMpo:
    """
    Multiset Matrix Product Operator for N electronic states.

    Stores N² MPO objects H^{α,β} representing the Hamiltonian
    matrix elements between electronic states α and β.

    Attributes
    ----------
    n_set : int
        Number of electronic states (N)
    msmpo : List[List[Mpo or list]]
        N×N grid of MPO objects. Empty list [] if no coupling.
    """

    def __init__(self, models: List[List[Model]], n_set: int):
        """
        Initialize MultisetMpo from model grid.

        Parameters
        ----------
        models : List[List[Model]]
            N×N grid of Model objects for each (α,β) pair
        n_set : int
            Number of electronic states
        """
        self.n_set = n_set
        self.models = models
        self.msmpo = [[[] for _ in range(n_set)] for _ in range(n_set)]
        self._construct_msmpo()

    def _construct_msmpo(self):
        """Construct N² MPO objects from models."""
        for alpha in range(self.n_set):
            for beta in range(self.n_set):
                model = self.models[alpha][beta]
                # Check if model has any Hamiltonian terms
                if len(model.ham_terms) == 0:
                    self.msmpo[alpha][beta] = []  # Empty MPO
                else:
                    self.msmpo[alpha][beta] = Mpo(model=model, terms=None)

    def __getitem__(self, key):
        """
        Access MPO for specific (α,β) pair.

        Parameters
        ----------
        key : tuple of int
            (alpha, beta) indices

        Returns
        -------
        Mpo or list
            MPO object for the pair, or empty list if no coupling
        """
        if isinstance(key, tuple) and len(key) == 2:
            alpha, beta = key
            return self.msmpo[alpha][beta]
        else:
            raise KeyError("MultisetMpo requires (alpha, beta) tuple indexing")

    def __setitem__(self, key, value):
        """Set MPO for specific (α,β) pair."""
        if isinstance(key, tuple) and len(key) == 2:
            alpha, beta = key
            self.msmpo[alpha][beta] = value
        else:
            raise KeyError("MultisetMpo requires (alpha, beta) tuple indexing")

    def __len__(self):
        """Return number of sites (assumes all non-empty MPOs have same length)."""
        for alpha in range(self.n_set):
            for beta in range(self.n_set):
                if len(self.msmpo[alpha][beta]) > 0:
                    return len(self.msmpo[alpha][beta])
        return 0

    def total_mpo(self) -> Mpo:
        """
        Sum all MPO components into a single total MPO.

        Returns
        -------
        Mpo
            Total Hamiltonian H = Σ_{α,β} H^{α,β}
        """
        from renormalizer.mps.lib import _sum

        mpos = []
        for alpha in range(self.n_set):
            row_mpos = []
            for beta in range(self.n_set):
                if len(self.msmpo[alpha][beta]) > 0:
                    row_mpos.append(self.msmpo[alpha][beta])
            if row_mpos:
                mpos.append(_sum(mps_list=row_mpos, compress=False, temp_m_trunc=None))

        if not mpos:
            raise ValueError("All MPOs are empty")

        return _sum(mps_list=mpos, compress=False, temp_m_trunc=None)

    @property
    def bond_dims(self):
        """Return bond dimensions of first non-empty MPO."""
        for alpha in range(self.n_set):
            for beta in range(self.n_set):
                if len(self.msmpo[alpha][beta]) > 0:
                    return self.msmpo[alpha][beta].bond_dims
        return []

    def __repr__(self):
        n_empty = sum(1 for a in range(self.n_set) for b in range(self.n_set)
                     if len(self.msmpo[a][b]) == 0)
        n_total = self.n_set * self.n_set
        return f"MultisetMpo(n_set={self.n_set}, MPOs={n_total-n_empty}/{n_total}, sites={len(self)})"
