# -*- coding: utf-8 -*-
"""
MultisetMps: MPS subclass for storing N set elements in batched tensors.

Each site tensor has shape [N_set, bond_left, phys, bond_right].
Quantum numbers stored as [N_set, bond_dim, qn_size].
"""

import logging
from typing import List
import numpy as np

from renormalizer.mps.mps import Mps
from renormalizer.mps.matrix import Matrix, asnumpy, asxp
from renormalizer.mps.backend import backend, xp
from renormalizer.model import Model

logger = logging.getLogger(__name__)


class MultisetMps(Mps):
    """MPS subclass storing N set elements in batched tensors.

    This class implements a batched representation where multiple MPS (representing
    different set elements) are stored together with a batch dimension. Each site
    tensor has shape [N_set, bond_left, phys, bond_right] instead of the standard
    [bond_left, phys, bond_right].

    Key architectural constraints:
    - MPS bond dimensions are SAME across all set elements (enforced by shared QR)
    - This enables safe batching without padding

    Parameters
    ----------
    n_set : int
        Number of set elements (batch size)
    model : Model
        Model information for the system

    Attributes
    ----------
    n_set : int
        Number of set elements in the batch
    qn_batched : List[np.ndarray]
        Quantum numbers with shape [N_set, bond_dim, qn_size] per site
    """

    def __init__(self, n_set: int, model: Model):
        """Initialize batched multiset MPS.

        Parameters
        ----------
        n_set : int
            Number of set elements (batch size)
        model : Model
            Model information
        """
        super().__init__()
        self.model = model
        self.n_set = n_set
        self.qn_batched = []  # [N_set, bond, qn] per site

    @property
    def is_multiset(self) -> bool:
        """Return True to indicate this is a multiset MPS."""
        return True

    def __getitem__(self, idx) -> Matrix:
        """Return batched tensor [N_set, bond_L, phys, bond_R].

        Parameters
        ----------
        idx : int
            Site index

        Returns
        -------
        Matrix
            Batched tensor with shape [N_set, bond_L, phys, bond_R]
        """
        return super().__getitem__(idx)

    def __setitem__(self, idx, value):
        """Set batched tensor at site idx.

        Parameters
        ----------
        idx : int
            Site index
        value : Matrix or np.ndarray
            Batched tensor with shape [N_set, bond_L, phys, bond_R]
        """
        if isinstance(value, np.ndarray):
            value = Matrix(value)
        # Validate shape
        if value.is_batched:
            assert value.batch_size == self.n_set, \
                f"Batch size mismatch: expected {self.n_set}, got {value.batch_size}"
        super().__setitem__(idx, value)

    def append(self, array):
        """Append a batched tensor to the MPS.

        Overrides MatrixProduct.append to properly handle 4D batched tensors.

        Parameters
        ----------
        array : np.ndarray or Matrix
            Batched tensor [N_set, bond_L, phys, bond_R]
        """
        new_mt = self._array2mt(array, len(self))

        # For batched tensors, check bond dimension at index 1 (not 0)
        if len(self._mp) != 0:
            if new_mt.is_batched:
                # 4D case: check new_mt.shape[1] == prev.shape[-1]
                assert new_mt.array.shape[1] == self._mp[-1].shape[-1], \
                    f"Bond dimension mismatch: new tensor has bond_L={new_mt.array.shape[1]}, " \
                    f"but previous tensor has bond_R={self._mp[-1].shape[-1]}"
            else:
                # 3D case: standard check
                assert new_mt.array.shape[0] == self._mp[-1].shape[-1]

        self._mp.append(new_mt)

    def get_single_set(self, alpha: int) -> Mps:
        """Extract single MPS from batch for compatibility with existing code.

        Parameters
        ----------
        alpha : int
            Set index to extract (0 <= alpha < N_set)

        Returns
        -------
        Mps
            Standard MPS for set element alpha
        """
        assert 0 <= alpha < self.n_set, f"Invalid set index {alpha}, must be in [0, {self.n_set})"

        mps = Mps()
        mps.model = self.model

        # Extract tensors for this set element
        for site_idx in range(len(self)):
            tensor_4d = self[site_idx].array  # [N, bond_L, phys, bond_R]
            mps.append(Matrix(tensor_4d[alpha]))  # [bond_L, phys, bond_R]

        # Extract quantum numbers
        if self.qn_batched:
            mps.qn = [qn[alpha] if qn.ndim == 3 else qn for qn in self.qn_batched]
        else:
            mps.qn = self.qn  # Use standard QN if batched QN not set

        mps.qnidx = self.qnidx
        mps.qntot = self.qntot
        mps.to_right = self.to_right
        mps.coeff = self.coeff if hasattr(self, 'coeff') else 1.0

        return mps

    def batched_dot(self, other: 'MultisetMps') -> np.ndarray:
        """Compute <ψ_α|φ_α> for all α in single batched operation.

        This method computes the inner product between corresponding set elements
        efficiently using batched operations.

        Parameters
        ----------
        other : MultisetMps
            Another batched MPS

        Returns
        -------
        np.ndarray
            Array [N_set] with dot products for each set element
        """
        assert self.n_set == other.n_set, "Batch sizes must match"
        assert len(self) == len(other), "MPS lengths must match"

        # Initialize result array
        result = np.ones(self.n_set, dtype=np.complex128 if self.is_complex else np.float64)

        # Contract site by site
        for site in range(len(self)):
            # Get tensors: [N, bond_L, phys, bond_R]
            self_tensor = self[site].array
            other_tensor = other[site].array

            # Contract over bond_L, phys, bond_R → [N]
            # einsum: 'nabc,nabc->n'
            site_overlap = np.einsum('nabc,nabc->n',
                                     np.conj(self_tensor),
                                     other_tensor)
            result *= site_overlap

        return result

    def batched_conj(self) -> 'MultisetMps':
        """Return complex conjugate of all set elements.

        Returns
        -------
        MultisetMps
            Conjugated batched MPS
        """
        new_mps = MultisetMps(self.n_set, self.model)

        for site_idx in range(len(self)):
            tensor = self[site_idx].array
            new_mps.append(Matrix(np.conj(tensor)))

        # Copy metadata
        new_mps.qn_batched = self.qn_batched.copy() if self.qn_batched else []
        new_mps.qn = self.qn.copy() if hasattr(self, 'qn') and self.qn else []
        new_mps.qnidx = self.qnidx
        new_mps.qntot = self.qntot
        new_mps.to_right = self.to_right
        if hasattr(self, 'coeff'):
            new_mps.coeff = np.conj(self.coeff)

        return new_mps

    def normalize_multiset(self, kind='mps_only'):
        """Normalize each set element independently.

        Parameters
        ----------
        kind : str
            Normalization mode:
            - 'mps_only': normalize MPS, keep coeff unchanged
            - 'mps_and_coeff': normalize both MPS and coeff
            - 'mps_norm_to_coeff': normalize MPS, transfer norm to coeff

        Returns
        -------
        self
            Normalized multiset MPS
        """
        # Compute norms for all set elements
        norms = self.batched_dot(self)  # [N]

        if kind == 'mps_only':
            # Normalize each state independently
            scale_factors = 1.0 / np.sqrt(norms)  # [N]
            # Scale only the first site to avoid compound scaling
            # (scaling all sites would result in factor^n_sites)
            if len(self) > 0:
                # Apply per-state scaling: multiply each state's first site by its scale factor
                for alpha in range(self.n_set):
                    self[0].array[alpha] *= scale_factors[alpha]

        elif kind == 'mps_and_coeff':
            scale_factors = 1.0 / np.sqrt(norms)  # [N]
            if len(self) > 0:
                for alpha in range(self.n_set):
                    self[0].array[alpha] *= scale_factors[alpha]
            if hasattr(self, 'coeff'):
                self.coeff = self.coeff / np.linalg.norm(self.coeff)

        elif kind == 'mps_norm_to_coeff':
            scale_factors = 1.0 / np.sqrt(norms)  # [N]
            if len(self) > 0:
                for alpha in range(self.n_set):
                    self[0].array[alpha] *= scale_factors[alpha]
            # Store average norm in coeff
            total_norm = np.sqrt(norms.sum())
            if hasattr(self, 'coeff'):
                self.coeff = self.coeff * total_norm
            else:
                self.coeff = total_norm
        else:
            raise ValueError(f"Unknown normalization kind: {kind}")

        return self

    def copy(self) -> 'MultisetMps':
        """Create a deep copy of the batched MPS.

        Returns
        -------
        MultisetMps
            Deep copy of this batched MPS
        """
        new = MultisetMps(self.n_set, self.model)

        # Copy all tensors
        for site_idx in range(len(self)):
            new.append(self[site_idx].array.copy())

        # Copy metadata
        new.qnidx = self.qnidx
        new.qntot = self.qntot.copy() if self.qntot is not None else None
        new.to_right = self.to_right
        new.dtype = self.dtype
        new.compress_config = self.compress_config

        # Copy batched QN
        if self.qn_batched:
            new.qn_batched = [qn.copy() for qn in self.qn_batched]

        # Copy coeff if exists
        if hasattr(self, 'coeff'):
            new.coeff = self.coeff.copy() if isinstance(self.coeff, np.ndarray) else self.coeff

        return new

    def to_complex(self) -> 'MultisetMps':
        """Convert to complex dtype.

        Returns
        -------
        MultisetMps
            Complex version of this batched MPS
        """
        if self.is_complex:
            return self.copy()

        new = MultisetMps(self.n_set, self.model)
        new.dtype = backend.complex_dtype

        # Convert all tensors to complex
        for site_idx in range(len(self)):
            tensor = self[site_idx].array.astype(backend.complex_dtype)
            new.append(tensor)

        # Copy metadata
        new.qnidx = self.qnidx
        new.qntot = self.qntot.copy() if self.qntot is not None else None
        new.to_right = self.to_right
        new.compress_config = self.compress_config

        # Copy batched QN
        if self.qn_batched:
            new.qn_batched = [qn.copy() for qn in self.qn_batched]

        # Copy coeff if exists
        if hasattr(self, 'coeff'):
            if isinstance(self.coeff, np.ndarray):
                new.coeff = self.coeff.astype(backend.complex_dtype)
            else:
                new.coeff = complex(self.coeff)

        return new

    def canonicalise_multiset(self, normalize=False):
        """QR sweep with QN consistency validation.

        This method performs QR decomposition to canonicalize the MPS while
        enforcing that all set elements maintain consistent quantum number
        structure (same bond dimensions and QN shapes).

        Parameters
        ----------
        normalize : bool
            Whether to normalize after canonicalization

        Returns
        -------
        self
            Canonicalized multiset MPS
        """
        # This is a placeholder - full implementation requires porting
        # the QR logic from the old multiset_model.py with QN consistency checks
        logger.warning("canonicalise_multiset not yet fully implemented")
        return self

    @property
    def mp_norm(self):
        """Compute the norm of the batched MPS.

        Returns
        -------
        float
            Total norm (sum of norms of all set elements)
        """
        norms = self.batched_dot(self)
        return float(np.sqrt(norms.sum()))
