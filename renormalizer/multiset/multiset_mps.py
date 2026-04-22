# -*- coding: utf-8 -*-

import logging
import os
from enum import Enum

import numpy as np

from renormalizer.model.model import Model
from renormalizer.mps import MpDm, Mps
from renormalizer.mps.lib import _sum
from renormalizer.utils import Quantity

logger = logging.getLogger(__name__)


class MsEvolveMethod(Enum):
    ms_evolve_tdvp_ps = "TDVP PS one-site with multiset-mps ansatz"


class MultisetMps:
    """
    Docstring for MultisetMps
    """

    def __init__(
        self,
        msmodel,
        N_electron: int,
        temperature: Quantity = Quantity(0, "K"),
        init_model: Model = None,
        method: str = "thermo_field",
        init_mp=None,
        msmps=None,
    ):
        if init_mp is not None and msmps is not None:
            raise ValueError("Only one of `init_mp` and `msmps` can be provided.")
        if init_mp is None and msmps is None:
            raise ValueError("Either `init_mp` or `msmps` should be provided.")
        self.MsModel = msmodel
        self.N_electron = N_electron
        self.msmps = [[] for _ in range(self.N_electron)]
        self.temperature = temperature
        self.init_model = self.MsModel[0][0] if init_model is None else init_model
        self.method = method
        self._init_mp = init_mp
        self._input_msmps = msmps
        self._ConstructMsMps()

    def _ConstructMsMps(self):
        if self._input_msmps is not None:
            if len(self._input_msmps) != self.N_electron:
                raise ValueError(
                    f"`msmps` should contain {self.N_electron} states. Got {len(self._input_msmps)}."
                )
            self.msmps = [mps.copy() for mps in self._input_msmps]
            return

        init_mp = self._init_mp
        for i in range(self.N_electron):
            self.msmps[i] = init_mp.copy()

    def copy(self) -> "MultisetMps":
        """
        Create a deep copy of an entire MultisetMps object
        """
        new = MultisetMps.__new__(MultisetMps)
        new.MsModel = self.MsModel
        new.N_electron = self.N_electron
        new.temperature = self.temperature
        new.init_model = self.init_model
        new.method = self.method
        new.msmps = [m.copy() for m in self.msmps]
        return new

    def to_complex(self) -> "MultisetMps":
        """
        Create a deep copy of a complex MultisetMps object
        """
        new = MultisetMps.__new__(MultisetMps)
        new.MsModel = self.MsModel
        new.N_electron = self.N_electron
        new.temperature = self.temperature
        new.init_model = self.init_model
        new.method = self.method
        new.msmps = [m.to_complex() for m in self.msmps]
        return new

    def total_mps(self) -> "Mps":
        return _sum(mps_list=self.msmps, compress=False, temp_m_trunc=None)

    def ms_normalize(self, kind):
        r"""normalize the wavefunction

        Parameters
        ----------
        kind: str
            "mps_only": the mps part is normalized and coeff is not modified;
            "mps_norm_to_coeff": the mps part is normalized and the norm is multiplied to coeff;
            "mps_and_coeff": both mps and coeff is normalized

        Returns
        -------
        ``self`` is overwritten.
        """

        total_tn_coeff = 0

        for alpha in range(self.N_electron):
            total_tn_coeff += self.msmps[alpha].conj().dot(self.msmps[alpha])

        total_tn_coeff = total_tn_coeff**0.5

        if kind in ["mps_only"]:
            for alpha in range(self.N_electron):
                self.msmps[alpha].scale(1.0 / total_tn_coeff, inplace=True)
        else:
            raise ValueError(f"kind={kind} is not valid.")

    def rho_el(self):
        rho = np.zeros((self.N_electron, self.N_electron), dtype=np.complex128)
        for alpha in range(self.N_electron):
            bra = self.msmps[alpha].conj()
            for beta in range(self.N_electron):
                rho[alpha, beta] = bra.dot(self.msmps[beta])
        return rho

    @property
    def e_occupations_multiset(self):
        return np.diag(self.rho_el()).real

    @property
    def ph_occupations_multiset(self):
        ph_occupations = None
        for mps in self.msmps:
            occupations = np.asarray(mps.ph_occupations)
            if ph_occupations is None:
                ph_occupations = occupations.copy()
            else:
                ph_occupations = ph_occupations + occupations

        if ph_occupations is None:
            return np.array([])
        if np.allclose(ph_occupations.imag, 0):
            return ph_occupations.real
        return ph_occupations

    def dump(self, fname):
        root, ext = os.path.splitext(fname)
        if ext == "":
            ext = ".npz"
            fname = root + ext

        state_paths = []
        state_types = []
        for idx, mps in enumerate(self.msmps):
            state_path = f"{root}_set{idx}{ext}"
            mps.dump(state_path)
            state_paths.append(state_path)
            state_types.append(type(mps).__name__)

        np.savez(
            fname,
            version="0.1",
            N_electron=self.N_electron,
            temperature=self.temperature.as_au(),
            method=self.method,
            state_paths=np.array(state_paths, dtype=object),
            state_types=np.array(state_types, dtype=object),
        )

    @classmethod
    def load(cls, msmodel, N_electron: int, fname: str, init_model: Model = None):
        npload = np.load(fname, allow_pickle=True)
        new = cls.__new__(cls)
        new.MsModel = msmodel
        new.N_electron = int(npload["N_electron"]) if "N_electron" in npload else N_electron
        new.temperature = Quantity(float(npload["temperature"])) if "temperature" in npload else Quantity(0)
        new.init_model = msmodel[0][0] if init_model is None else init_model
        method = npload["method"] if "method" in npload else "thermo_field"
        new.method = method.item() if hasattr(method, "item") else str(method)
        new.msmps = []

        for state_path, state_type in zip(npload["state_paths"].tolist(), npload["state_types"].tolist()):
            state_type = state_type.item() if hasattr(state_type, "item") else str(state_type)
            if state_type == "MpDm":
                state = MpDm.load(new.init_model, state_path)
            elif state_type == "Mps":
                state = Mps.load(new.init_model, state_path)
            else:
                raise ValueError(f"Unsupported multiset state type in dump: {state_type}")
            new.msmps.append(state)
        return new
