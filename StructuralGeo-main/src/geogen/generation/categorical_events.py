""" Categorical definitions for sampling of broad categories of GeoWords."""

__all__ = [
    "BaseStrata",
    "Sediment",
    "Erosion",
    "Dike",
    "Sills",
    "Pluton",
    "OreDeposit",
    "Fold",
    "Fault",
    "Slip",
    "Mountains",
    "End",
]

import warnings
from collections import namedtuple
from typing import List

import numpy as np

from geogen.generation.geowords import *
from geogen.model import CompoundProcess, GeoProcess


def _word(word_class, rng_contract, name):
    """Instantiate a nested word with a stable, name-derived RNG namespace."""
    kwargs = {}
    if rng_contract is not None:
        kwargs["rng_contract"] = rng_contract.child(name)
    return word_class(**kwargs)


class _EventTemplateClass(GeoWord):
    """
    A special case of GeoWord that selects from a set of cases with associated probabilities.
    This class is used to form more general categories of events that can be sampled from.

    The Events include a name, probability of selection, and a sequence of actions (GeoWords or GeoProcesses)
    that form the history of the event.

    Parameters
    ----------
    cases : List[Event]
        A list of Events to sample from. Each event should have a name, probability, and a sequence of GeoWords or GeoProcesses.
    rng : Optional[np.random.Generator]
        A random number generator for reproducibility.
    """

    Event = namedtuple("Case", ["name", "p", "processes"])

    def __init__(self, cases: List[Event], seed=None, rng_contract=None):
        super().__init__(seed=seed, rng_contract=rng_contract)
        self.rng = self.rng_for("event_subtype")
        self.cases = cases
        self.selected_case = None
        self.probabilities = None
        self._validate_cases()
        self._validate_probabilities()

    def generate(self):
        """
        Generate the geological history by selecting a case and building the corresponding history.

        Returns
        -------
        geo.CompoundProcess
            A sampled geological history snippet with a CompoundProcess wrapper.
        """
        self.hist.clear()
        self.build_history()
        name = f"{self.__class__.__name__}: {self.selected_case.name}"
        geoprocess = CompoundProcess(self.hist.copy(), name=name)
        return geoprocess

    def build_history(self):
        """
        Randomly select a case based on probabilities and build the corresponding history.
        """
        assert self.probabilities is not None, "Probabilities are not defined."
        selected_index = self.rng.choice(len(self.cases), p=self.probabilities)
        self.selected_case = self.cases[selected_index]
        self.add_process(self.selected_case.processes)

    def _validate_cases(self):
        """
        Ensure that the case list is correctly defined with valid types.
        """
        if not self.cases:
            raise ValueError("Cases are not defined.")

        for case in self.cases:
            if not isinstance(case.name, str):
                raise TypeError(f"Case name must be a string, got {type(case.name).__name__}.")
            if not isinstance(case.p, float):
                raise TypeError(f"Case probability must be a float, got {type(case.p).__name__}.")
            if not isinstance(case.processes, list):
                raise TypeError(f"Case processes must be a list, got {type(case.processes).__name__}.")
            for process in case.processes:
                if not isinstance(process, (GeoProcess, GeoWord)):
                    raise TypeError(
                        f"Processes must be instances of GeoProcess or GeoWord, got {type(process).__name__}."
                    )

    def _validate_probabilities(self):
        """
        Ensure that the probabilities sum to 1 and renormalize if necessary.
        """
        probabilities = np.array([case.p for case in self.cases])
        sum_prob = np.sum(probabilities)

        if not np.isclose(sum_prob, 1.0):
            warnings.warn(
                f"{self.__class__.__name__}: Probabilities sum to {sum_prob:.4f}, but should sum to 1.0. Renormalizing.",
                RuntimeWarning,
            )
            probabilities = np.array(probabilities) / sum_prob

        self.probabilities = probabilities


class BaseStrata(_EventTemplateClass):
    """
    A sampling regime for base strata.
    """

    def __init__(self, seed=None, rng_contract=None):
        cases = [
            self.Event(
                name="Basement",
                p=0.27,
                processes=[
                    _word(InfiniteBasement, rng_contract, "basement_00"),
                    _word(Sediment, rng_contract, "basement_sediment_01"),
                ],
            ),
            self.Event(
                name="Sediment: Markov",
                p=0.22,
                processes=[
                    _word(InfiniteSedimentMarkov, rng_contract, "markov_00")
                ],
            ),
            self.Event(
                name="Sediment: Uniform",
                p=0.22,
                processes=[
                    _word(InfiniteSedimentUniform, rng_contract, "uniform_00")
                ],
            ),
            self.Event(
                name="Sediment: Tilted Markov",
                p=0.29,
                processes=[
                    _word(InfiniteSedimentTilted, rng_contract, "tilted_markov_00")
                ],
            ),
        ]
        super().__init__(cases=cases, seed=seed, rng_contract=rng_contract)


class Sediment(_EventTemplateClass):
    """
    A sampling regime for sediment events.
    """

    def __init__(self, seed=None, rng_contract=None):
        cases = [
            self.Event(
                name="Fine",
                p=0.4,
                processes=[_word(FineRepeatSediment, rng_contract, "fine_00")],
            ),
            self.Event(
                name="Coarse",
                p=0.5,
                processes=[_word(CoarseRepeatSediment, rng_contract, "coarse_00")],
            ),
            self.Event(
                name="Single",
                p=0.1,
                processes=[_word(SingleRandSediment, rng_contract, "single_00")],
            ),
        ]
        super().__init__(cases=cases, seed=seed, rng_contract=rng_contract)


class Erosion(_EventTemplateClass):
    """
    A sampling regime for erosion events.
    """

    def __init__(self, seed=None, rng_contract=None):
        cases = [
            self.Event(
                name="Flat",
                p=0.20,
                processes=[_word(FlatUnconformity, rng_contract, "flat_00")],
            ),
            self.Event(
                name="Tilted",
                p=0.25,
                processes=[_word(TiltedUnconformity, rng_contract, "tilted_00")],
            ),
            self.Event(
                name="TiltCutFill",
                p=0.20,
                processes=[_word(TiltCutFill, rng_contract, "tilt_cut_fill_00")],
            ),
            self.Event(
                name="Wave",
                p=0.35,
                processes=[_word(WaveUnconformity, rng_contract, "wave_00")],
            ),
        ]
        super().__init__(cases=cases, seed=seed, rng_contract=rng_contract)


class Dike(_EventTemplateClass):
    """
    A sampling regime for intrusion events.
    """

    def __init__(self, seed=None, rng_contract=None):
        cases = [
            self.Event(
                name="Dike",
                p=0.4,
                processes=[_word(DikePlaneWord, rng_contract, "dike_00")],
            ),
            self.Event(
                name="WarpedDike",
                p=0.4,
                processes=[_word(SingleDikeWarped, rng_contract, "warped_dike_00")],
            ),
            self.Event(
                name="DikeGroup",
                p=0.2,
                processes=[_word(DikeGroup, rng_contract, "dike_group_00")],
            ),
        ]
        super().__init__(cases=cases, seed=seed, rng_contract=rng_contract)


class Sills(_EventTemplateClass):
    """
    A sampling regime for sill events.
    """

    def __init__(self, seed=None, rng_contract=None):
        cases = [
            self.Event(
                name="SillSingle",
                p=0.2,
                processes=[_word(SillWord, rng_contract, "sill_single_00")],
            ),
            # Note the sill system places a large sediment deposit at same time for embedding
            self.Event(
                name="SillSystem",
                p=0.8,
                processes=[_word(SillSystem, rng_contract, "sill_system_00")],
            ),
        ]
        super().__init__(cases=cases, seed=seed, rng_contract=rng_contract)


class Pluton(_EventTemplateClass):
    """
    A sampling regime for volcanic events.
    """

    def __init__(self, seed=None, rng_contract=None):
        cases = [
            self.Event(
                name="Laccolith",
                p=0.4,
                processes=[_word(Laccolith, rng_contract, "laccolith_00")],
            ),
            self.Event(
                name="Lopolith",
                p=0.6,
                processes=[_word(Lopolith, rng_contract, "lopolith_00")],
            ),
        ]
        super().__init__(cases=cases, seed=seed, rng_contract=rng_contract)


class OreDeposit(_EventTemplateClass):
    """
    A sampling regime for ore deposit events.

    Can be expanded to include more types in the future
    """

    def __init__(self, seed=None, rng_contract=None):
        cases = [
            self.Event(
                name="BlobCluster",
                p=1.0,
                processes=[_word(BlobCluster, rng_contract, "blob_cluster_00")],
            ),
        ]
        super().__init__(cases=cases, seed=seed, rng_contract=rng_contract)


class Fold(_EventTemplateClass):
    """
    A sampling regime for folding events.
    """

    def __init__(self, seed=None, rng_contract=None):
        cases = [
            self.Event(
                name="Simple",
                p=0.2,
                processes=[_word(SimpleFold, rng_contract, "simple_00")],
            ),
            self.Event(
                name="Shaped",
                p=0.3,
                processes=[_word(ShapedFold, rng_contract, "shaped_00")],
            ),
            self.Event(
                name="Fourier",
                p=0.5,
                processes=[_word(FourierFold, rng_contract, "fourier_00")],
            ),
        ]
        super().__init__(cases=cases, seed=seed, rng_contract=rng_contract)


class Fault(_EventTemplateClass):
    """A sampling regime for fault events."""

    def __init__(self, seed=None, rng_contract=None):
        cases = [
            self.Event(
                name="Normal", p=0.1, processes=[_word(FaultNormal, rng_contract, "normal_00")]
            ),
            self.Event(
                name="Reverse", p=0.1, processes=[_word(FaultReverse, rng_contract, "reverse_00")]
            ),
            self.Event(
                name="StrikeSlip", p=0.1, processes=[_word(FaultStrikeSlip, rng_contract, "strike_slip_00")]
            ),
            self.Event(
                name="HorstGraben", p=0.1, processes=[_word(FaultHorstGraben, rng_contract, "horst_graben_00")]
            ),
            self.Event(
                name="StrikeSlip", p=0.25, processes=[_word(FaultStrikeSlip, rng_contract, "strike_slip_01")]
            ),
            self.Event(
                name="FullyRandom", p=0.2, processes=[_word(FaultRandom, rng_contract, "fully_random_00")]
            ),
            self.Event(
                name="Sequence", p=0.15, processes=[_word(FaultSequence, rng_contract, "sequence_00")]
            ),
        ]
        super().__init__(cases=cases, seed=seed, rng_contract=rng_contract)
        
class Mountains(_EventTemplateClass):
    """A sampling regime for mountain events."""

    def __init__(self, seed=None, rng_contract=None):
        cases = [
            self.Event(
                name="TiltedMountains",
                p=1.0,
                processes=[_word(TiltedMountains, rng_contract, "tilted_mountains_00")],
            ),
        ]
        super().__init__(cases=cases, seed=seed, rng_contract=rng_contract)


# TODO: Implement Slip events in GeoWords and add to the Slip class
class Slip(_EventTemplateClass):
    """A sampling regime for slip events."""

    def __init__(self, seed=None, rng_contract=None):
        cases = [
            self.Event(
                name="Null",
                p=1.0,
                processes=[_word(NullWord, rng_contract, "null_00")],
            )
        ]
        super().__init__(cases=cases, seed=seed, rng_contract=rng_contract)
        NotImplementedError()


class End(_EventTemplateClass):
    """An ending flag for the geostory."""

    def __init__(self, seed=None, rng_contract=None):
        cases = [
            self.Event(
                name="Termination of Sequence",
                p=1.0,
                processes=[_word(NullWord, rng_contract, "termination_00")],
            )
        ]
        super().__init__(cases=cases, seed=seed, rng_contract=rng_contract)
