# This file uses source code from the files strawberryfields/apps/similarity.py from
# https://github.com/XanaduAI/strawberryfields/blob/master/strawberryfields/apps/similarity.py,
# Copyright 2019 Xanadu Quantum Technologies Inc. licensed under the Apache 2.0 license.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter
from typing import Generator, Union

import networkx as nx
import numpy as np
from scipy.special import factorial

import strawberryfields as sf
from strawberryfields.backends import BaseGaussianState
from sympy.utilities.iterables import multiset_permutations


def sample_to_event(sample: list, max_count_per_mode: int) -> Union[int, None]:
    r"""Provides the event corresponding to a given sample.

    For an input ``max_count_per_mode``, events are expressed here simply by the total photon
    number :math:`k`.

    Args:
        sample (list[int]): a sample from GBS
        max_count_per_mode (int): the maximum number of photons counted in any given mode for a
            sample to be categorized as an event. Samples with counts exceeding this value are
            attributed the event ``None``.

    Returns:
        int or None: the event of the sample
    """
    if max(sample) <= max_count_per_mode:
        return sum(sample)

    return None


def sample_to_orbit(sample: list) -> list:
    """Provides the orbit corresponding to a given sample.

    Args:
        sample (list[int]): a sample from GBS

    Returns:
        list[int]: the orbit of the sample
    """
    return sorted(filter(None, sample), reverse=True)


def event_to_sample(photon_number: int, max_count_per_mode: int, modes: int) -> list:
    """Generates a sample selected uniformly at random from the specified event.

    Args:
        photon_number (int): number of photons in the event
        max_count_per_mode (int): maximum number of photons per mode in the event
        modes (int): number of modes in the sample

    Returns:
        list[int]: a sample in the event
    """
    if max_count_per_mode < 0:
        raise ValueError("Maximum number of photons per mode must be non-negative")

    if max_count_per_mode * modes < photon_number:
        raise ValueError(
            "No valid samples can be generated. Consider increasing the "
            "max_count_per_mode or reducing the number of photons."
        )

    cards = []
    orbs = []

    for orb in orbits(photon_number):
        if max(orb) <= max_count_per_mode:
            cards.append(orbit_cardinality(orb, modes))
            orbs.append(orb)

    norm = sum(cards)
    prob = [c / norm for c in cards]

    orbit = orbs[np.random.choice(len(prob), p=prob)]

    return orbit_to_sample(orbit, modes)


def orbit_to_sample(orbit: list, modes: int) -> list:
    """Generates a sample selected uniformly at random from the specified orbit.

    **Example usage:**

    >>> orbit_to_sample([2, 1, 1], 6)
    [0, 1, 2, 0, 1, 0]

    Args:
        orbit (list[int]): orbit to generate a sample from
        modes (int): number of modes in the sample

    Returns:
        list[int]: a sample in the orbit
    """
    if modes < len(orbit):
        raise ValueError("Number of modes cannot be smaller than length of orbit")

    sample = orbit + [0] * (modes - len(orbit))
    np.random.shuffle(sample)
    return sample


def prob_event_mc(
        graph: nx.Graph,
        photon_number: int,
        max_count_per_mode: int,
        n_mean: float = 5,
        samples: int = 1000,
        loss: float = 0.0,
) -> float:
    r"""Gives a Monte Carlo estimate of the probability of a given event for the input graph.

    To make this estimate, several samples from the event are drawn uniformly at random using
    :func:`event_to_sample`. The GBS probabilities of these samples are then calculated and the
    sum is used to create an estimate of the event probability.

    Args:
        graph (nx.Graph): input graph encoded in the GBS device
        photon_number (int): number of photons in the event
        max_count_per_mode (int): maximum number of photons per mode in the event
        n_mean (float): total mean photon number of the GBS device
        samples (int): number of samples used in the Monte Carlo estimation
        loss (float): fraction of photons lost in GBS

    Returns:
        float: Monte Carlo estimated event probability
    """
    if samples < 1:
        raise ValueError("Number of samples must be at least one")
    if n_mean < 0:
        raise ValueError("Mean photon number must be non-negative")
    if not 0 <= loss <= 1:
        raise ValueError("Loss parameter must take a value between zero and one")
    if photon_number < 0:
        raise ValueError("Photon number must not be below zero")
    if max_count_per_mode < 0:
        raise ValueError("Maximum number of photons per mode must be non-negative")

    modes = graph.order()
    state = _get_state(graph, n_mean, loss)

    prob = 0

    for _ in range(samples):
        sample = event_to_sample(photon_number, max_count_per_mode, modes)
        prob += state.fock_prob(sample, cutoff=photon_number + 1)

    prob = prob * event_cardinality(photon_number, max_count_per_mode, modes) / samples

    return prob


def prob_event_exact(
    graph: nx.Graph,
    photon_number: int,
    max_count_per_mode: int,
    n_mean: float = 5,
    loss: float = 0.0,
) -> float:
    r"""Gives the exact probability of a given event for the input graph.

    Events are made up of multiple orbits. To calculate an event probability, we can sum over
    the probabilities of its constituent orbits using :func:`prob_orbit_exact`.

    Args:
        graph (nx.Graph): input graph encoded in the GBS device
        photon_number (int): number of photons in the event
        max_count_per_mode (int): maximum number of photons per mode in the event
        n_mean (float): total mean photon number of the GBS device
        loss (float): fraction of photons lost in GBS

    Returns:
        float: exact event probability
    """

    if n_mean < 0:
        raise ValueError("Mean photon number must be non-negative")
    if not 0 <= loss <= 1:
        raise ValueError("Loss parameter must take a value between zero and one")
    if photon_number < 0:
        raise ValueError("Photon number must not be below zero")
    if max_count_per_mode < 0:
        raise ValueError("Maximum number of photons per mode must be non-negative")

    prob = 0

    for orbit in orbits(photon_number):
        if max(orbit) <= max_count_per_mode:
            prob += prob_orbit_exact(graph, orbit, n_mean, loss)
    return prob


def prob_orbit_exact(graph: nx.Graph, orbit: list, n_mean: float = 5, loss: float = 0.0) -> float:
    r"""Gives the exact probability of a given orbit for the input graph.

    The exact probability of an orbit is the sum of probabilities of
    all possible GBS output patterns that belong to it:

    .. math::
       p(O) = \sum_{S \in O} p(S)

    where :math:`S` are samples belonging to :math:`O`.

    Args:
        graph (nx.Graph): input graph encoded in the GBS device
        orbit (list[int]): orbit for which to calculate the probability
        n_mean (float): total mean photon number of the GBS device
        loss (float): fraction of photons lost in GBS

    Returns:
        float: exact orbit probability
    """

    if n_mean < 0:
        raise ValueError("Mean photon number must be non-negative")
    if not 0 <= loss <= 1:
        raise ValueError("Loss parameter must take a value between zero and one")

    modes = graph.order()
    photons = sum(orbit)
    state = _get_state(graph, n_mean, loss)

    click = orbit + [0] * (modes - len(orbit))
    prob = 0

    for pattern in multiset_permutations(click):
        prob += state.fock_prob(pattern, cutoff=photons + 1)

    return prob


def _get_state(graph: nx.Graph, n_mean: float = 5, loss: float = 0.0) -> BaseGaussianState:
    # TODO: Needs to be readjusted for Piquasso backend
    r"""Embeds the input graph into a GBS device and returns the corresponding Gaussian state."""
    modes = graph.order()
    A = nx.to_numpy_array(graph)
    mean_photon_per_mode = n_mean / float(modes)

    p = sf.Program(modes)

    # pylint: disable=expression-not-assigned
    with p.context as q:
        sf.ops.GraphEmbed(A, mean_photon_per_mode=mean_photon_per_mode) | q

        if loss:
            for _q in q:
                sf.ops.LossChannel(1 - loss) | _q

    eng = sf.LocalEngine(backend="gaussian")
    return eng.run(p).state


def event_cardinality(photon_number: int, max_count_per_mode: int, modes: int) -> int:
    r"""Gives the number of samples belonging to the input event.

    For example, for three modes, there are six samples in an :math:`E_{k=2, n_{\max}=2}` event:
    [1, 1, 0], [1, 0, 1], [0, 1, 1], [2, 0, 0], [0, 2, 0], and [0, 0, 2].

    Args:
        photon_number (int): number of photons in the event
        max_count_per_mode (int): maximum number of photons per mode in the event
        modes (int): number of modes in counted samples

    Returns:
        int: number of samples in the event
    """
    cardinality = 0

    for orb in orbits(photon_number):
        if max(orb) <= max_count_per_mode:
            cardinality += orbit_cardinality(orb, modes)

    return cardinality


def orbits(photon_number: int) -> Generator[list, None, None]:
    """Generate all the possible orbits for a given photon number.

    Provides a generator over the integer partitions of ``photon_number``.
    Code derived from `website <http://jeromekelleher.net/generating-integer-partitions.html>`__
    of Jerome Kelleher's, which is based upon an algorithm from Ref. :cite:`kelleher2009generating`.

    Args:
        photon_number (int): number of photons to generate orbits from

    Returns:
        Generator[list[int]]: orbits with total photon number adding up to ``photon_number``
    """
    a = [0] * (photon_number + 1)
    k = 1
    y = photon_number - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield sorted(a[: k + 2], reverse=True)
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield sorted(a[: k + 1], reverse=True)


def orbit_cardinality(orbit: list, modes: int) -> int:
    """Gives the number of samples belonging to the input orbit.

    For example, there are three possible samples in the orbit [2, 1, 1] with three modes: [1, 1,
    2], [1, 2, 1], and [2, 1, 1]. With four modes, there are 12 samples in total.

    Args:
        orbit (list[int]): orbit; we count how many samples are contained in it
        modes (int): number of modes in the samples

    Returns:
        int: number of samples in the orbit
    """
    sample = orbit + [0] * (modes - len(orbit))
    counts = list(Counter(sample).values())
    return int(factorial(modes) / np.prod(factorial(counts)))
