import numpy as np
from numpy.typing import NDArray

from typing import Any

TimeRange = tuple[float, float]
Args = dict[str, Any]


class SequenceGenerator:
    dim: int

    def __init__(self, dim, rng: np.random.Generator = None):
        self.dim = dim
        self._rng = rng if rng else np.random.default_rng()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def sample(self, time_range: TimeRange, delta: float) -> NDArray:
        return self._sample_impl(time_range, delta)

    def _sample_impl(self, time_range: TimeRange, delta: float) -> NDArray:
        del time_range, delta
        raise NotImplementedError


class Product(SequenceGenerator):
    def __init__(
        self, seq_gens: list[SequenceGenerator], rng: np.random.Generator = None
    ):
        super().__init__(len(seq_gens), rng)
        self._seq_gens = seq_gens

    def _sample_impl(self, time_range, delta):
        samples = tuple(g.sample(time_range, delta) for g in self._seq_gens)

        return np.hstack(samples)


class GaussianSequence(SequenceGenerator):
    def __init__(self, mean=0.0, std=1.0, dim=1, rng=None):
        super().__init__(dim, rng)

        self._mean = mean
        self._std = std

    def _sample_impl(self, time_range, delta):
        n_control_vals = int(
            1 + np.floor((time_range[1] - time_range[0]) / delta)
        )
        control_seq = self._rng.normal(
            loc=self._mean, scale=self._std, size=(n_control_vals, self.dim)
        )

        return control_seq


class GaussianSqWave(SequenceGenerator):
    def __init__(self, period, mean=0.0, std=1.0, dim=1, rng=None):
        super().__init__(dim, rng)

        self._period = period
        self._mean = mean
        self._std = std

    def _sample_impl(self, time_range, delta):
        n_control_vals = int(
            1 + np.floor((time_range[1] - time_range[0]) / delta)
        )

        n_amplitude_vals = int(np.ceil(n_control_vals / self._period))

        amp_seq = self._rng.normal(
            loc=self._mean, scale=self._std, size=(n_amplitude_vals, self.dim)
        )

        control_seq = np.repeat(amp_seq, self._period, axis=0)[:n_control_vals]
        return control_seq


class LogNormalSqWave(SequenceGenerator):
    def __init__(self, period, mean=0.0, std=1.0, dim=1, rng=None):
        super().__init__(dim, rng)

        self._period = period
        self._mean = mean
        self._std = std

    def _sample_impl(self, time_range, delta):
        n_control_vals = int(
            1 + np.floor((time_range[1] - time_range[0]) / delta)
        )

        n_amplitude_vals = int(np.ceil(n_control_vals / self._period))

        amp_seq = self._rng.lognormal(
            mean=self._mean, sigma=self._std, size=(n_amplitude_vals, self.dim)
        )

        control_seq = np.repeat(amp_seq, self._period, axis=0)[:n_control_vals]

        return control_seq


class UniformSqWave(SequenceGenerator):
    def __init__(self, period, min=0.0, max=1.0, dim=1, rng=None):
        super().__init__(dim, rng)

        self._period = period
        self._min = min
        self._max = max

    def _sample_impl(self, time_range, delta):
        n_control_vals = int(
            1 + np.floor((time_range[1] - time_range[0]) / delta)
        )

        n_amplitude_vals = int(np.ceil(n_control_vals / self._period))

        amp_seq = self._rng.uniform(
            low=self._min, high=self._max, size=(n_amplitude_vals, self.dim)
        )

        control_seq = np.repeat(amp_seq, self._period, axis=0)[:n_control_vals]

        return control_seq


class RandomWalkSequence(SequenceGenerator):
    def __init__(self, mean=0.0, std=1.0, dim=1, rng=None):
        super().__init__(dim, rng)

        self._mean = mean
        self._std = std

    def _sample_impl(self, time_range, delta):
        n_control_vals = int(
            1 + np.floor((time_range[1] - time_range[0]) / delta)
        )

        control_seq = np.cumsum(
            self._rng.normal(
                loc=self._mean, scale=self._std, size=(n_control_vals, self.dim)
            ),
            axis=1,
        )

        return control_seq


class SinusoidalSequence(SequenceGenerator):
    def __init__(self, max_freq=1.0, rng=None):
        super().__init__(1, rng)

        self._amp_mean = 1.0
        self._amp_std = 1.0
        self._mf = max_freq

    def _sample_impl(self, time_range, delta):
        amplitude = self._rng.lognormal(
            mean=self._amp_mean, sigma=self._amp_std
        )
        frequency = self._rng.uniform(0, self._mf)

        n_control_vals = int(
            1 + np.floor((time_range[1] - time_range[0]) / delta)
        )
        time = np.linspace(time_range[0], time_range[1], n_control_vals)

        return (amplitude * np.sin(np.pi * frequency / delta * time)).reshape(
            (-1, 1)
        )


_seqgen_names = {
    "GaussianSequence": GaussianSequence,
    "UniformSqWave": UniformSqWave,
    "GaussianSqWave": GaussianSqWave,
    "LogNormalSqWave": LogNormalSqWave,
    "RandomWalkSequence": RandomWalkSequence,
    "SinusoidalSequence": SinusoidalSequence,
}


def get_sequence_generator(name: str, args: Args | list[Args]):
    if name == "Product":
        if not isinstance(args, list):
            raise TypeError(
                "To construct a Product, args should be a list "
                "of (name, args) for each coordinate."
            )

        components = [
            get_sequence_generator(el["name"], el["args"]) for el in args
        ]
        return Product(components)

    else:
        if not isinstance(args, dict):
            raise TypeError(
                "args should be a dictionary of arguments to __init__."
            )

        return _seqgen_names[name](**args)
