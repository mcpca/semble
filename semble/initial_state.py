import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator
from typing import Any


class InitialStateGenerator:
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def sample(self, rng: Generator) -> NDArray:
        return self._sample_impl(rng)

    def _sample_impl(self, rng: Generator):
        raise NotImplementedError


class GaussianInitialState(InitialStateGenerator):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def _sample_impl(self, rng: Generator):
        return rng.standard_normal(size=self.n)


class UniformInitialState(InitialStateGenerator):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def _sample_impl(self, rng: Generator):
        return rng.uniform(size=self.n)


class HHFSInitialState(InitialStateGenerator):
    def __init__(self):
        super().__init__()

    def _sample_impl(self, rng: Generator):
        x0 = rng.uniform(size=(4,))
        x0[0] = 2.0 * x0[0] - 1.0

        return x0


class HHRSAInitialState(InitialStateGenerator):
    def __init__(self):
        super().__init__()

    def _sample_impl(self, rng: Generator):
        x0 = rng.uniform(size=(5,))
        x0[0] = 2.0 * x0[0] - 1.0

        return x0


class HHFFEInitialState(InitialStateGenerator):
    def __init__(self):
        super().__init__()

    def _sample_impl(self, rng: Generator):
        x0 = rng.uniform(size=(10,))
        x0[0] = 2.0 * x0[0] - 1.0
        x0[5] = 2.0 * x0[5] - 1.0

        return x0


class HHFBEInitialState(InitialStateGenerator):
    def __init__(self):
        super().__init__()

    def _sample_impl(self, rng: Generator):
        x0 = rng.uniform(size=(11,))
        x0[0] = 2.0 * x0[0] - 1.0
        x0[5] = 2.0 * x0[5] - 1.0

        return x0


class HHIBInitialState(InitialStateGenerator):
    def __init__(self):
        super().__init__()

    def _sample_impl(self, rng: Generator):
        x0 = rng.uniform(size=(7,))
        x0[0] = 2.0 * x0[0] - 1.0

        return x0


class GreenshieldsInitialState(InitialStateGenerator):
    def __init__(self, n_cells, n_sections):
        super().__init__()

        self.n_cells = n_cells
        self.n_sec = n_sections
        self.sec_size = self.n_cells // self.n_sec

    def _sample_impl(self, rng: Generator):
        x0_vals = rng.uniform(0.0, 0.5, size=(self.n_sec,))
        x0 = np.empty((self.n_cells,))
        x0[0 : self.sec_size * self.n_sec] = np.repeat(x0_vals, self.sec_size)
        x0[self.sec_size * self.n_sec : -1] = x0[self.sec_size * self.n_sec - 1]

        return x0


_initstategen_names = {
    "GaussianInitialState": GaussianInitialState,
    "UniformInitialState": UniformInitialState,
    "HHFSInitialState": HHFSInitialState,
    "HHRSAInitialState": HHRSAInitialState,
    "HHIBInitialState": HHIBInitialState,
    "HHFFEInitialState": HHFFEInitialState,
    "HHFBEInitialState": HHFBEInitialState,
    "GreenshieldsInitialState": GreenshieldsInitialState,
}


def get_initial_state_generator(
    name: str, args: dict[str, Any]
) -> InitialStateGenerator:
    return _initstategen_names[name](**args)
