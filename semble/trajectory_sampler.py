from scipy.integrate import solve_ivp
import numpy as np
from numpy.typing import NDArray

from typing import Literal, TypedDict, NotRequired

from .dynamics import Dynamics, Dims, get_dynamics
from .sequence_generators import SequenceGenerator, get_sequence_generator, Args
from .initial_state import InitialStateGenerator, get_initial_state_generator


class TrajectorySampler:
    def __init__(
        self,
        dynamics: Dynamics,
        control_delta: float,
        control_generator: SequenceGenerator,
        method: str | None = None,
        initial_state_generator: InitialStateGenerator | None = None,
        seed: int | None = None,
    ):
        self._ode_method = dynamics.default_method if not method else method
        self._dyn = dynamics
        self._delta = control_delta  # control sampling time
        self._seq_gen = control_generator

        self._state_generator = (
            initial_state_generator
            if initial_state_generator
            else dynamics.default_initial_state()
        )

        self._rng = np.random.default_rng(seed=seed)
        self._seq_gen_rng, self._ist_rng = self._rng.spawn(2)

        self._init_time = 0.0

    def dims(self) -> Dims:
        return self._dyn.dims()

    def reset_rngs(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed=seed)
        self._seq_gen_rng, self._ist_rng = self._rng.spawn(2)

    def get_time_samples(
        self,
        time_horizon: float,
        n_samples: int,
        method: Literal["lhs", "linspace"],
    ) -> NDArray:
        if method == "lhs":
            t_samples = self._init_time + (
                time_horizon - self._init_time
            ) * lhs(n_samples, self._rng)
            t_samples = np.sort(np.append(t_samples, [self._init_time]))

        elif method == "linspace":
            t_samples = np.linspace(
                self._init_time, self._init_time + time_horizon, num=n_samples
            )
        else:
            raise ValueError(
                "Unsupported value for 'method' "
                "(should be one of 'lhs', 'linspace')"
            )

        return t_samples

    def sample_features(self, time_horizon: float) -> tuple[NDArray, NDArray]:
        x0 = self._state_generator.sample(self._ist_rng)

        u = self._seq_gen.sample(
            time_range=(self._init_time, time_horizon),
            delta=self._delta,
            rng=self._seq_gen_rng,
        )

        return x0, u

    def get_example(
        self,
        time_horizon: float,
        n_samples: int,
        time_sample_method: Literal["lhs", "linspace"] = "lhs",
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        t_samples = self.get_time_samples(
            time_horizon, n_samples, time_sample_method
        )

        x0, u = self.sample_features(time_horizon)

        def f(t, y):
            n_control = int(np.floor((t - self._init_time) / self._delta))
            u_val = u[n_control]  # get u(t)

            return self._dyn(y, u_val)

        traj = solve_ivp(
            f,
            (self._init_time, time_horizon),
            x0,
            t_eval=t_samples,
            method=self._ode_method,
        )

        x_traj = traj.y.T
        t = traj.t.reshape(-1, 1)

        return x0, t, x_traj, u


def lhs(n_samples: int, rng: np.random.Generator) -> NDArray:
    """Performs Latin Hypercube sampling on the unit interval."""
    bins_start_val = np.linspace(0.0, 1.0, n_samples + 1)[:-1]
    samples = (
        rng.uniform(size=(n_samples,)) / n_samples
    )  # sample a delta for each bin
    return bins_start_val + samples


class SpecEntry(TypedDict):
    name: str
    args: Args


class SequenceGeneratorSpec(TypedDict):
    name: str
    args: Args | list[Args]


class TSamplerSpec(TypedDict):
    dynamics: SpecEntry
    sequence_generator: SequenceGeneratorSpec
    initial_state_generator: NotRequired[SpecEntry]
    method: NotRequired[str]
    control_delta: float


def make_trajectory_sampler(args: TSamplerSpec) -> TrajectorySampler:
    dynamics = get_dynamics(args["dynamics"]["name"], args["dynamics"]["args"])

    sequence_generator = get_sequence_generator(
        args["sequence_generator"]["name"],
        args["sequence_generator"]["args"],
    )

    if "initial_state_generator" in args:
        init_state_gen = get_initial_state_generator(
            args["initial_state_generator"]["name"],
            args["initial_state_generator"]["args"],
        )
    else:
        init_state_gen = None

    sampler = TrajectorySampler(
        dynamics=dynamics,
        control_delta=args["control_delta"],
        control_generator=sequence_generator,
        method=args.get("method"),
        initial_state_generator=init_state_gen,
    )

    return sampler
