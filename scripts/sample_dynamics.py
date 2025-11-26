import argparse, sys

import semble

import yaml
from matplotlib import pyplot as plt
import numpy as np

from typing import cast


def main(args):
    with open(args.spec, "r") as f:
        spec: semble.TSamplerSpec = yaml.load(f, Loader=yaml.FullLoader)

    sampler = semble.make_trajectory_sampler(spec)
    sampler.reset_rngs()
    n_dims = sampler.dims()[-1]
    dynamics = sampler._dyn

    if args.continuous_state:
        dynamics = cast(semble.dynamics.ContinuousStateDynamics, dynamics)
        state_axis = dynamics.get_space_axis()
        n_plots = 2
    else:
        state_axis = None
        n_plots = n_dims + 1

    fig, ax = plt.subplots(n_plots, 1, sharex=True)
    fig.canvas.mpl_connect("close_event", on_close_window)
    plt.ion()

    while True:
        _, t, y, u = sampler.get_example(
            args.time_horizon,
            n_samples=int(10 * args.time_horizon),
            time_sample_method="linspace",
        )

        if not args.continuous_state:
            for k in range(n_plots - 1):
                ax[k].plot(t, y[:, k])
                ax[k].set_ylabel(r"$x_{}$".format(k))
        else:
            ax[0].pcolormesh(t.squeeze(), state_axis, y.T)
            ax[0].set_ylabel(r"$y$")

        ax[-1].step(
            np.arange(0.0, args.time_horizon, spec["control_delta"]),
            u[:-1],
            where="post",
        )
        ax[-1].set_ylabel(r"$u$")
        ax[-1].set_xlabel(r"$t$")
        plt.draw()

        # Wait for key press
        skip = False
        while not skip:
            skip = plt.waitforbuttonpress()

        for ax_ in ax:
            ax_.clear()


def on_close_window(_):
    sys.exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "spec",
        type=str,
        help="Path to YAML file describing the trajectory sampler.",
    )
    ap.add_argument(
        "time_horizon", type=float, help="Length of the sampled trajectories"
    )
    ap.add_argument(
        "--continuous_state",
        action="store_true",
        help="Whether the state is a continuous variable",
    )

    main(ap.parse_args())
