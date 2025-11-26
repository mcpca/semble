# semble
## A simple package for simulating control systems.

This is a utility package to easily generate synthetic datasets of
(continuous-time) control system trajectories in a reproducible manner.

The architecture is based on composing a control system of the form
$\dot{x} = f(x, u)$ with probability distributions for the initial condition
$x_0 = x(0)$ and the control input $u$.
Together, these specify a distribution on system trajectories which can be
sampled from.
`semble` allows one to specify such a distribution in a simple human-readable
text format.

## Example use

An example of how to use the package is provided in
[scripts/sample_dynamics.py](scripts/sample_dynamics.py).
A `TrajectorySampler` object may be created using the helper function
`make_trajectory_sampler` which takes a `TSamplerSpec` type dictionary.
This can be created using any format which may be read into a Python `dict`,
as long as the fields conform to the definition of `TSamplerSpec`.
For instance, the following yaml file:
```yaml
dynamics:
  name: VanDerPol
  args:
    damping: 1.0
sequence_generator:
  name: GaussianSqWave
  args:
    period: 1
control_delta: 0.5
initial_state:
  name: GaussianInitialState
  args:
    n: 2
```
defines a Van der Pol system with control amplitudes sampled from a standard
normal distribution every 0.5 seconds, and initial state sampled from a standard
normal distribution.
If we save its contents as `example_vdp_spec.yaml`, we can sample and plot the
resulting trajectories by running
```sh
python scripts/sample_dynamics.py example_vdp_spec.yaml 15
```
where the second argument is the time length of the trajectories.

## Scope

The list of available dynamical systems is in [semble/dynamics.py](semble/dynamics.py).
These are heavily dependent on my own needs for other projects,
but contributions are welcome as long as they conform to the structure of
the existing `Dynamics` classes and do not introduce additional dependencies.

Distributions for sampling control inputs and initial states are in
[sequence_generators.py](semble/sequence_generators.py)
and [initial_state.py](semble/initial_state.py), respectively.
At the moment, only piecewise control inputs are supported.

`Dynamics` subclasses can declare default initial state distributions and integration methods.
[Scipy's `solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
is used for integration, so the method string must be a valid input to this function.
If these are not defined in the class definition or in the specification,
standard normal initial state and RK45 integrator are used by default.

## Python version and dependencies

Development targets Python 3.11, numpy 1.26 and scipy 1.15 so as to support
platforms with older toolchains.
