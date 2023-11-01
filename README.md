# Bayesian optimization of Sandia's FlameD OpenFOAM case

> This is part of the **MACHINE LEARNING IN COMBUSTION** workshop provided by NHR4CES' SDL Energy conversion.
> It serves as an illustration for the "Optimization of combustion processes with Bayesian algorithms" talk.

## Purpose

This optimization study is carried out to showcase the capabilities of [foamBO](https://github.com/FoamScience/OpenFOAM-Multi-Objective-Optimization). In particular, I focus on:
- **Ease of setup:** Users only need to prepare a `config.yaml` to describe their optimization workflow
- **Valuable insights:** with the "relative feature importances" 
- **Multi-objective optimization:** and studying the trade-offs between objectives
- **Parallel trial evaluation**

## How to run?

1. Clone this repository
2. `pip3 install foamBO`
3. Run `foamBO` in the root directory of this repository (where `config.yaml` is)
4. If you want to watch how the optimization is doing in a live feed, run `foamDash` in another terminal
   - This will show a live feed of optimization objectives as they come in
   - And once each trial is done, a screenshot of the final state of that particular simulation is shown

## TODO

- Run the trials with load-balanced adaptive mesh refinement.
- Run the trials on the HPC. This is easy, check [this example](https://github.com/FoamScience/OpenFOAM-Multi-Objective-Optimization/tree/main/examples/slurm) for inspiration.
- Add chemistry mechanisms
