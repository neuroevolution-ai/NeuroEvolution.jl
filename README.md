# NeuroEvolution.jl

*Deep Neuroevolution on the GPU implemented in Julia*

## Quick start

1. The NeuroEvolution.jl framework uses the CUDA.jl package, install this package first and make sure it is working: https://cuda.juliagpu.org/stable/
2. Execute the train.jl to start the training.

## Running the Unit tests

You need our slightly modified Deap version for some unit tests, more precisely for executing the optimizers.jl script. This modified Deap version exposes some extra states of the CMA-ES algorithm, but the calculation itself does not differ from the original version. To install our modified Deap version do the following: 

1. Uninstall the deap package if already installed via `pip uninstall deap`
2. Run `pip install git+https://github.com/neuroevolution-ai/deap@test-cma-es-in-julia` 
3. Then the testing should work
