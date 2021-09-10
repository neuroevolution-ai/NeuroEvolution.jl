# NeuroEvolution.jl

*Neuroevolution on the GPU using the Julia programming language*

## Quick start

1. The NeuroEvolution.jl framework uses the CUDA.jl package, install this package first and make sure it is working: https://cuda.juliagpu.org/stable/

2. We use PyCall to integrate the covariance matrix adaptation evolution strategy (CMA-ES) from the deap package. Further, we modified the original deap package to use the GPU instead of the CPU to speed up the calculation of the eigenvalues in order to update the covariance matrix. To install our deap version, execute the following:

```bash
using Conda
Conda.pip_interop(true)
Conda.pip("install","git+https://github.com/neuroevolution-ai/deap.git@eigenvalues-on-gpu")
```

3. Execute the train.jl to start the training.