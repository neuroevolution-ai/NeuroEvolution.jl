# Matrix multiplication in GPU Julia
# Code from here: https://discourse.julialang.org/t/cuda-kernel-error/31876

using CUDA

""" Compute C = A * B """
function kernel_matmul(C, A, B, m, p)
  tx = threadIdx().x

  for j in 1:1000
    Cvalue = 0.0f0

    if tx <= m
      for i = 1:p 
        Cvalue += A[i, tx] * B[i]
        #@cuprintln("tx $tx, i $i, res: $(A[tx, i] * B[i])")
      end
      C[tx] = Cvalue
      #@cuprintln(C[tx])
    end
  end

  
  
  return nothing
end

using Test

# Turn this on for speed
CUDA.allowscalar(false)

precision = Float32
m, p = 1000, 1000 # matrix sizes: C[m,1] = A[m,p] * B[p,1]
A, B, C = rand(precision,m,p), rand(precision,p), rand(precision,m)

Ad, Bd, Cd = CuArray(A), CuArray(B), CuArray(C)

# CPU
C = A*B

# CUBLAS
#Cd = Ad * Bd

CUDA.allowscalar(true)
#@test C == Cd

@cuda blocks=112 threads=m kernel_matmul(Cd', Ad', Bd', m, p)

#@time C = A*B

println("Finished")
