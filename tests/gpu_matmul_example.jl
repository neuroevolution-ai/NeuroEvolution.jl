# Matrix multiplication in GPU Julia
# Code from here: https://discourse.julialang.org/t/cuda-kernel-error/31876
# Also interesting: https://discourse.julialang.org/t/casting-annotations-and-numeric-types-for-cudanative/19853/2

using CUDA

""" Compute C = A * B """
function kernel_matmul_fast(C, A, B, m, p)
  tx = threadIdx().x

  sA = @cuDynamicSharedMem(Float32, (m,p))
  sB = @cuDynamicSharedMem(Float32, p)

  for j in 1:p
    sA[tx, j] = A[tx, j]
  end

  if tx == 1
    for j in 1:p
        sB[j] = B[j]
    end
  end

  #if blockIdx().x == 1
  #  for j in 1:p
  #    @cuprintln("A: $(A[tx, j])")
  #    @cuprintln("sA: $(sA[tx, j])")
  #  end
  #end

  # Wait until all threads finish preloading
  sync_threads()

  for j in 1:2000
    Cvalue = 0.0f0

    if tx <= m
      for i = 1:p 
        Cvalue += sA[tx, i] * sB[i]
        #@cuprintln("tx $tx, i $i, res: $(A[tx, i] * B[i])")
      end
      C[tx] = Cvalue
      #@cuprintln(C[tx])
    end
  end
  
  return nothing
end

""" Compute C = A * B """
function kernel_matmul(C, A, B, m, p)
  tx = threadIdx().x
  for index in 1:1000
  for j in 1:2000
    Cvalue = 0.0f0

    if tx <= m
      for i = 1:p 
        Cvalue += A[tx, i] * B[i]
        #@cuprintln("tx $tx, i $i, res: $(A[tx, i] * B[i])")
      end
      C[tx] = Cvalue
      #@cuprintln(C[tx])
    end
  end
  end
  return nothing
end

using Test

# Turn this on for speed
CUDA.allowscalar(false)

precision = Float32
m, p = 100, 100 # matrix sizes: C[m,1] = A[m,p] * B[p,1]
A, B, C = rand(precision,m,p), rand(precision,p), rand(precision,m)

Ad, Bd, Cd = CuArray(A), CuArray(B), CuArray(C)

# CPU
C = A*B
# CUBLAS
#Cd = Ad * Bd

CUDA.allowscalar(true)
#@test C == Cd
#for i in 1:100
  #@cuda blocks=112 threads=m shmem=sizeof(Float32)*m*p kernel_matmul(Cd, Ad, Bd, m, p)
  #global Cd = Ad * Bd
  @cuda blocks=112 threads=m kernel_matmul(Cd, Ad, Bd, m, p)
  CUDA.synchronize()
#end
#@time C = A*B



#println(Cd)
#println(A*B)

println("Finished")

