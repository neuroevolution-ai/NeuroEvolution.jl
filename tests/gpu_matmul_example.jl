# Matrix multiplication in GPU Julia
# Code from here: https://discourse.julialang.org/t/cuda-kernel-error/31876
# Also interesting: https://discourse.julialang.org/t/casting-annotations-and-numeric-types-for-cudanative/19853/2

using CUDA
#using Test

""" Compute C = A * B """
function kernel_matmul_fast(C, A, B, m, p)
  tx = threadIdx().x

  # Important: The second @cuDynamicSharedMem allocation needs an offset of sizeof(sA), as it uses a single kernel-level buffer
  sA = @cuDynamicSharedMem(Float32, (m,p))
  #sA = @cuStaticSharedMem(Float32, (m,p))
  sB = @cuDynamicSharedMem(Float32, p)
  #sB = @cuStaticSharedMem(Float32, p)

  # Initialize shared memory for A
  for j in 1:p
    @inbounds  sA[tx, j] = A[tx, j]
  end

  # Initialize shared memory for B
  if tx == 1
    for j in 1:p
        sB[j] = B[j]
    end
  end

  # Wait until all threads finish preloading
  sync_threads()

  for j in 1:2000
    Cvalue = 0.0f0

    if tx <= m
      for i = 1:p 
        Cvalue += sA[tx, i] * sB[i]
        #@cuprintln("tx $tx, i $i, res: $(A[tx, i] * B[i])")
      end
      @inbounds  C[tx] = Cvalue
      #@cuprintln(C[tx])
    end
  end
  return nothing
end

""" Compute C = A * B """
function kernel_matmul(C, A, B, m, p)
  tx = threadIdx().x

  for j in 1:2000
    Cvalue = 0.0f0

    if tx <= m
      for i = 1:p 
        Cvalue += A[tx, i] * B[i]
      end
      @inbounds C[tx] = Cvalue
    end
  end
  
  return nothing
end

m, p = 100, 100 # matrix sizes: C[m,1] = A[m,p] * B[p,1]

A, B, C = rand(Float32,m,p), rand(Float32,p), rand(Float32,m)
Ad, Bd, Cd = CuArray(A), CuArray(B), CuArray(C)

# CPU
#C = A*B

# CUBLAS
#Cd = Ad * Bd

#@test C == Cd
for i in 1:100
  #@cuda blocks=112 threads=m kernel_matmul_fast(Cd, Ad, Bd, m, p)
  @cuda blocks=112 threads=m shmem=sizeof(Float32)*(m+1)*p kernel_matmul_fast(Cd, Ad, Bd, m, p)
  #@cuda blocks=112 threads=m kernel_matmul(Cd, Ad, Bd, m, p)
  CUDA.synchronize()
end

#@time C = A*B

#println(Cd)
#println(A*B)

println("Finished")
