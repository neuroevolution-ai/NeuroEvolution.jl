using LinearAlgebra
using CUDA
using Distributions
using Random

struct OptimizerCmaEsCfg
    population_size::Int
    sigma::Float64

    function OptimizerCmaEsCfg(configuration::OrderedDict)
        new(configuration["population_size"], configuration["sigma"])
    end
end

mutable struct OptimizerCmaEs
    lambda_::Any
    dim::Any
    chiN::Any
    mu::Any
    weights::Any
    mueff::Any
    cc::Any
    cs::Any
    ps::Any
    pc::Any
    centroid::Any
    update_count::Any
    ccov1::Any
    ccovmu::Any
    C::Any
    sigma::Any
    damps::Any
    diagD::Any
    B::Any
    BD::Any
    genomes::Any

    function OptimizerCmaEs(individual_size::Int, optimizer_configuration::OrderedDict; test = false, eigenvectors1 = Nothing, indx1 = Nothing)

        config = OptimizerCmaEsCfg(optimizer_configuration)

        centroid = zeros(individual_size)

        dim = individual_size
        pc = zeros(dim)
        ps = zeros(dim)
        chiN = sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim^2))

        C = Matrix(1.0I, dim, dim)

        C_GPU = CuArray(C)
        val_GPU, vec_GPU = CUDA.CUSOLVER.syevd!('V', 'U', C_GPU)
        diagD = Array(val_GPU)
        B = Array(vec_GPU)

        indx = sortperm(diagD)

        if test == true
            eigenvectors_test = copy(B)
            B = copy(eigenvectors1)
            indx_test = copy(indx)
            indx = copy(indx1)
            diagD_test = copy(diagD)
        else
            eigenvectors_test = Nothing
            indx_test = Nothing
            diagD_test = Nothing
        end

        diagD = diagD[indx] .^ 0.5
        B = B[:, indx]
        BD = B .* diagD'

        lambda_ = config.population_size
        update_count = 0

        # Compute params
        mu = Int(lambda_ / 2)
        weights = log(mu + 0.5) .- log.(collect(1:mu))
        weights /= sum(weights)
        mueff = 1 / sum(weights .^ 2)
        cc = 4 / (dim + 4)
        cs = (mueff + 2) / (dim + mueff + 3)
        ccov1 = 2 / ((dim + 1.3)^2 + mueff)
        ccovmu = 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)^2 + mueff)
        ccovmu = min(1 - ccov1, ccovmu)
        damps = 1 + 2 * max(0, sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        genomes = zeros(lambda_, individual_size)

        optimizer = new(lambda_, dim, chiN, mu, weights, mueff, cc, cs, ps, pc, centroid, update_count, ccov1, ccovmu, C, config.sigma, damps, diagD, B, BD, genomes)

        return optimizer, eigenvectors_test, indx_test, diagD_test
    end
end

function ask(optimizer::OptimizerCmaEs; test = false, randoms1 = Nothing)

    randoms = rand(Normal(), size(optimizer.genomes))

    if test == true
        randoms_test = copy(randoms)
        randoms = copy(randoms1)
    else
        randoms_test = Nothing
    end

    optimizer.genomes = optimizer.centroid' .+ (optimizer.sigma .* (randoms * optimizer.BD'))

    return optimizer.genomes, randoms_test

end

function tell(optimizer::OptimizerCmaEs, rewards_training; test = false, eigenvectors1 = Nothing, indx1 = Nothing)

    genomes_sorted = optimizer.genomes[sortperm(rewards_training, rev = true), :]

    old_centroid = copy(optimizer.centroid)
    optimizer.centroid = genomes_sorted[1:optimizer.mu, :]' * optimizer.weights

    c_diff = optimizer.centroid - old_centroid

    # Cumulation : update evolution path
    optimizer.ps =
        (1 - optimizer.cs) .* optimizer.ps +
        sqrt(optimizer.cs * (2 - optimizer.cs) * optimizer.mueff) ./ optimizer.sigma *
        optimizer.B *
        ((1 ./ optimizer.diagD) .* optimizer.B' * c_diff)

    hsig = float(
        norm(optimizer.ps) /
        sqrt(1.0 - (1 - optimizer.cs)^(2 * (optimizer.update_count + 1))) /
        optimizer.chiN < (1.4 + 2 / (optimizer.dim + 1)),
    )

    optimizer.update_count += 1

    optimizer.pc =
        (1 - optimizer.cc) * optimizer.pc +
        hsig * sqrt(optimizer.cc * (2 - optimizer.cc) * optimizer.mueff) / optimizer.sigma *
        c_diff

    # Update covariance matrix
    artmp = genomes_sorted[1:optimizer.mu, :]' .- old_centroid

    optimizer.C =
        (
            1 - optimizer.ccov1 - optimizer.ccovmu +
            (1 - hsig) * optimizer.ccov1 * optimizer.cc * (2 - optimizer.cc)
        ) * optimizer.C +
        optimizer.ccov1 * optimizer.pc * optimizer.pc' +
        optimizer.ccovmu * (optimizer.weights' .* artmp) * artmp' / optimizer.sigma^2

    optimizer.sigma *=
        exp((norm(optimizer.ps) / optimizer.chiN - 1) * optimizer.cs / optimizer.damps)

    C_GPU = CuArray(optimizer.C)
    val_GPU, vec_GPU = CUDA.CUSOLVER.syevd!('V', 'U', C_GPU)
    optimizer.diagD = Array(val_GPU)
    optimizer.B = Array(vec_GPU)

    indx = sortperm(optimizer.diagD)

    if test == true
        eigenvectors_test = copy(optimizer.B)
        optimizer.B = copy(eigenvectors1)
        indx_test = copy(indx)
        indx = copy(indx1)
        diagD_test = copy(optimizer.diagD)
    else
        eigenvectors_test = Nothing
        indx_test = Nothing
        diagD_test = Nothing
    end

    optimizer.diagD = optimizer.diagD[indx] .^ 0.5
    optimizer.B = optimizer.B[:, indx]
    optimizer.BD = optimizer.B .* optimizer.diagD'

    return eigenvectors_test, indx_test, diagD_test

end
