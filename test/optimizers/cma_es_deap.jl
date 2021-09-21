using PyCall
using Conda


mutable struct OptimizerCmaEsDeap
    opt::Any
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

    function OptimizerCmaEsDeap(individual_size::Int, configuration::Dict)
        scriptdir = @__DIR__
        pushfirst!(PyVector(pyimport("sys")."path"), scriptdir)
        optimizer_deap = pyimport("cma_es_deap")
        opt = optimizer_deap.OptimizerCmaEsDeap(individual_size, configuration)

        lambda_ = opt.strategy.lambda_
        dim = opt.strategy.dim
        chiN = opt.strategy.chiN
        mu = opt.strategy.mu
        weights = opt.strategy.weights
        mueff = opt.strategy.mueff
        cc = opt.strategy.cc
        cs = opt.strategy.cs
        ps = opt.strategy.ps
        pc = opt.strategy.pc
        centroid = opt.strategy.centroid
        update_count = opt.strategy.update_count
        ccov1 = opt.strategy.ccov1
        ccovmu = opt.strategy.ccovmu
        C = opt.strategy.C
        sigma = opt.strategy.sigma
        damps = opt.strategy.damps
        diagD = opt.strategy.diagD
        B = opt.strategy.B
        BD = opt.strategy.BD
        eigenvectors = opt.strategy.eigenvectors
        indx = opt.strategy.indx
        genomes = zeros(opt.strategy.lambda_, individual_size)

        optimizer = new(opt, lambda_, dim, chiN, mu, weights, mueff, cc, cs, ps, pc, centroid, update_count, ccov1, ccovmu, C, sigma, damps, diagD, B, BD, genomes)

        return optimizer, eigenvectors, indx .+ 1
    end
end

function ask(optimizer)

    genomes_list, population_size, individual_size, strategy = optimizer.opt.ask()

    # The genomes need to be reshaped into a MxN matrix.
    for i = 1:population_size
        for j = 1:individual_size
            optimizer.genomes[i, j] = (genomes_list[i])[j]
        end
    end

    return optimizer.genomes, strategy.randoms
end

function tell(optimizer, rewards)

    strategy = optimizer.opt.tell(rewards)

    optimizer.centroid = strategy.centroid
    optimizer.ps = strategy.ps
    optimizer.update_count = strategy.update_count
    optimizer.pc = strategy.pc
    optimizer.C = strategy.C
    optimizer.sigma = strategy.sigma
    optimizer.diagD = strategy.diagD
    optimizer.B = strategy.B
    optimizer.BD = strategy.BD

    return strategy.eigenvectors, strategy.indx .+ 1

end
