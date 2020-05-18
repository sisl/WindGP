using GaussianProcesses
using ElasticArrays
using PDMats
using NearestNeighbors
using LinearAlgebra
using ForwardDiff
using Optim

mutable struct GPLA{X<:AbstractMatrix, Y<:AbstractVector, M<:GaussianProcesses.Mean, K<:GaussianProcesses.Kernel, NOI<:GaussianProcesses.Param} <: GaussianProcesses.GPBase
    x::X
    y::Y
    k::Int64 #Local Neighborhood Size

    action_dims::Int64
    state_dims::Int64
    dim::Int64
    mean::M
    kernel::K
    logNoise::NOI

    Kfull::Dict{Tuple{Int64, Int64}, Float64}
    Klocal::Matrix{Float64}

    kdtree::Union{Nothing, KDTree}

    "Auxiliary variables used to optimize GP hyperparameters"
    data::GaussianProcesses.KernelData
    mll::Float64
    dmll::Vector{Float64}
    # target::Float64
    # dtarget::Vector{Float64}
    function GPLA{X, Y, M, K, NOI}(x::X, y::Y, k::Int64, action_dims::Int64, state_dims::Int64, mean::M, kernel::K, logNoise::NOI) where {X, Y, M, K, NOI}
        Kfull = Dict{Tuple{Int64, Int64}, Float64}()
        Klocal = Matrix{Float64}(undef, k, k)
        data = GaussianProcesses.KernelData(kernel, x, x)
        gp = new{X, Y, M, K, NOI}(ElasticArray(x), ElasticArray(y), k, action_dims, state_dims, action_dims+state_dims, mean, kernel, logNoise, Kfull, Klocal, nothing, data)
        initialize!(gp)
    end
end

function GPLA(x::AbstractMatrix, y::AbstractVector, k::Integer, action_dims::Integer, state_dims::Integer, mean::GaussianProcesses.Mean, kernel::GaussianProcesses.Kernel, logNoise::Real)
    lns = GaussianProcesses.wrap_param(logNoise)
    GPLA{typeof(x),typeof(y),typeof(mean),typeof(kernel), typeof(lns)}(x, y, k, action_dims, state_dims, mean, kernel, lns)
end

function initialize!(gp::GPLA)
    n_obs = size(gp.y, 1)
    if n_obs != 0
        for i = 1:n_obs
            for j = i:n_obs
                if i == j
                    data = GaussianProcesses.KernelData(gp.kernel, gp.x[:,i:i], gp.x[:,j:j])
                    gp.Kfull[(i, j)] = GaussianProcesses.cov(gp.kernel, gp.x[:,i:i], gp.x[:,j:j], data)[1] + exp(2*gp.logNoise.value)+eps()
                else
                    data = GaussianProcesses.KernelData(gp.kernel, gp.x[:,i:i], gp.x[:,j:j])
                    gp.Kfull[(i, j)] = GaussianProcesses.cov(gp.kernel, gp.x[:,i:i], gp.x[:,j:j], data)[1]
                end
            end
        end
        gp.kdtree = KDTree(gp.x)
        update_mll!(gp)
    end
    return gp
end

function update_K!(gp::GPLA)
    n_obs = size(gp.y, 1)
    if n_obs != 0
        for i = 1:n_obs
            for j = i:n_obs
                if i == j
                    data = GaussianProcesses.KernelData(gp.kernel, gp.x[:,i:i], gp.x[:,j:j])
                    gp.Kfull[(i, j)] = GaussianProcesses.cov(gp.kernel, gp.x[:,i:i], gp.x[:,j:j], data)[1] + exp(2*gp.logNoise.value)+eps()
                else
                    data = GaussianProcesses.KernelData(gp.kernel, gp.x[:,i:i], gp.x[:,j:j])
                    gp.Kfull[(i, j)] = GaussianProcesses.cov(gp.kernel, gp.x[:,i:i], gp.x[:,j:j], data)[1]
                end
            end
        end
    end
    return gp
end

function mll_local(Kfull, ys, ms, idx, neighbors)
    neighbors = neighbors[neighbors .!= idx]
    k = length(neighbors)
    Kxx = zeros(Float64, k, k)
    Kxf = zeros(Float64, k)
    y_obs = zeros(Float64, k)
    Kff = Kfull[(idx, idx)]
    mf = ms[idx]
    for i  = 1:k
        for j in i:k
            i_idx = min(neighbors[i], neighbors[j])
            j_idx = max(neighbors[i], neighbors[j])
            Kxx[i, j] = Kfull[(i_idx, j_idx)]
            Kxx[j, i] = Kfull[(i_idx, j_idx)]
        end
        i_idx = min(idx, neighbors[i])
        j_idx = max(idx, neighbors[i])
        Kxf[i] = Kfull[(i_idx, j_idx)]
        y_obs[i] = ys[neighbors[i]] - ms[neighbors[i]]
    end
    Kxx = PDMat(GaussianProcesses.make_posdef!(Kxx)...)
    μ = mf + GaussianProcesses.dot(Kxf, Kxx \ y_obs)
    Σ = Kff - GaussianProcesses.dot(Kxf, Kxx \ Kxf)
    σ² = max(Σ, 0.0)
    σ = sqrt(σ²)
    log_p = -0.5*((ys[idx] - μ)/σ)^2 - 0.5*log(2*pi) - log(σ)
    param_tuple = (μ = μ, σ²  = σ², Kxx = Kxx, Kxf = Kxf, Kff = Kff, mf = mf, y = y_obs, neighbors = neighbors)
    return log_p, param_tuple
end

function make_K(Kfull_dict, nobs::Int64)
    Kfull = zeros(Float64, nobs, nobs)
    for (k,v) in Kfull_dict
        i = k[1]
        j = k[2]
        Kfull[i, j] = v
        if i != j
            Kfull[j, i] = v
        end
    end
    return Kfull
end

function update_mll!(gp::GPLA)
    nx = size(gp.x, 2)
    μ = GaussianProcesses.mean(gp.mean, gp.x)
    if nx <= gp.k
        y = gp.y - μ
        Kfull = make_K(gp.Kfull, nx)
        cK = PDMat(GaussianProcesses.make_posdef!(Kfull)...)
        α = cK \ y
        gp.mll = - (GaussianProcesses.dot(y, α) + GaussianProcesses.logdet(cK) + log(2*pi) * nx) / 2
    else
        neighbors, _ = knn(gp.kdtree, extract_value(gp.x), gp.k + 1)
        gp.mll = 0.0
        for i = 1:nx
            log_p, test = mll_local(gp.Kfull, gp.y, μ, i, neighbors[i])
            gp.mll += log_p
        end
    end
end

"""
     update_dmll!(gp::GPE, ...)
Update the gradient of the marginal log-likelihood of Gaussian process `gp`.
"""
function update_dmll!(gp::GPLA;
                    noise::Bool=true, # include gradient component for the logNoise term
                    domean::Bool=true, # include gradient components for the mean parameters
                    kern::Bool=true, # include gradient components for the spatial kernel parameters
                    )
    n_mean_params = GaussianProcesses.num_params(gp.mean)
    n_kern_params = GaussianProcesses.num_params(gp.kernel)
    gp.dmll = zeros(Float64, noise + domean * n_mean_params + kern * n_kern_params)
    nobs = size(gp.y, 1)
    μ = GaussianProcesses.mean(gp.mean, gp.x)
    neighbors, _ = knn(gp.kdtree, gp.x, gp.k + 1)
    for i = 1:nobs
        _, params = mll_local(gp.Kfull, gp.y, μ, i, neighbors[i])
        d_dmll = zeros(Float64, noise + domean * n_mean_params + kern * n_kern_params)
        dmll_local!(d_dmll, gp, i, n_mean_params, n_kern_params, params; noise = noise, domean = domean, kern = kern)
        gp.dmll += d_dmll
    end
end

function dmll_local!(dmll::AbstractVector, gp::GPLA, idx::Int64, n_mean_params, n_kern_params, params;
                noise::Bool=true, domean::Bool=true, kern::Bool=true)
    i=1
    if noise
        @assert GaussianProcesses.num_params(gp.logNoise) == 1
        dmll[i] = dmll_noise(gp, idx, params)
        i += 1
    end
    if domean && n_mean_params>0
        dmll_m = @view(dmll[i:i+n_mean_params-1])
        dmll_mean!(dmll_m, gp, idx, params)
        i += n_mean_params
    end
    if kern
        dmll_k = @view(dmll[i:end])
        dmll_kern!(dmll_k, gp, idx, params)
    end
    return dmll
end

function dmll_noise(gp::GPLA, idx::Int64, params::NamedTuple)
    y = gp.y[idx]
    Knoise = diagm(repeat([2*exp(2*gp.logNoise.value)], gp.k))
    dμ = -dot(params.Kxf, params.Kxx \ Knoise * (params.Kxx \ params.y))
    dσ = dot(params.Kxf, params.Kxx \ Knoise * (params.Kxx \ params.Kxf)) + 2*exp(2*gp.logNoise.value)
    dσ /= sqrt(params.σ²)*2.0
    dlog_p = -(y - params.μ)/sqrt(params.σ²)*(-dμ/sqrt(params.σ²) - (y - params.μ)/params.σ²*dσ)
    dlog_p -= dσ/sqrt(params.σ²)
    return dlog_p
end

function dmll_mean!(dmll::AbstractVector, gp::GPLA, idx::Int64, params::NamedTuple)
    y = gp.y[idx]
    x = gp.x[:,idx:idx]
    d_mean = -ones(Float64, gp.k)
    dμ = GaussianProcesses.grad_stack(gp.mean, x)*(1.0 + GaussianProcesses.dot(params.Kxf, params.Kxx \ d_mean))
    dσ = [0.0]
    dlog_p = -(y - params.μ)/sqrt(params.σ²)*(-dμ/sqrt(params.σ²) - (y - params.μ)/params.σ²*dσ)
    dlog_p -= dσ/sqrt(params.σ²)

    for i in 1:length(dlog_p)
        dmll[i] = dlog_p[i]
    end
    return dmll
end

function dmll_kern!(dmll::AbstractVector, gp::GPLA, idx::Int64, params::NamedTuple)
    y = gp.y[idx]
    x = gp.x[:,idx:idx]
    dKxx = grad_stack(gp.kernel, gp.x[:,params.neighbors], gp.x[:,params.neighbors])
    dKff = grad_stack(gp.kernel, x, x)[1, 1, :]
    dKxf = grad_stack(gp.kernel, x, gp.x[:,params.neighbors])[1,:,:]
    for i in 1:size(dKxf,2)
        dμ = GaussianProcesses.dot(dKxf[:,i], params.Kxx \ params.y)
        dμ -= GaussianProcesses.dot(params.Kxf, (params.Kxx \ dKxx[:,:,i]) * (params.Kxx \ params.y))

        dσ = dKff[i] - GaussianProcesses.dot(dKxf[:,i], params.Kxx \ params.Kxf)
        dσ += GaussianProcesses.dot(params.Kxf, (params.Kxx \ dKxx[:,:,i]) * (params.Kxx \ params.Kxf))
        dσ -= GaussianProcesses.dot(params.Kxf, params.Kxx \ dKxf[:,i])
        dσ /= sqrt(params.σ²)*2.0

        dlog_p = -(y - params.μ)/sqrt(params.σ²)*(-dμ/sqrt(params.σ²) - (y - params.μ)/params.σ²*dσ)
        dlog_p -= dσ/sqrt(params.σ²)
        dmll[i] = dlog_p
    end
    return dmll
end

function grad_stack(k::GaussianProcesses.Kernel, X1::AbstractMatrix, X2::AbstractMatrix)
    data = GaussianProcesses.KernelData(k, X1, X2)
    nobs1 = size(X1, 2)
    nobs2 = size(X2, 2)
    stack = Array{eltype(X1)}(undef, nobs1, nobs2, GaussianProcesses.num_params(k))
    GaussianProcesses.grad_stack!(stack, k, X1, X2, data)
end

function reset_gp!(gp::GPLA)
    gp.x = zeros(Float64, gp.dim, 0)
    gp.y = zeros(Float64, 0)
    gp.Kfull = sizehint!(Dict{Tuple{Int64, Int64}, Float64}(), 3000)
end

function extract_value(x::Array)
    x_out = zeros(Float64, size(x, 1), size(x, 2))
    for j = 1:size(x,2)
        for i = 1:size(x,1)
            if x[i] isa Real
                x_out[i, j] = x[i, j]
            else
                x_out[i, j] = x[i, j].value
            end
        end
    end
    return x_out #TODO: Remove?
end

GaussianProcesses.get_params_kwargs(::GPLA; kwargs...) = delete!(Dict(kwargs), :lik)

function GaussianProcesses.get_params(gp::GPLA; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    params = Float64[]
    if noise; append!(params, GaussianProcesses.get_params(gp.logNoise)); end
    if domean
        append!(params, GaussianProcesses.get_params(gp.mean))
    end
    if kern
        append!(params, GaussianProcesses.get_params(gp.kernel))
    end
    return params
end

function GaussianProcesses.optimize!(gp::GPLA; method = GaussianProcesses.LBFGS(), domean::Bool = true, kern::Bool = true,
                   noise::Bool = true, lik::Bool = true,
                   meanbounds = nothing, kernbounds = nothing,
                   noisebounds = nothing, likbounds = nothing, kwargs...)
    params_kwargs = GaussianProcesses.get_params_kwargs(gp; domean=domean, kern=kern, noise=noise, lik=lik)
    func = GaussianProcesses.get_optim_target(gp; params_kwargs...)
    init = GaussianProcesses.get_params(gp; params_kwargs...)  # Initial hyperparameter values
    if meanbounds == kernbounds == noisebounds == likbounds == nothing
        results = GaussianProcesses.optimize(func, init; method=method, autodiff = :forward, kwargs...)     # Run optimizer
    else
        lb, ub = GaussianProcesses.bounds(gp, noisebounds, meanbounds, kernbounds, likbounds;
                        domean = domean, kern = kern, noise = noise, lik = lik)
        results = GaussianProcesses.optimize(func.f, func.df, lb, ub, init, Fminbox(method))
    end
    GaussianProcesses.set_params!(gp, Optim.minimizer(results); params_kwargs...)
    update_K!(gp)
    update_mll!(gp)
    return results
end

function GaussianProcesses.set_params!(gp::GPLA, hyp::AbstractVector;
                     noise::Bool=true, domean::Bool=true, kern::Bool=true)
    n_noise_params = GaussianProcesses.num_params(gp.logNoise)
    n_mean_params = GaussianProcesses.num_params(gp.mean)
    n_kern_params = GaussianProcesses.num_params(gp.kernel)

    i = 1
    if noise
        GaussianProcesses.set_params!(gp.logNoise, hyp[1:n_noise_params])
        i += n_noise_params
    end

    if domean && n_mean_params>0
        GaussianProcesses.set_params!(gp.mean, hyp[i:i+n_mean_params-1])
        i += n_mean_params
    end

    if kern
        GaussianProcesses.set_params!(gp.kernel, hyp[i:i+n_kern_params-1])
        i += n_kern_params
    end
end

function GaussianProcesses.get_optim_target(gp::GPLA; params_kwargs...)
    function ltarget(hyp::AbstractVector)
        prev = GaussianProcesses.get_params(gp; params_kwargs...)
        try
            GaussianProcesses.set_params!(gp, hyp; params_kwargs...)
            update_K!(gp)
            update_mll!(gp)
        catch err
            # reset parameters to remove any NaNs
            GaussianProcesses.set_params!(gp, prev; params_kwargs...)

            if !all(isfinite.(hyp))
                println(err)
                return Inf
            elseif isa(err, ArgumentError)
                println(err)
                return Inf
            elseif isa(err, LinearAlgebra.PosDefException)
                println(err)
                return Inf
            else
                throw(err)
            end
        end
    end
    function ltarget_and_dltarget!(grad::AbstractVector, hyp::AbstractVector)
        prev = GaussianProcesses.get_params(gp; params_kwargs...)
        try
            GaussianProcesses.set_params!(gp, hyp; params_kwargs...)
            update_K!(gp)
            update_mll!(gp)
            update_dmll!(gp; params_kwargs...)
            grad[:] = -gp.dmll
            return -gp.mll
        catch err
            # reset parameters to remove any NaNs
            GaussianProcesses.set_params!(gp, prev; params_kwargs...)
            if !all(isfinite.(hyp))
                println(err)
                return Inf
            elseif isa(err, ArgumentError)
                println(err)
                return Inf
            elseif isa(err, LinearAlgebra.PosDefException)
                println(err)
                return Inf
            else
                throw(err)
            end
        end
    end

    function dltarget!(grad::AbstractVector, hyp::AbstractVector)
        ltarget_and_dltarget!(grad::AbstractVector, hyp::AbstractVector)
    end
    xinit = GaussianProcesses.get_params(gp; params_kwargs...)
    func = GaussianProcesses.OnceDifferentiable(ltarget, dltarget!, ltarget_and_dltarget!, xinit)
    return func
end
