using ProgressBars
using GaussianProcesses
using LinearAlgebra: dot, logdet
using PDMats: PDMat
using Random
using NearestNeighbors

mean = GaussianProcesses.mean
Mean = GaussianProcesses.Mean
CovarianceStrategy = GaussianProcesses.CovarianceStrategy
KernelData = GaussianProcesses.KernelData
AbstractPDMat = GaussianProcesses.AbstractPDMat
log2π = GaussianProcesses.log2π
make_posdef! = GaussianProcesses.make_posdef!
update_cK! = GaussianProcesses.update_cK!
alloc_cK = GaussianProcesses.alloc_cK
get_value = GaussianProcesses.get_value

mutable struct CustomWindSparseKernel{T<:Real} <: Kernel
    # SquaredExponentialKernel
    l_sq::T       # lengthscale
    σs_sq::T      # signal variance

    # LinearExponentialKernel
    l_lin::T      # lengthscale
    σs_lin::T     # signal variance

    # WindLogLawKernel
    d::T          # zero-plane displacement
    zₒ::T         # roughness length

    NN::Dict

    priors::Array
end

CustomWindSparseKernel(l_sq::T, σs_sq::T, l_lin::T, σs_lin::T, d::T, zₒ::T, NN::Dict) where T = CustomWindSparseKernel{T}(l_sq, σs_sq, l_lin, σs_lin, d, zₒ, NN, [])

function GaussianProcesses.set_params!(se::CustomWindSparseKernel, hyp::AbstractVector)
    length(hyp) == 6 || throw(ArgumentError("Needed 6 parameters, received $(length(hyp))."))
    se.l_sq, se.σs_sq, se.l_lin, se.σs_lin, se.d, se.zₒ = hyp 
end

GaussianProcesses.get_params(se::CustomWindSparseKernel{T}) where T = T[se.l_sq, se.σs_sq, se.l_lin, se.σs_lin, se.d, se.zₒ]
GaussianProcesses.get_param_names(se::CustomWindSparseKernel) = [:l_sq, :σs_sq, :l_lin, :σs_lin, :d, :zₒ]
GaussianProcesses.num_params(se::CustomWindSparseKernel) = 6


"""
GaussianProcesses.cov(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix)
Create covariance matrix from kernel `k` and matrices of observations `X1` and `X2`, where
each column is an observation.
"""
function GaussianProcesses.cov(k::CustomWindSparseKernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData=GaussianProcesses.EmptyData())
    dim1, nobs1 = size(X1)
    dim2, nobs2 = size(X2)
    dim1==dim2 || throw(ArgumentError("X1 and X2 must have same dimension"))
    cK = Array{promote_type(eltype(X1), eltype(X2))}(undef, nobs1, nobs2)
    GaussianProcesses.cov!(cK, k, X1, X2, data)
end

function GaussianProcesses.cov!(cK::AbstractMatrix, k::CustomWindSparseKernel, X::AbstractMatrix, data::KernelData=GaussianProcesses.EmptyData())
    dim, nobs = size(X)
    (nobs,nobs) == size(cK) || throw(ArgumentError("cK has size $(size(cK)) and X has size $(size(X))"))
    @inbounds for j in 1:nobs
        for i in 1:nobs
            cK[i,j] = GaussianProcesses.cov_ij(k, X, X, data, i, j, dim)
        end
    end
    return cK
end

function GaussianProcesses.cov!(cK::AbstractMatrix, k::CustomWindSparseKernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData=GaussianProcesses.EmptyData())
    if X1 === X2
        return GaussianProcesses.cov!(cK, k, X1, data)
    end
    dim1, nobs1 = size(X1)
    dim2, nobs2 = size(X2)
    dim1==dim2 || throw(ArgumentError("X1 and X2 must have same dimension"))
    dim = size(X1, 1)
    (nobs1,nobs2) == size(cK) || throw(ArgumentError("cK has size $(size(cK)) X1 $(size(X1)) and X2 $(size(X2))"))
    @inbounds for i in 1:nobs1
        for j in 1:nobs2
            cK[i,j] = GaussianProcesses.cov_ij(k, X1, X2, data, i, j, dim)
        end
    end
    return cK
end

function GaussianProcesses.cov_ij(k::CustomWindSparseKernel, X1::AbstractMatrix, X2::AbstractMatrix, i::Int, j::Int, dim::Int)

    l_sq, σs_sq, l_lin, σs_lin, d, zₒ = k.l_sq, k.σs_sq, k.l_lin, k.σs_lin, k.d, k.zₒ

    x_star = @view(X1[:,i])
    x = @view(X2[:,j])
    
    z_star = x_star[3]
    z = x[3]

    k_val = 1        
    
    # SquaredExponentialKernel
    r = x[1:2] - x_star[1:2]
    r_sq = dot_product(r,r)
    k_val *= σs_sq^2 * exp(-r_sq/(2*l_sq))

    # LinearExponentialKernel
    r = abs(z - z_star)
    k_val *= σs_lin^2 * exp(-r/(2*l_lin))

    # WindLogLawKernel
    ratio = log((z_star-d)/zₒ) / log((z-d)/zₒ)
    # k_val *= (ratio - m[z_star]/uₓ) / (1 - m[z]/uₓ)
    k_val *= ratio

    return k_val
end

function GaussianProcesses.cov_ij(k::CustomWindSparseKernel, x1::AbstractArray, x2::AbstractArray)

    l_sq, σs_sq, l_lin, σs_lin, d, zₒ = k.l_sq, k.σs_sq, k.l_lin, k.σs_lin, k.d, k.zₒ

    x_star = x1
    x = x2
    
    z_star = x_star[3]
    z = x[3]

    k_val = 1        
    
    # SquaredExponentialKernel
    r = x[1:2] - x_star[1:2]
    r_sq = dot_product(r,r)
    k_val *= σs_sq^2 * exp(-r_sq/(2*l_sq))

    # LinearExponentialKernel
    r = abs(z - z_star)
    k_val *= σs_lin^2 * exp(-r/(2*l_lin))

    # WindLogLawKernel
    ratio = log((z_star-d)/zₒ) / log((z-d)/zₒ)
    # k_val *= (ratio - m[z_star]/uₓ) / (1 - m[z]/uₓ)
    k_val *= ratio

    return k_val
end

@inline GaussianProcesses.cov_ij(k::CustomWindSparseKernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i::Int, j::Int, dim::Int) = GaussianProcesses.cov_ij(k, X1, X2, i, j, dim)

# function GaussianProcesses.predictMVN!(gp, Kxx, Kff, Kfx, Kxf, mx, αf)
#     # mu = mx + Kfx' * αf       # this is old.
#     mu = mx + Kxf * αf          # this is new. Kxf need not be equal to Kfx'.
#     Lck = GaussianProcesses.whiten!(Kff, Kfx)
#     GaussianProcesses.subtract_Lck!(Kxx, Lck)

#     # println("I AM ENTERED.")  # for debug
#     return mu, abs.(Kxx)
# end

function GaussianProcesses.predict_y(gp::GPE, xpred::AbstractVector, numNeighbors::Int, grid_dist::Int, altitudes::AbstractArray)  # predict a single point, with added noise
    μ, σ2 = predict_f(gp, xpred, numNeighbors, grid_dist, altitudes)
    return μ, σ2 .+ noise_variance(gp)
end

function GaussianProcesses.predict_f(gp::GPE, xpred::AbstractVector, numNeighbors::Int, grid_dist::Int, altitudes::AbstractArray)  # predict a single point
    
    X_gp = gp.x
    kernel = gp.kernel

    kdtree_X_gp = KDTree(X_gp)
    neighbors_of_xpred, _ = knn(kdtree_X_gp, xpred, numNeighbors)
    sort!(neighbors_of_xpred)

    # neighbors_of_xpred = getNearestNeighborsFaster(xpred, X_gp, numNeighbors, grid_dist, altitudes)
    num_of_neigh = length(neighbors_of_xpred)

    X_active = X_gp[:, neighbors_of_xpred]
    empty_cK = alloc_cK(num_of_neigh) 

    K_xx = active_cK = update_cK!(empty_cK, X_active, gp.kernel, get_value(gp.logNoise), gp.data, gp.covstrat)

    K_fx = getSparse_K(kernel, xpred, X_active)
    K_xf = getSparse_K(kernel, X_active, xpred)
    K_f = GaussianProcesses.cov_ij(kernel, xpred, xpred)

    mf = mean(gp.mean, xpred)
    mx = mean(gp.mean, X_active)
    yx = gp.y[neighbors_of_xpred]

    μ_star = mf + dot(K_fx, inv(K_xx.mat) * (yx - mx))    
    Σ_star = ones(1,1)*K_f  # convert from Float64 to Array
    Lck = GaussianProcesses.whiten!(active_cK, K_xf)
    GaussianProcesses.subtract_Lck!(Σ_star, Lck)

    Σ_star = abs.(Σ_star[1])
    # σ_star = sqrt(Σ_star)

    return μ_star, Σ_star
end

function Random.rand(gp::GPBase, kernel::CustomWindSparseKernel, Xs_gp::AbstractArray, numNeighbors::Int, grid_dist_Xs::Int)
    """ Randomly samples an entire farm, based on Sequential Gaussian Simulation """
    X_gp = gp.x
    
    if size(Xs_gp) == (3,)                  # if there is only one point, make AbstractArray 2 dimensional.
        Xs_gp = transform4GPjl([Xs_gp])
    end
    
    X_gp_set = Set(eachcol(X_gp))           # lookup in Set is O(1), we will take advantage of this.
    Xs_samples_val = Float64[]

    kdtree_X_gp = KDTree(X_gp)
    kdtree_Xs_gp = KDTree(Xs_gp)

    prequal_samples = Set{Int}()            # indices of previously sampled xs points. these should be unique w.r.t. points in X_gp.
    prequal_samples_val = Float64[]

    for (xs_idx, xs) in tqdm(enumerate(eachcol(Xs_gp)))

        neighbors_of_xs_in_Xs, dist2Xs = knn(kdtree_Xs_gp, xs, numNeighbors)
        neighbors_of_xs_in_Xs = collect(intersect(prequal_samples, Set{Int}(neighbors_of_xs_in_Xs)))   # only take points in tree if they have been sampled earlier.

        neighbors_of_xs_in_Xs_values = prequal_samples_val[neighbors_of_xs_in_Xs]

        neighbors_of_xs_in_X, dist2X = knn(kdtree_X_gp, xs, numNeighbors)
        neighbors_of_xs_in_X_values = gp.y[neighbors_of_xs_in_X]

        closest_neighbors_of_xs = hcat(Xs_gp[:,neighbors_of_xs_in_Xs], X_gp[:,neighbors_of_xs_in_X])
        closest_neighbors_of_xs_values = vcat(neighbors_of_xs_in_Xs_values, neighbors_of_xs_in_X_values)

        num_of_neigh = size(closest_neighbors_of_xs, 2)
        empty_cK = alloc_cK(num_of_neigh)

        sort_neighs = sortperm(closest_neighbors_of_xs[end, :])   # sort by altitude to prevent non-PSD.  
        X_active = closest_neighbors_of_xs[:, sort_neighs]
        

        K_xx = active_cK = update_cK!(empty_cK, X_active, kernel, get_value(gp.logNoise), gp.data, gp.covstrat)
        
        K_fx = getSparse_K(kernel, xs, X_active)
        K_xf = getSparse_K(kernel, X_active, xs)
        K_f = GaussianProcesses.cov_ij(kernel, xs, xs)
        
        mf = mean(gp.mean, xs)
        mx = mean(gp.mean, closest_neighbors_of_xs)


        yx = closest_neighbors_of_xs_values[sort_neighs]     
        
        μ_star = mf + dot(K_fx, inv(K_xx.mat) * (yx - mx))        
        Σ_star = ones(1,1)*K_f                          # convert from Float64 to Array
        Lck = GaussianProcesses.whiten!(active_cK, K_xf)
        GaussianProcesses.subtract_Lck!(Σ_star, Lck)
        
        Σ_star = abs.(Σ_star[1]) + noise_variance(gp)   # mimics predict_y.

        xs_dist = Normal(μ_star, Σ_star)
        xs_sampled_val = rand(xs_dist)
        push!(Xs_samples_val, xs_sampled_val)

        if !(xs in X_gp_set)                            # enforces uniqueness w.r.t. points in X_gp.
            push!(prequal_samples, xs_idx)
            push!(prequal_samples_val, xs_sampled_val)
        else
            push!(prequal_samples_val, NaN)
        end

    end
    return Xs_samples_val
end




"""kernels/GPE.jl, Line 195"""
function GaussianProcesses.update_mll!(gp::GPE; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    # if kern | noise
    #     GaussianProcesses.update_cK!(gp)
    # end
    μ = mean(gp.mean, gp.x)
    y = gp.y - μ
    
    if typeof(gp.kernel) <: CustomWindSparseKernel
        gp.alpha = [0.0]       # this is new.
    else
        gp.alpha = gp.cK \ y                # this is old, which is inherited from PDMats.jl, but the approximation error is too large.
    end

    # gp.alpha = gp.cK \ y 
    # Marginal log-likelihood
    # gp.mll = - (dot(y, gp.alpha) + logdet(gp.cK) + log2π * gp.nobs) / 2
    gp.mll = getSparse_mll(gp)
    gp
end

function getSparse_K(kernel, X1, X2)
    """get a smaller cK matrix for a single x point"""
    
    X1_length = size(X1,2)
    X2_length = size(X2,2)

    cK = Array{Float64}(undef, X1_length, X2_length)

    for i in 1:X1_length
        x1 = X1[:,i]

        for j in 1:X2_length
            x2 = X2[:,j]
            cK[i,j] = GaussianProcesses.cov_ij(kernel, x1, x2)
        end
    end

    return cK
end


function getSparse_mll(gp; return_sum = true)

    X_gp = gp.x
    kernel = gp.kernel
    NN = kernel.NN
    # cK = gp.cK

    mll = Array{Float64}(undef, size(X_gp,2), 1)

    for idx in tqdm(1:size(X_gp,2))
        x = X_gp[:,idx]
        neighbors_of_x = sort(NN[x])
        num_of_neigh = length(neighbors_of_x)
        
        X_active = X_gp[:, neighbors_of_x]
        empty_cK = alloc_cK(num_of_neigh) 

        K_xx = active_cK = update_cK!(empty_cK, X_active, gp.kernel, get_value(gp.logNoise), gp.data, gp.covstrat)
        # K_xx = cK.mat[neighbors_of_x, neighbors_of_x]
    
        K_fx = getSparse_K(kernel, x, X_active)
        K_xf = getSparse_K(kernel, X_active, x)
        K_f = GaussianProcesses.cov_ij(kernel, x, x)
    
        mean_f = mean(gp.mean, gp.x)
        
        mf = mean_f[idx]
        yf = gp.y[idx]
        
        mx = mean_f[neighbors_of_x]
        yx = gp.y[neighbors_of_x]
    
        μ_star = mf + dot(K_fx, inv(K_xx.mat) * (yx - mx))
        # Σ_star = K_f - dot(K_fx, inv(K_xx) * K_xf)
        
        # Σbuffer, chol = make_posdef!(K_xx, cK.chol.factors[neighbors_of_x, neighbors_of_x])
        # new_cK = PDMat(Σbuffer, chol)

        Σ_star = ones(1,1)*K_f  # convert from Float64 to Array
        Lck = GaussianProcesses.whiten!(active_cK, K_xf)
        GaussianProcesses.subtract_Lck!(Σ_star, Lck)

        Σ_star = abs.(Σ_star[1])
        σ = sqrt(Σ_star)
        
        mll[idx] = -0.5*((yf - μ_star)/σ)^2 - 0.5*log2π - log(σ)

    end

    return_sum ? sum(mll) : mll
    # return sum(mll)
end


function getNearestNeighbors(Xs_gp::AbstractMatrix, X_gp, n; return_coordinates=false)
    """ Finds the closest neighbors of points in Xs_gp with respect to points in X_gp. """

    # Key: A single coordinate.
    # Val: An array of n-nearest neighbors.
    NN = Dict{AbstractArray, AbstractArray}()

    println("## Getting all neighbors ##")
    for xs in tqdm(unique(eachcol(Xs_gp)))
        
        temp = [euclidean_dist(xs, X_gp[:,jdx]) for jdx in 1:size(X_gp,2)]
        p = sortperm(temp)

        if return_coordinates
            best_n = X_gp[:,p][:,(1:n)]
            NN[xs] = best_n
        else
            NN[xs] = p[1:n]
        end

    end

    return NN
end

function getNearestNeighbors(xs::AbstractVector, X_gp, n; exclude_itself = false, return_coordinates=false)
    """ Finds the closest neighbors of point `xs` with respect to points in X_gp. """

    println("## Getting all neighbors ##")
        
    temp = [euclidean_dist(xs, X_gp[:,jdx]) for jdx in 1:size(X_gp,2)]
    p = sortperm(temp)

    if return_coordinates
        exclude_itself ? best_n = X_gp[:,p][:,(2:n+1)] : best_n = X_gp[:,p][:,(1:n)]
        return best_n
    else
        exclude_itself ? best_n = p[2:n+1] : best_n = p[1:n]
        return best_n
    end

end

function getNearestNeighborsFaster(Xs_gp::AbstractMatrix, X_gp, n, grid_dist, altitudes; prequal_samples=Set(), return_as_list=false, return_coordinates=false)
    """ Finds the closest neighbors of points in Xs_gp with respect to points in X_gp. """
    """ Same as `getNearestNeighbors`, but does not search entire space. """
    """ When `prequal_samples` is entered, the points in that set are considered too, in addition to X_gp """

    X_gp_set = Set(eachcol(X_gp))
    
    if !return_coordinates
        X_gp_idxs = Dict(item => idx for (idx,item) in enumerate(eachcol(X_gp)))
    end
    
    # Key: A single coordinate.
    # Val: An array of n-nearest neighbors.
    NN = Dict{AbstractArray, AbstractArray}()
    
    println("## Getting all neighbors ##")
    for xs0 in tqdm(eachcol(Xs_gp))
        
        xs = copy(xs0)    # copy is sufficient since elements of Float64 are immutable.
        xs[1:2] = nearestRound(xs0[1:2], grid_dist)
        samples = Set()
        jdx = 1

        let jdx=jdx
        while length(samples) < n
            gs = grid_dist * jdx

            for h in altitudes
                push!(samples, vcat(xs[1:2], h))

                push!(samples, vcat(xs[1]-gs, xs[2], h))
                push!(samples, vcat(xs[1]+gs, xs[2], h))
                
                push!(samples, vcat(xs[1], xs[2]-gs, h))
                push!(samples, vcat(xs[1], xs[2]+gs, h))

                push!(samples, vcat(xs[1]-gs, xs[2]-gs, h))
                push!(samples, vcat(xs[1]+gs, xs[2]+gs, h))

                push!(samples, vcat(xs[1]-gs, xs[2]+gs, h))
                push!(samples, vcat(xs[1]+gs, xs[2]-gs, h))
            end
            jdx += 1
            intersect!(samples, X_gp_set)
        end
        end

        union!(samples, prequal_samples)

        temp = [euclidean_dist(xs0,item) for item in samples]
        p = sortperm(temp)
        best_n = collect(Array{Float64,1}, samples)[p][1:n]

        if return_coordinates
            return_as_list ? NN[xs0] = best_n : NN[xs0] = transform4GPjl(best_n)
        else
            NN[xs0] = [X_gp_idxs[item] for item in best_n]
        end

    end

    return NN
end


function getNearestNeighborsFaster(xs0::AbstractVector, X_gp, n, grid_dist, altitudes; prequal_samples=Set(), return_as_list=false, return_coordinates=false)
    """ Finds the closest neighbors of point `xs` with respect to points in X_gp. """
    """ Same as `getNearestNeighbors`, but does not search entire space. """

    X_gp_set = Set(eachcol(X_gp))
    
    if !return_coordinates
        X_gp_idxs = Dict(item => idx for (idx,item) in enumerate(eachcol(X_gp)))
    end
    
    xs = copy(xs0)    # copy is sufficient since elements of Float64 are immutable.
    xs[1:2] = nearestRound(xs[1:2], grid_dist)
    samples = Set()
    jdx = 1

    let jdx=jdx
    while length(samples) < n
        gs = grid_dist * jdx

        for h in altitudes
            push!(samples, vcat(xs[1:2], h))

            push!(samples, vcat(xs[1]-gs, xs[2], h))
            push!(samples, vcat(xs[1]+gs, xs[2], h))
            
            push!(samples, vcat(xs[1], xs[2]-gs, h))
            push!(samples, vcat(xs[1], xs[2]+gs, h))

            push!(samples, vcat(xs[1]-gs, xs[2]-gs, h))
            push!(samples, vcat(xs[1]+gs, xs[2]+gs, h))

            push!(samples, vcat(xs[1]-gs, xs[2]+gs, h))
            push!(samples, vcat(xs[1]+gs, xs[2]-gs, h))
        end
        jdx += 1
        intersect!(samples, X_gp_set)
    end
    end

    union!(samples, prequal_samples)
    
    temp = [euclidean_dist(xs0,item) for item in samples]
    p = sortperm(temp)
    best_n = collect(Array{Float64,1}, samples)[p][1:n]

    if return_coordinates
        result = return_as_list ? best_n : transform4GPjl(best_n)
        return result
    else
        return sort([X_gp_idxs[item] for item in best_n])
    end

end

function getNearestNeighborsFaster(X::AbstractArray, Y, n, grid_dist; prequal_samples=Set(), return_as_list=false, return_coordinates=false)
    """ Use all altitudes inside `Y` if not specified as an input to `getNearestNeighborsFaster`. """
    altitudes = unique(Y[end,:])
    return getNearestNeighborsFaster(X, Y, n, grid_dist, altitudes; prequal_samples=prequal_samples, return_as_list=return_as_list, return_coordinates=return_coordinates)
end

