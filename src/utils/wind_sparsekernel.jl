using ProgressBars
using GaussianProcesses
using LinearAlgebra: dot, logdet
using PDMats: PDMat

mean = GaussianProcesses.mean
Mean = GaussianProcesses.Mean
CovarianceStrategy = GaussianProcesses.CovarianceStrategy
KernelData = GaussianProcesses.KernelData
AbstractPDMat = GaussianProcesses.AbstractPDMat
log2π = GaussianProcesses.log2π
make_posdef! = GaussianProcesses.make_posdef!

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
    @inbounds for j in tqdm(1:nobs)
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

function GaussianProcesses.predictMVN!(gp, Kxx, Kff, Kfx, Kxf, mx, αf)
    # mu = mx + Kfx' * αf       # this is old.
    mu = mx + Kxf * αf          # this is new. Kxf need not be equal to Kfx'.
    Lck = GaussianProcesses.whiten!(Kff, Kfx)
    GaussianProcesses.subtract_Lck!(Kxx, Lck)

    # println("I AM ENTERED.")  # for debug
    return mu, abs.(Kxx)
end

function GaussianProcesses.predictMVN(gp, xpred::AbstractMatrix, xtrain::AbstractMatrix, ytrain::AbstractVector, kernel::Kernel, meanf::Mean, alpha::AbstractVector, covstrat::CovarianceStrategy, Ktrain::AbstractPDMat)
    crossdata = KernelData(kernel, xtrain, xpred)
    priordata = KernelData(kernel, xpred, xpred)
    Kcross = GaussianProcesses.cov(kernel, xtrain, xpred, crossdata)
    Kcross2 = GaussianProcesses.cov(kernel, xpred, xtrain, crossdata)     # this is newly added.
    Kpred = GaussianProcesses.cov(kernel, xpred, xpred, priordata)
    mx = mean(meanf, xpred)

    mu, Sigma_raw = GaussianProcesses.predictMVN!(gp, Kpred, Ktrain, Kcross, Kcross2, mx, alpha)
    return mu, Sigma_raw
end

"""kernels/GPE.jl, Line 380"""
function GaussianProcesses.predict_full(gp::GPE, xpred::AbstractMatrix)
    if typeof(gp.kernel) <: CustomWindSparseKernel
        GaussianProcesses.predictMVN(gp, xpred, gp.x, gp.y, gp.kernel, gp.mean, gp.alpha, gp.covstrat, gp.cK)
    else   # Default behavior of GaussianProcesses.jl.
        GaussianProcesses.predictMVN(xpred, gp.x, gp.y, gp.kernel, gp.mean, gp.alpha, gp.covstrat, gp.cK)
    end
end

"""kernels/GPE.jl, Line 195"""
function GaussianProcesses.update_mll!(gp::GPE; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    if kern | noise
        GaussianProcesses.update_cK!(gp)
    end
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



function getNearestNeighbors(Xs_gp, X_gp, n; return_coordinate=false)
    """ Finds the closest neighbors of points in Xs_gp with respect to points in X_gp. """

    # Key: A single coordinate.
    # Val: An array of n-nearest neighbors.
    NN = Dict{AbstractArray, AbstractArray}()

    println("## Getting all neighbors ##")
    for idx in tqdm(1:size(Xs_gp,2))
        # @show idx;
        xs = Xs_gp[:,idx]

        temp = [euclidean_dist(xs,X_gp[:,jdx]) for jdx in 1:size(X_gp,2)]
        p = sortperm(temp)

        if return_coordinate
            best_n = X_gp[:,p][:,(1:n+1)]  # deliberately include the point itself.
            NN[xs] = best_n
        else
            NN[xs] = p[1:n+1]              # deliberately include the point itself.
        end

    end

    return NN
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


function getSparse_cK(gp, X_gp, neighbors_of_x, numNeighbors)
    """get a smaller cK matrix for a single x point"""
    
    cK = Array{Float64}(undef, numNeighbors+1, numNeighbors+1)

    for (r,i) in enumerate(neighbors_of_x)
        x1 = X_gp[:,i]
        for (c,j) in enumerate(neighbors_of_x)
            x2 = X_gp[:,j]
            cK[r,c] = GaussianProcesses.cov_ij(gp.kernel, x1, x2)
        end
    end

    return cK
end

function getSparse_mll(gp)

    X_gp = gp.x
    kernel = gp.kernel
    NN = kernel.NN
    cK = gp.cK

    mll = Array{Float64}(undef, size(X_gp,2), 1)

    for idx in tqdm(1:size(X_gp,2))
        x = X_gp[:,idx]
        neighbors_of_x = sort(NN[x])
        # neighbors_of_x = collect(1:499)  # TODO: remove this. (debug)
        
        X_active = X_gp[:, neighbors_of_x]
        
        # K_xx = getSparse_K(kernel, X_active, X_active)
        K_xx = cK.mat[neighbors_of_x, neighbors_of_x]
    
        K_fx = getSparse_K(kernel, x, X_active)
        K_xf = getSparse_K(kernel, X_active, x)
        K_f = GaussianProcesses.cov_ij(kernel, x, x)
    
        mean_f = GaussianProcesses.mean(gp.mean, gp.x)
        
        mf = mean_f[idx]
        yf = gp.y[idx]
        
        mx = mean_f[neighbors_of_x]
        yx = gp.y[neighbors_of_x]
    
        μ_star = mf + dot(K_fx, inv(K_xx) * (yx - mx))
        # Σ_star = K_f - dot(K_fx, inv(K_xx) * K_xf)
        
        Σbuffer, chol = make_posdef!(K_xx, cK.chol.factors[neighbors_of_x, neighbors_of_x])
        new_cK = PDMat(Σbuffer, chol)

        Σ_star = ones(1,1)*K_f  # convert from Float64 to Array
        Lck = GaussianProcesses.whiten!(new_cK, K_xf)
        GaussianProcesses.subtract_Lck!(Σ_star, Lck)


        noise = 0
        # noise = exp(2*-2)+eps()
        Σ_star = abs.(Σ_star[1] + noise)
        
        σ = sqrt(Σ_star)
        
        mll[idx] = -0.5*((yf - μ_star)/σ)^2 - 0.5*log(2*pi) - log(σ)

    end

    return sum(mll)
end