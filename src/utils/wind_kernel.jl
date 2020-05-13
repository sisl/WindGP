using GaussianProcesses
using LinearAlgebra: dot, logdet

mean = GaussianProcesses.mean
Mean = GaussianProcesses.Mean
CovarianceStrategy = GaussianProcesses.CovarianceStrategy
KernelData = GaussianProcesses.KernelData
AbstractPDMat = GaussianProcesses.AbstractPDMat
log2π = GaussianProcesses.log2π

mutable struct CustomWindKernel{T<:Real} <: Kernel
    # SquaredExponentialKernel
    l_sq::T       # lengthscale
    σs_sq::T      # signal variance

    # LinearExponentialKernel
    l_lin::T      # lengthscale
    σs_lin::T     # signal variance

    # WindLogLawKernel
    d::T          # zero-plane displacement
    zₒ::T         # roughness length

    priors::Array
end

CustomWindKernel(l_sq::T, σs_sq::T, l_lin::T, σs_lin::T, d::T, zₒ::T) where T = CustomWindKernel{T}(l_sq, σs_sq, l_lin, σs_lin, d, zₒ, [])

function GaussianProcesses.set_params!(se::CustomWindKernel, hyp::AbstractVector)
    length(hyp) == 6 || throw(ArgumentError("Needed 6 parameters, received $(length(hyp))."))
    se.l_sq, se.σs_sq, se.l_lin, se.σs_lin, se.d, se.zₒ = hyp 
end

GaussianProcesses.get_params(se::CustomWindKernel{T}) where T = T[l_sq, σs_sq, l_lin, σs_lin, d, zₒ]
GaussianProcesses.get_param_names(se::CustomWindKernel) = [:l_sq, :σs_sq, :l_lin, :σs_lin, :d, :zₒ]
GaussianProcesses.num_params(se::CustomWindKernel) = 6


"""
GaussianProcesses.cov(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix)
Create covariance matrix from kernel `k` and matrices of observations `X1` and `X2`, where
each column is an observation.
"""
function GaussianProcesses.cov(k::CustomWindKernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData=GaussianProcesses.EmptyData())
    dim1, nobs1 = size(X1)
    dim2, nobs2 = size(X2)
    dim1==dim2 || throw(ArgumentError("X1 and X2 must have same dimension"))
    cK = Array{promote_type(eltype(X1), eltype(X2))}(undef, nobs1, nobs2)
    GaussianProcesses.cov!(cK, k, X1, X2, data)
end

function GaussianProcesses.cov!(cK::AbstractMatrix, k::CustomWindKernel, X::AbstractMatrix, data::KernelData=GaussianProcesses.EmptyData())
    dim, nobs = size(X)
    (nobs,nobs) == size(cK) || throw(ArgumentError("cK has size $(size(cK)) and X has size $(size(X))"))
    @inbounds for j in 1:nobs
        for i in 1:nobs
            cK[i,j] = GaussianProcesses.cov_ij(k, X, X, data, i, j, dim)
        end
    end
    return cK
end

function GaussianProcesses.cov!(cK::AbstractMatrix, k::CustomWindKernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData=GaussianProcesses.EmptyData())
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

function GaussianProcesses.cov_ij(k::CustomWindKernel, X1::AbstractMatrix, X2::AbstractMatrix, i::Int, j::Int, dim::Int)

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

@inline GaussianProcesses.cov_ij(k::CustomWindKernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i::Int, j::Int, dim::Int) = GaussianProcesses.cov_ij(k, X1, X2, i, j, dim)

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
    if typeof(gp.kernel) <: CustomWindKernel
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
    
    if typeof(gp.kernel) <: CustomWindKernel
        gp.alpha = inv(gp.cK.mat) * y       # this is new.
    else
        gp.alpha = gp.cK \ y                # this is old, which is inherited from PDMats.jl, but the approximation error is too large.
    end

    # gp.alpha = gp.cK \ y 
    # Marginal log-likelihood
    gp.mll = - (dot(y, gp.alpha) + logdet(gp.cK) + log2π * gp.nobs) / 2
    gp
end