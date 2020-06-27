"""
Wind Log Kernel
Custom kernel for the wind farm domain.
This version uses SEIso as the x-y kernel.
"""

mutable struct WLK_SEIso{T<:Real} <: GaussianProcesses.Isotropic{GaussianProcesses.SqEuclidean}
    # SquaredExponentialKernel
    ℓ2_sq::T       # lengthscale
    σ2_sq::T      # signal variance

    # LinearExponentialKernel
    ℓ_lin::T      # lengthscale
    σ2_lin::T     # signal variance

    # WindLogLawKernel
    d::T          # zero-plane displacement
    zₒ::T         # roughness length

    # Priors for kernel parameters
    priors::Array
end

cov(se::WLK_SEIso, r::Number) = error("This call should not have been made.")




"""
GaussianProcesses.cov(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix)
Create covariance matrix from kernel `k` and matrices of observations `X1` and `X2`, where
each column is an observation.
"""
function GaussianProcesses.cov(k::WLK_SEIso, X1::AbstractMatrix, X2::AbstractMatrix, data::GaussianProcesses.KernelData=GaussianProcesses.EmptyData())
    dim1, nobs1 = size(X1)
    dim2, nobs2 = size(X2)
    dim1==dim2 || throw(ArgumentError("X1 and X2 must have same dimension"))
    cK = Array{promote_type(eltype(X1), eltype(X2))}(undef, nobs1, nobs2)
    GaussianProcesses.cov!(cK, k, X1, X2, data)
end

function GaussianProcesses.cov!(cK::AbstractMatrix, k::WLK_SEIso, X::AbstractMatrix, data::GaussianProcesses.KernelData=GaussianProcesses.EmptyData())
    dim, nobs = size(X)
    (nobs,nobs) == size(cK) || throw(ArgumentError("cK has size $(size(cK)) and X has size $(size(X))"))
    @inbounds for j in 1:nobs
        for i in 1:nobs
            cK[i,j] = GaussianProcesses.cov_ij(k, X, X, data, i, j, dim)
        end
    end
    return cK
end

function GaussianProcesses.cov!(cK::AbstractMatrix, k::WLK_SEIso, X1::AbstractMatrix, X2::AbstractMatrix, data::GaussianProcesses.KernelData=GaussianProcesses.EmptyData())
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

# function GaussianProcesses.cov_ij(k::WLK_SEIso, X1::AbstractMatrix, X2::AbstractMatrix, i::Int, j::Int, dim::Int)

#     ℓ2_sq, σ2_sq, ℓ_lin, σ2_lin, d, zₒ = k.ℓ2_sq, k.σ2_sq, k.ℓ_lin, k.σ2_lin, k.d, k.zₒ

#     x_star = @view(X1[:,i])
#     x = @view(X2[:,j])
    
#     z_star = x_star[3]
#     z = x[3]

#     k_val = 1        
    
#     # SquaredExponentialKernel
#     r = x[1:2] - x_star[1:2]
#     r_sq = dot_product(r,r)
#     k_val *= σ2_sq * exp(-r_sq/(2*ℓ2_sq))

#     # LinearExponentialKernel
#     r = abs(z - z_star)
#     k_val *= σ2_lin * exp(-r/(2*l_lin))

#     # WindLogLawKernel
#     ratio = log((z_star-d)/zₒ) / log((z-d)/zₒ)
#     # k_val *= (ratio - m[z_star]/uₓ) / (1 - m[z]/uₓ)
#     k_val *= ratio

#     return k_val
# end


function GaussianProcesses.cov(k::WLK_SEIso, x1::AbstractVector, x2::AbstractVector, data::GaussianProcesses.KernelData=GaussianProcesses.EmptyData())

    ℓ2_sq, σ2_sq, ℓ_lin, σ2_lin, d, zₒ = k.ℓ2_sq, k.σ2_sq, k.ℓ_lin, k.σ2_lin, k.d, k.zₒ

    x_star = x1
    x = x2
    
    z_star = x_star[3]
    z = x[3]

    k_val = 1        
    
    # SquaredExponentialKernel
    r = x[1:2] - x_star[1:2]
    r_sq = dot_product(r,r)
    k_val *= σ2_sq * exp(-r_sq/(2*ℓ2_sq))

    # LinearExponentialKernel
    r = abs(z - z_star)
    k_val *= σ2_lin * exp(-r/(2*l_lin))

    # WindLogLawKernel
    ratio = log((z_star-d)/zₒ) / log((z-d)/zₒ)
    # k_val *= (ratio - m[z_star]/uₓ) / (1 - m[z]/uₓ)
    k_val *= ratio

    return k_val
end

GaussianProcesses.cov_ij(k::WLK_SEIso, X1::AbstractMatrix, X2::AbstractMatrix, i::Int, j::Int, dim::Int) = GaussianProcesses.cov(k, @view(X1[:,i]), @view(X2[:,j]))

@inline GaussianProcesses.cov_ij(k::WLK_SEIso, X1::AbstractMatrix, X2::AbstractMatrix, data::GaussianProcesses.IsotropicData, i::Int, j::Int, dim::Int) = GaussianProcesses.cov_ij(k, X1, X2, i, j, dim)
@inline GaussianProcesses.cov_ij(k::WLK_SEIso, X1::AbstractMatrix, X2::AbstractMatrix, data::GaussianProcesses.KernelData, i::Int, j::Int, dim::Int) = GaussianProcesses.cov_ij(k, X1, X2, i, j, dim)

WLK_SEIso(ll_sq::T, lσ_sq::T, ll_lin::T, lσ_lin::T, d::T, zₒ::T) where T = WLK_SEIso{T}(exp(2 * ll_sq), exp(2 * lσ_sq), exp(ll_lin), exp(2 * lσ_lin), d, zₒ, [])

GaussianProcesses.num_params(se::WLK_SEIso) = 6

function GaussianProcesses.set_params!(se::WLK_SEIso, hyp::AbstractVector)
    length(hyp) == GaussianProcesses.num_params(se) || throw(ArgumentError("Should have retrieved $(GaussianProcesses.num_params(se)), but received $(length(hyp))."))
    ll_sq, lσ_sq, ll_lin, lσ_lin, d, zₒ = hyp   # extract hyp to vars
    se.ℓ2_sq, se.σ2_sq, se.ℓ_lin, se.σ2_lin, se.d, se.zₒ = exp(2 * ll_sq), exp(2 * lσ_sq), exp(ll_lin), exp(2 * lσ_lin), d, zₒ
end

GaussianProcesses.get_params(se::WLK_SEIso{T}) where T = T[log(se.ℓ2_sq) / 2, log(se.σ2_sq) / 2, log(se.ℓ_lin) / 2, log(se.σ2_lin) / 2, se.d, se.zₒ]

GaussianProcesses.get_param_names(se::WLK_SEIso) = [:ℓ2_sq, :σ2_sq, :ℓ_lin, :σ2_lin, :d, :zₒ]






# TODO: Write the derivative functions.


@inline function dKij_dθp(kern::WLK_SEIso, X1::AbstractMatrix, X2::AbstractMatrix, data, i::Int, j::Int, p::Int, dim::Int)
    xy_dist = euclidean_dist(X1[1:2,i], X2[1:2,j])  # Distance in coordinates 
    z_dist = abs(X1[3,i] - X2[3,j])                 # Distance in altitude
    return dk_dθp(kern, xy_dist, z_dist, p)
end

@inline dk_dθp(se::WLK_SEIso, r::Real, p::Int) = error("This call should not have been made.")


# Retrieves derivative of cov function w.r.t. param index `p`.
@inline function dk_dθp(se::WLK_SEIso, xy_dist::Number, z_dist::Number, p::Int)
    if p==1         # ℓ2_sq
        return dk_dll(se, r)

    elseif p==2     # σ2_sq
        return dk_dlσ(se, r)
    
    elseif p==3     # ℓ_lin
        return dk_dlσ(se, r)

    elseif p==4     # σ2_lin
        return dk_dlσ(se, r)
    
    elseif p==5     # d
        return dk_dlσ(se, r)

    elseif p==6     # zₒ
        return dk_dlσ(se, r)


    else
        return NaN
    end
end




@inline dk_dll(se::WLK_SEIso, r::Real) = r/se.ℓ2*cov(se,r)
