"""
Custom kernel for the wind farm domain.
This version uses SEIso as the x-y kernel.
"""

mutable struct WLL_SEIso{T<:Real} <: Isotropic{SqEuclidean}
    # SquaredExponentialKernel
    l2_sq::T       # lengthscale
    σ2_sq::T      # signal variance

    # LinearExponentialKernel
    l2_lin::T      # lengthscale
    σ2_lin::T     # signal variance

    # WindLogLawKernel
    d::T          # zero-plane displacement
    zₒ::T         # roughness length

    # Priors for kernel parameters
    priors::Array
end

cov(se::WLL_SEIso, r::Number) = error("This call should not have been made.")

# TODO: Add the cov and cov! functions here

WLL_SEIso(l_sq::T, σ_sq::T, l_lin::T, σ_lin::T, d::T, zₒ::T) where T = WLL_SEIso{T}(exp(2 * l_sq), exp(2 * σ_sq), exp(2 * l_lin), exp(2 * σ_lin), d, zₒ, [])

num_params(se::WLL_SEIso) = 6

function set_params!(se::WLL_SEIso, hyp::AbstractVector)
    length(hyp) == num_params(se) || throw(ArgumentError("Should have retrieved $(num_params(se)), but received $(length(hyp))."))
    l_sq, σ_sq, l_lin, σ_lin, d, zₒ = hyp   # extract hyp to vars
    se.l2_sq, se.σ2_sq, se.l2_lin, se.σ2_lin, se.d, se.zₒ = exp(2 * l_sq), exp(2 * σ_sq), exp(2 * l_lin), exp(2 * σ_lin), d, zₒ
end

get_params(se::WLL_SEIso{T}) where T = T[log(se.l2_sq) / 2, log(se.σ2_sq) / 2, log(se.l2_lin) / 2, log(se.σ2_lin) / 2, se.d, se.zₒ]

get_param_names(se::WLL_SEIso) = [:l2_sq, :σ2_sq, :l2_lin, :σ2_lin, :d, :zₒ]



@inline function dKij_dθp(kern::WLL_SEIso, X1::AbstractMatrix, X2::AbstractMatrix, data, i::Int, j::Int, p::Int, dim::Int)
    xy_dist = euclidean_dist(X1[1:2,i], X2[1:2,j])  # Distance in coordinates 
    z_dist = abs(X1[3,i] - X2[3,j])                 # Distance in altitude
    return dk_dθp(kern, xy_dist, z_dist, p)
end

@inline function dk_dθp(se::WLL_SEIso, r::Real, p::Int) = error("This call should not have been made.")


# Retrieves derivative of cov function w.r.t. param index `p`.
@inline function dk_dθp(se::WLL_SEIso, xy_dist::Number, z_dist::Number, p::Int)
    if p==1         # l2_sq
        return dk_dll(se, r)

    elseif p==2     # σ2_sq
        return dk_dlσ(se, r)
    
    elseif p==3     # l2_lin
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




@inline dk_dll(se::WLL_SEIso, r::Real) = r/se.ℓ2*cov(se,r)
