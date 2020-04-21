# Here, we will define the gausian process.
# Then, a custom kernel will be defined for the wind domain.
# Finally, a faster sampler will be designed.

using Distributions
include("./utils/misc.jl")

abstract type GP end
abstract type Mean end
abstract type Kernel end

struct GaussianProcess <: GP
    X::AbstractArray
    Y::AbstractArray
    mean::Mean
    kernel::Kernel
    σn::Float64      # noise variance
end

struct CustomMean <: Mean
    M::Union{Function, Dict}      # an object that returns the mean for any location x.
end

struct ConstantMean <: Mean
    M::Union{Function, Dict}
end

struct SquaredExponentialKernel <: Kernel
    l::Float64      # lengthscale
    σs::Float64     # signal variance
    K::Function     # function of the kernel

    function SquaredExponentialKernel(l::Float64, σs::Float64)
        function K(x, x_star; l::Float64=l, σs::Float64=σs)
            r = x - x_star
            r_sq = dot_product(r,r)
            return σs^2 * exp(-r_sq/(2*l))
        end
        new(l,σs,K)
    end
end

function predictPosterior(X_star, gp::GP)
    # Uses entire covariance matrix.

    X = gp.X
    Y = gp.Y
    
    σn = gp.σn
    Σ = σn .* eye(length(X))

    m = gp.mean.M
    M_X = [m(x) for x in X]
    M_Xs = [m(x_star) for x_star in X_star]
    
    k = gp.kernel.K
    K_X = [k(x1,x2) for x1 in X, x2 in X]
    K_Xs = [k(xs1,xs2) for xs1 in X_star, xs2 in X_star]
    K_XsX = [k(x_star,x) for x_star in X_star, x in X]
    K_XXs = [k(x,x_star) for x in X, x_star in X_star]  # Array(K_XsX')

    # Calculate posterior mean and variance
    μ_star = M_Xs + K_XsX * inv(K_X + Σ) * (Y - M_X)
    σ_star = K_Xs - K_XsX * inv(K_X + Σ) * K_XXs

    makeHermitian!(σ_star)    # gets rid of round-off errors.

    gp_dist = MvNormal(μ_star,σ_star)
end