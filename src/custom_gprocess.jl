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

struct ConstantMean <: Mean
    M::Union{Function, Dict}
end

struct CustomMean <: Mean
    M::Union{Function, Dict}      # an object that returns the mean for any location x.
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

struct WindLogLawKernel <: Kernel
    """ Considers only the z-dimension"""
    m::Mean                       # mean function
    d::Float64                    # zero-plane displacement
    zₒ::Float64                   # roughness length
    fₓ::Union{Function, Dict}     # object that outputs y = f(x)
    K::Function                   # function of the kernel

    function WindLogLawKernel(m::Mean, d::Float64, zₒ::Float64, fₓ::Union{Function, Dict})
        function K(z_star, z; m::Mean=m, d::Float64=d, zₒ::Float64=zₒ, fₓ::Union{Function, Dict}=fₓ)
            m = m.M
            uₓ = fₓ(z)

            # Calculate u(z_star)/u(z).
            function logLaw(z_star, z, zₒ, d)
                ratio = log((z_star-d)/zₒ) / log((z-d)/zₒ)
                
                if ratio > 1
                    return 1/ratio
                else
                    return ratio
                end
                # return ratio

            end
            
            return (logLaw(z_star, z, zₒ, d) - m(z_star)/uₓ) / (1 - m(z)/uₓ)
        end
        new(m,d,zₒ,fₓ,K)
    end
end

struct CompositeWindKernel <: Kernel
    K1::Kernel      # Kernel 1 (x,y)
    K2::Kernel      # Kernel 2 (z)
    K::Function     # Combination of Kernel 1 and Kernel 2

    function CompositeWindKernel(K1::Kernel, K2::Kernel)
        K(x, x_star, K1::Kernel=K1, K2::Kernel=K2) = K1.K(x[1:2], x_star[1:2]) * K2.K(x[3], x_star[3])
        new(K1,K2,K)
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
    

    function sek(x, x_star; l::Float64=l, σs::Float64=σs)
        r = x - x_star
        r_sq = dot_product(r,r)
        return σs^2 * exp(-r_sq/(2*l))
    end

    function ck(x1,x2;ll=1000)
        r = x1[3] - x2[3]
        r_sq = abs(r)
        return exp(-r_sq/(2*ll))
    end

    K_X = [k(x1,x2)*ck(x1,x2) for x1 in X, x2 in X]
    # K_X .*= eye(length(X)) .* length(X)
    

    K_Xs = [k(xs1,xs2) for xs1 in X_star, xs2 in X_star]
    K_XsX = [k(x_star,x) for x_star in X_star, x in X]
    K_XXs = [k(x,x_star) for x in X, x_star in X_star]  # Array(K_XsX')

    K_XsX = [sek(x1[1:2], x2[1:2])*exactLogLaw(x1[3], x2[3], gp.kernel.K2.zₒ, gp.kernel.K2.d)*ck(x1,x2) for x1 in X_star, x2 in X]


    # Calculate posterior mean and variance
    μ_star = M_Xs + K_XsX * inv(K_X + Σ) * (Y - M_X)

    K_XsX = [k(x_star,x) for x_star in X_star, x in X]
    σ_star = K_Xs - K_XsX * inv(K_X + Σ) * K_XXs

    makeHermitian!(σ_star)    # gets rid of round-off errors.

    gp_dist = Distributions.MvNormal(μ_star,σ_star)
end