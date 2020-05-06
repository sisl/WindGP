# Here, we will define the gausian process.
# Then, a custom kernel will be defined for the wind domain.
# Finally, a faster sampler will be designed.

using Distributions
using ProgressBars
using DataStructures

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
    M::Union{Dict, DefaultDict}
end

struct CustomMean <: Mean
    M::Union{Dict, DefaultDict}      # an object that returns the mean for any location x.
end


struct SquaredExponentialKernel <: Kernel
    l::Float64      # lengthscale
    σs::Float64     # signal variance
end

function getKernelValue(x_star, x, kernel::SquaredExponentialKernel)    
    l, σs  = kernel.l, kernel.σs
    r = x - x_star
    r_sq = dot_product(r,r)
    return σs^2 * exp(-r_sq/(2*l))
end


struct LinearExponentialKernel <: Kernel
    l::Float64      # lengthscale
    σs::Float64     # signal variance
end

function getKernelValue(x_star, x, kernel::LinearExponentialKernel)    
    l, σs  = kernel.l, kernel.σs
    r = abs(x - x_star)
    return σs^2 * exp(-r/(2*l))
end


struct CustomTripleKernel <: Kernel
    # SquaredExponentialKernel
    l_sq::Float64      # lengthscale
    σs_sq::Float64     # signal variance

    # LinearExponentialKernel
    l_lin::Float64      # lengthscale
    σs_lin::Float64     # signal variance

    # WindLogLawKernel
    d::Float64                                  # zero-plane displacement
    zₒ::Float64                                 # roughness length
    fₓ::Union{Dict, DefaultDict}                # object that outputs y = f(x)
    m::Mean                                     # GP mean function
end


struct WindLogLawKernel <: Kernel
    """ Should consider only the z-dimension """
    m::Mean                                     # mean function
    d::Float64                                  # zero-plane displacement
    zₒ::Float64                                 # roughness length
    fₓ::Union{Dict, DefaultDict}                # object that outputs y = f(x)
end

function getKernelValue(z_star, z, kernel::WindLogLawKernel)    
    m_dict, d, zₒ, fₓ =  kernel.m.M, kernel.d, kernel.zₒ, kernel.fₓ
    m(x) = m_dict[x]
    uₓ = fₓ[z]
    ratio = log((z_star-d)/zₒ) / log((z-d)/zₒ)
    return (ratio - m(z_star)/uₓ) / (1 - m(z)/uₓ)
end


struct CompositeWindKernel <: Kernel
    Kxy::AbstractArray{T} where T <: Kernel       # Array of Kernels used for fitting (x,y)
    Kz ::AbstractArray{T} where T <: Kernel       # Array of Kernels used for fitting (z)
end

function getKernelValue(x_star, x, kernel::CompositeWindKernel)    
    Kxy, Kz = kernel.Kxy, kernel.Kz
    K_val = 1

    for kernel in Kxy
        K_val *= getKernelValue(x_star[1:2], x[1:2], kernel)
    end

    for kernel in Kz
        K_val *= getKernelValue(x_star[3], x[3], kernel)
    end

    return K_val
end


function predictPosterior(X_star, gp::GP)
    # Uses entire covariance matrix.

    X = gp.X
    Y = gp.Y
    
    σn = gp.σn
    Σ = σn .* eye(length(X))

    m_dict = gp.mean.M
    m(x) = m_dict[x]

    M_X = [m(x) for x in X]
    M_Xs = [m(x_star) for x_star in X_star]
    
    kernel = gp.kernel                                      # composite kernel.
    k(x_star, x) = getKernelValue(x_star, x, kernel)        # function of kernel.

    K_X = [k(x1,x2) for x1 in X, x2 in tqdm(X)]
    K_X += 1e-6 .* eye(length(X))
    
    K_Xs = [k(xs1,xs2) for xs1 in X_star, xs2 in tqdm(X_star)]
    K_XsX = [k(x_star,x) for x_star in X_star, x in tqdm(X)]
    K_XXs = [k(x,x_star) for x in X, x_star in tqdm(X_star)]

    # Calculates posterior mean and variance.
    μ_star = M_Xs + K_XsX * inv(K_X + Σ) * (Y - M_X)
    σ_star = K_Xs - K_XsX * inv(K_X + Σ) * K_XXs

    # dropBelowThreshold!(σ_star; threshold=1e-6)     # gets rid of small neg. numbers preventing Hermiticity.
    makeHermitian!(σ_star; inflation=1e-4)          # gets rid of round-off errors preventing Hermiticity.
    # σ_star = abs.(σ_star) 

    gp_dist = Distributions.MvNormal(μ_star,σ_star)
end


function getKernelMatrix(X_star, X, kernel::CustomTripleKernel)    
    l_sq, σs_sq  = kernel.l_sq, kernel.σs_sq
    l_lin, σs_lin  = kernel.l_lin, kernel.σs_lin
    m, d, zₒ, fₓ =  kernel.m.M, kernel.d, kernel.zₒ, kernel.fₓ

    K_matrix = Array{Float64,2}(undef, length(X_star), length(X))

    for i in tqdm(1:length(X_star))
        # println("$i of $(length(X_star))")
        for (j,x) in enumerate(X)
            k_val = 1
            
            x_star = X_star[i]
            
            z_star = x_star[3]
            z = x[3]
            uₓ = fₓ[z]

            # SquaredExponentialKernel
            r = x[1:2] - x_star[1:2]
            r_sq = dot_product(r,r)
            k_val *= σs_sq^2 * exp(-r_sq/(2*l_sq))

            # LinearExponentialKernel
            r = abs(z - z_star)
            k_val *= σs_lin^2 * exp(-r/(2*l_lin))

            # WindLogLawKernel
            ratio = log((z_star-d)/zₒ) / log((z-d)/zₒ)
            k_val *= (ratio - m[z_star]/uₓ) / (1 - m[z]/uₓ)

            K_matrix[i,j] = k_val
        end
    end
    return K_matrix
end

function predictPosteriorFaster(X_star, gp::GP)
    # Uses entire covariance matrix.

    X = gp.X
    Y = gp.Y
    
    σn = gp.σn
    Σ = σn .* eye(length(X))

    m_dict = gp.mean.M
    m(x) = m_dict[x]

    M_X = [m(x) for x in X]
    M_Xs = [m(x_star) for x_star in X_star]
    
    kernel = gp.kernel

    display("## Calculating K_X (1/4) ##")
    K_X = getKernelMatrix(X,X,kernel)
    K_X += 1e-6 .* eye(length(X))
    
    display("## Calculating K_Xs (2/4) ##")
    K_Xs = getKernelMatrix(X_star,X_star,kernel)

    display("## Calculating K_XsX (3/4) ##")
    K_XsX = getKernelMatrix(X_star,X,kernel)

    display("## Calculating K_XXs (4/4) ##")
    K_XXs = getKernelMatrix(X,X_star,kernel)

    # Calculates posterior mean and variance.
    display("## Hold on while I take the inverse of K_X of size $(size(K_X)) ##")
    invK_X = inv(K_X + Σ)
    μ_star = M_Xs + K_XsX * invK_X * (Y - M_X)
    σ_star = K_Xs - K_XsX * invK_X * K_XXs

    # dropBelowThreshold!(σ_star; threshold=1e-6)     # gets rid of small neg. numbers preventing Hermiticity.
    makeHermitian!(σ_star; inflation=1e-4)          # gets rid of round-off errors preventing Hermiticity.
    # σ_star = abs.(σ_star)                           # gets rid of negative covariance.

    gp_dist = Distributions.MvNormal(μ_star,σ_star)
end
