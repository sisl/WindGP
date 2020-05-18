using Optim
using LinearAlgebra: isposdef
using Distributions
include("./dataparser_GWA.jl")
include("./utils/wind_sparsekernel.jl")
include("./utils/misc.jl")


function objFunctionValue(X_gp, Y, d, NN, opt_init)
    """ Notice that this we are trying to minimize this function, therefore the negative of mll is returned. """

    l_sq, σs_sq, l_lin, σs_lin, zₒ = opt_init

    if l_lin<100.0 return Inf end
    if !(0.01 < zₒ < 0.5) return Inf end

    j = CustomWindSparseKernel(l_sq, σs_sq, l_lin, σs_lin, d, zₒ, NN)

    try 
        gp = GPE(X_gp, Y, MeanConst(0.0), j)
        return -gp.mll
    catch
        return Inf
    end
end

function fetchOptimizedGP(X_gp, Y, d, NN, opt_final)
    l_sq, σs_sq, l_lin, σs_lin, zₒ = opt_final
    j = CustomWindSparseKernel(l_sq, σs_sq, l_lin, σs_lin, d, zₒ, NN)
    gp = GPE(X_gp, Y, MeanConst(0.0), j)
    return gp
end



farm = "AltamontCA"
grid_dist = 220

Map = get_3D_data(farm; altitudes=[10, 50, 100, 150, 200])
Map_150 = Map[150]

nx = 20
ny = 20

X = []
Y = []

for h in [10, 50, 100, 150, 200]
    append!(X, [[j, i, Float64(h)] for i in 0.0:grid_dist:(nx-1)*grid_dist for j in 0.0:grid_dist:(ny-1)*grid_dist])
    # append!(X,[[i,j,Float64(h)] for i in 1.0:nx for j in 1.0:ny])
    append!(Y, vec(Map[h][1:nx,1:ny]))
end

X_star = []
append!(X_star, [[j, i, Float64(150)] for i in 0.0:grid_dist:(nx-1)*grid_dist for j in 0.0:grid_dist:(ny-1)*grid_dist])
# append!(X_star, [[i,j,150] for i in 1.0:nx for j in 1.0:ny])

X_gp = transform4GPjl(X)
Xs_gp = transform4GPjl(X_star)

## Create the final GP.
# GaussianProcess PARAMS
# σn_gp = 0.0

# SquaredExponentialKernel PARAMS
l_sq = exp(1) * grid_dist^2
σs_sq = exp(2)

# LinearExponentialKernel PARAMS
l_lin = 10000.0
σs_lin = 1.0

# WindLogLawKernel PARAMS
d = 0.0
zₒ = 0.05

# Optim.jl PARAMS
opt_method = NelderMead()
opt_settings = Optim.Options(show_trace=true, iterations = 1)

opt_init = [l_sq, σs_sq, l_lin, σs_lin, zₒ]
opt_init0 = deepcopy(opt_init)

numNeighbors = 10
NN = getNearestNeighbors(X_gp, X_gp, numNeighbors)

result = Optim.optimize(lambda -> objFunctionValue(X_gp, Y, d, NN, lambda), opt_init, opt_method, opt_settings)
opt_final = result.minimizer

gp = fetchOptimizedGP(X_gp, Y, d, NN, opt_final)



######## Find mll with the new method.



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






function GaussianProcesses.cov_ij(k::CustomWindKernel, x1::AbstractArray, x2::AbstractArray)

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

# TODOs: 1) Fix `noise`.

function find_mll(numNeighbors)
    mll = 0
    
    NN = getNearestNeighbors(X_gp, X_gp, numNeighbors)

    for idx in 1:size(X_gp,2)
        x = X_gp[:,idx]
        neighbors_of_x = NN[x]
    
        X_active = X_gp[:, neighbors_of_x]
        
        Kxx = getSparse_K(kernel, X_active, X_active)
    
        K_fx = getSparse_K(kernel, x, X_active)
        K_xf = getSparse_K(kernel, X_active, x)
        K_f = getSparse_K(kernel, x, x)[1]
    
        mean_f = GaussianProcesses.mean(gp.mean, gp.x)
        mf = mean_f[idx]
        mx = mean_f[neighbors_of_x]
        yx = gp.y[neighbors_of_x]
        yf = gp.y[idx]
    
        μ_star = mf + dot(K_fx, inv(Kxx) * (yx - mx))
        σ_star = K_f - dot(K_fx, inv(Kxx) * K_xf)
        
        noise = exp(2*-2)+eps()
        σ_star = abs.(σ_star + noise)
    
        (yf - μ_star)^2 / σ_star
    
        mll -= ((yf - μ_star)^2 / σ_star + log(σ_star) + log2π) / 2
        @show mll
    end
    return mll
end

mu, sigma = predict_f(gp, gp.x)
function doubleCheck_mll(gp, mu, sigma)

    mll = 0
    # mu, sigma = predict_f(gp, gp.x)

    for (i,y) in enumerate(gp.y)
        @show mll
        mll += -0.5*((y - mu[i])/sigma[i]^2)^2 - 0.5*log(2*pi) - log(sigma[i]^2)
        # mll -= 0.5*( (y-mu[i])^2*sigma[i] + log2π + 2*log(sqrt(sigma[i])) ) 
    end
    return mll
end