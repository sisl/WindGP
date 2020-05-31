using Optim
using LinearAlgebra: isposdef
using Distributions
include("../src/dataparser_GWA.jl")
include("../src/utils/wind_sparsekernel.jl")
include("../src/utils/misc.jl")


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

function doubleCheck_mll(gp, X_gp)
    μ_star, Σ_star = predict_f(gp, X_gp)
    σ = sqrt.(abs.(Σ_star))
    MLL = sum( -0.5*((Y - μ_star)./σ).^2 - 0.5*log(2*pi).*ones(gp.nobs,1) - log.(σ) )
end


"""
Test Structure:
X_gp  --> altitudes = [10, 50, 100, 200];   grid_dist = 220
Xs_gp --> altitudes = [150, 195];           grid_dist = 55
"""

farm = "AltamontCA"
grid_dist = 220
altitudes=[10, 50, 100, 150, 200]

Map = get_3D_data(farm; altitudes=altitudes)

# nx = 30
# ny = 30
nx,ny = size(Map[altitudes[1]])

X = []
Y = Float64[]

for h in [10, 50, 100, 200]
    append!(X, [[j, i, Float64(h)] for i in 0.0:grid_dist:(nx-1)*grid_dist for j in 0.0:grid_dist:(ny-1)*grid_dist])
    # append!(X,[[i,j,Float64(h)] for i in 1.0:nx for j in 1.0:ny])
    append!(Y, vec(Map[h][1:nx,1:ny]))
end

grid_dist_Xs = 55
X_star = []

for h in [150, 195]
    append!(X_star, [[j, i, Float64(h)] for i in 0.0:grid_dist_Xs:(nx-1)*grid_dist for j in 0.0:grid_dist_Xs:(ny-1)*grid_dist])
    # append!(X_star, [[i,j,h] for i in 1.0:nx for j in 1.0:ny])
end

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
opt_settings = Optim.Options(show_trace=true, iterations = 25)

opt_init = [l_sq, σs_sq, l_lin, σs_lin, zₒ]
opt_init0 = deepcopy(opt_init)

numNeighbors = 5
NN = getNearestNeighborsFaster(X_gp, X_gp, numNeighbors, grid_dist, altitudes)

# result = Optim.optimize(lambda -> objFunctionValue(X_gp, Y, d, NN, lambda), opt_init, opt_method, opt_settings)
# opt_final = result.minimizer


gp0 = fetchOptimizedGP(X_gp, Y, d, NN, opt_init0)

# A solution for "AltamontCA" after 25 iterations.
opt_final = [240508.59409150065, 10.543542479573873, 1326.788505255554, 0.09138343487060108, 0.09633453277598292]
gp = fetchOptimizedGP(X_gp, Y, d, NN, opt_final)
