### GWA tests with Optim.

include("../src/custom_gprocess.jl")
include("../src/optimfunctions_gprocess.jl")
include("../src/dataparser_GWA.jl")
using DataStructures
using LinearAlgebra
using Optim

farm = "AltamontCA"

altitudes = [10, 50, 100, 150, 200]     # meters
grid_dist = 220                         # meters

Map = get_3D_data(farm; altitudes=altitudes)


nx = 10
ny = 10

X = []
Y = []

for h in altitudes
    append!(X, [[i, j, Float64(h)] for i in 0.0:grid_dist:(nx-1)*grid_dist for j in 0.0:grid_dist:(ny-1)*grid_dist])
    # append!(X,[[i,j,Float64(h)] for i in 1.0:nx for j in 1.0:ny])
    append!(Y, vec(Map[h][1:nx,1:ny]))
end

##### OPTIMIZATION #####

## Initial values ##

# GaussianProcess PARAMS
σn_gp = 0.0
gp_mean_val = 6.5               # you will get division by zero if you set this equal to fₓ.

# SquaredExponentialKernel PARAMS
l_sq = exp(1) * grid_dist^2
σs_sq = exp(2)

# LinearExponentialKernel PARAMS
l_lin = 1000.0
σs_lin = exp(1)

# WindLogLawKernel PARAMS
d = 0.0
zₒ = 0.05
fₓ_val = average(Y)             # you will get division by zero if you set this to zero.


## Optimization options
opt_settings = Optim.Options(show_trace=true, iterations = 100)
opt_init = [σn_gp, gp_mean_val, l_sq, σs_sq, l_lin, σs_lin, zₒ, fₓ_val]
result = Optim.optimize(objFunctionValue, opt_init, opt_settings)
opt_final = result.minimizer

## Compare initial and optimal points
initVal = objFunctionValue(opt_init)
finalVal = objFunctionValue(opt_final)
