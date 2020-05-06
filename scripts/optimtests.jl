### GWA tests with Optim.

include("../src/custom_gprocess.jl")
include("../src/optimfunctions_gprocess.jl")
include("../src/dataparser_GWA.jl")
using DataStructures
using LinearAlgebra
using Optim

farm = "AltamontCA"

altitudes = [100, 150, 200]     # meters
grid_dist = 220                         # meters

Map = get_3D_data(farm; altitudes=altitudes)


nx = 10
ny = 10

X = []
Y = []

for h in altitudes
    # append!(X, [[i, j, Float64(h)] for i in 0.0:grid_dist:(nx-1)*grid_dist for j in 0.0:grid_dist:(ny-1)*grid_dist])
    append!(X,[[i,j,Float64(h)] for i in 1.0:nx for j in 1.0:ny])
    append!(Y, vec(Map[h][1:nx,1:ny]))
end


# GaussianProcess PARAMS
σn_gp = 0.0
gp_mean_val = 0.0               # you will get division by zero if you set this equal to fₓ.

# SquaredExponentialKernel PARAMS
l_sq = exp(1)
σs_sq = exp(2)

# LinearExponentialKernel PARAMS
l_lin = 10000.0
σs_lin = exp(1)

# WindLogLawKernel PARAMS
d = 0.0
zₒ = 0.05
fₓ_val = average(Y)             # you will get division by zero if you set this to zero.

##### OPTIMIZATION #####


# onjVal = objFunctionValue(X, Y, σn_gp, gp_mean_val, l_sq, σs_sq, l_lin, σs_lin, d, zₒ, fₓ_val)

gp_mean = CustomMean(DefaultDict(gp_mean_val))
fₓ = DefaultDict(fₓ_val)
gp_kernel = CustomTripleKernel(l_sq, σs_sq, l_lin, σs_lin, d, zₒ, fₓ, gp_mean)

n = length(Y)
Σ = σn_gp .* eye(n)

m_dict = gp_mean.M
m(x) = m_dict[x]
M_X = [m(x) for x in X]

K_X = getKernelMatrix(X,X,gp_kernel)
K_X += 1e-6 .* eye(length(X))

@show modelComplexityTerm = -0.5 * log(det(K_X + Σ))
@show dataFitTerm = -0.5 * ((Y - M_X)' * inv(K_X + Σ) * (Y - M_X))
@show constantTerm = -0.5 * n * log(2*pi)

modelComplexityTerm + dataFitTerm + constantTerm