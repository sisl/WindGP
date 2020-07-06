using Distributions
using LinearAlgebra: diag, isposdef
using Random
using GaussianProcesses
import Statistics

""" The true function we will be simulating from. """
function fstar(x::Float64)
    return abs(x-5)*cos(2*x)
end

σy = 10.0
n = 5000

Random.seed!(1) # for reproducibility
Xdistr = Beta(7,7)
ϵdistr = Normal(0,σy)
x = rand(Xdistr, n)*10
X = Matrix(x')
Y = fstar.(x) .+ rand(ϵdistr,n)
k = SEIso(log(0.3), log(5.0))

# x values to predict
xx = range(0, stop=10, length=200)

# exact GP
@time gp_full = GPE(X, Y, MeanConst(mean(Y)), k, log(σy))

# extract predictions
@time μ_exact, Σ_exact = predict_f(gp_full, xx; full_cov=true)

# subset of regressors
Xu = Matrix(quantile(x, [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.98])')
@time gp_SOR = GaussianProcesses.SoR(X, Xu, Y, MeanConst(mean(Y)), k, log(σy));

# approximate predictions
@time predict_f(gp_SOR, xx; full_cov=true);




println("============================== Wind Farm ==============================")


using Optim
include("../src/dataparser_GWA.jl")
include("../src/utils/misc.jl")

farm = "AltamontCA"
grid_dist = 220
altitudes = [10, 50, 100, 150, 200]

Map = get_3D_data(farm; altitudes=altitudes)

nx = 10
ny = 10
# nx,ny = size(Map[altitudes[1]])

X0 = []
Y_gp = Float64[]

for h in [150]
    append!(X0, [[i, j, Float64(h)] for j in 0.0:grid_dist:(ny-1)*grid_dist for i in 0.0:grid_dist:(nx-1)*grid_dist])
    # append!(X,[[i,j,Float64(h)] for i in 1.0:nx for j in 1.0:ny])
    append!(Y_gp, vec(Map[h][1:nx,1:ny]))
end

grid_dist_Xs = 55
X_star = []

for h in [150]
    append!(X_star, [[i, j, Float64(h)] for j in 0.0:grid_dist_Xs:(ny-1)*grid_dist for i in 0.0:grid_dist_Xs:(nx-1)*grid_dist])
    # append!(X_star, [[i,j,h] for i in 1.0:nx for j in 1.0:ny])
end

X_gp = transform4GPjl(X0)
Xs_gp = transform4GPjl(X_star)

# SquaredExponentialKernel PARAMS
l_sq = exp(1)^0.5 * grid_dist
σs_sq = exp(2)

kern_gp = SEIso(log(l_sq), log(σs_sq))
Xu_gp_idxs = Int.(trunc.(quantile(1:length(X0), range(0, stop=1,length=800))))
Xu_gp = X_gp[:, Xu_gp_idxs]

# exact GP
gp_full_WF = GPE(X_gp, Y_gp, MeanConst(mean(Y_gp)), kern_gp, log(σy))

# extract predictions
μ_exact_WF, Σ_exact_WF = predict_f(gp_full_WF, Xs_gp; full_cov=true)

# subset of regressors
gp_SOR_WF = GaussianProcesses.SoR(X_gp, Xu_gp, Y_gp, MeanConst(mean(Y_gp)), kern_gp, log(σy))

# approximate predictions
μ_SoR_WF, Σ_SoR_WF = predict_f(gp_SOR_WF, Xs_gp; full_cov=true)


### PLOT STUFF ###
using Plots
p1 = heatmap(reshape(Y_gp, (nx,ny)))

nx_star = length(0.0:grid_dist_Xs:(nx-1)*grid_dist)
ny_star = length(0.0:grid_dist_Xs:(ny-1)*grid_dist)

# predicted values
p2 = heatmap(reshape(μ_exact_WF, (nx_star,ny_star)))
p3 = heatmap(reshape(μ_SoR_WF, (nx_star,ny_star)))