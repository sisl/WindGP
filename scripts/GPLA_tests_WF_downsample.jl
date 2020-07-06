using Distributions
using LinearAlgebra: diag, isposdef
using Random
using GaussianProcesses
import Statistics
using Optim
using ImageTransformations
include("../src/dataparser_GWA.jl")
include("../src/utils/misc.jl")
include("../src/utils/GP_MCMC_with_tqdm.jl")
include("../src/GPLA.jl")

#=========== PARAMS ==========#
# Parsing Farm Data
farm = "AltamontCA"
grid_dist = 220
grid_dist_Xs = 220
altitudes = [10, 50, 100, 150, 200]
nx = 90
ny = nx

# ImageTransformations
IMG_SIZE = (div(nx, 2), div(ny, 2))

# GPLA
NUM_NEIGHBORS = 10

# MCMC
N_ITER = 100
BURN = div(N_ITER, 2)
HMC_ϵ = 0.001

# LBFGS
N_SAMPLES = div(N_ITER, 5)
SET_SIZE = 10
#=============================#


# Load wind farm data
Map = get_3D_data(farm; altitudes=altitudes)

# Downsample farm data
img = Map[150][1:nx,1:ny]
img_ds = imresize(img, IMG_SIZE)

img_locs = [[i,j] for i in 0.0:grid_dist:(nx-1)*grid_dist, j in 0.0:grid_dist:(ny-1)*grid_dist]
img_locs_ds = imresize(img_locs, (45, 45))

Y_gp = vec(img_ds)
X = [[item...,150] for item in vec(img_locs_ds)]
X_gp = transform4GPjl(X)


# Points to approximate for in the end
X_star = []

for h in [150]
    append!(X_star, [[i, j, Float64(h)] for j in 0.0:grid_dist_Xs:(ny-1)*grid_dist for i in 0.0:grid_dist_Xs:(nx-1)*grid_dist])
end

Xs_gp = transform4GPjl(X_star)

σy = 10.0

# SquaredExponentialKernel PARAMS
l_sq = exp(1)^0.5 * grid_dist
σs_sq = exp(2)

kern_gp = SEIso(log(l_sq), log(σs_sq))

# # exact GP
# gp_full_WF = GPE(X_gp, Y_gp, MeanConst(mean(Y_gp)), kern_gp, log(σy))

# # extract predictions
# μ_exact_WF, Σ_exact_WF = predict_f(gp_full_WF, Xs_gp; full_cov=true)


# GPLA
@time gpla_WF = GPLA(X_gp, Y_gp, 10, 0, 0, MeanConst(mean(Y_gp)), kern_gp, log(σy))

# MCMC (Note: takes about 4-5 mins for N_ITER=100, HMC_ϵ=0.025 with 400 pts)
@time posterior_samples = mcmc(gpla_WF, nIter=N_ITER, burn=BURN, Lmin=20, Lmax=30, ε=HMC_ϵ)  


function sample_from_mcmc_posteriors(posterior_samples, N_SAMPLES=50, SET_SIZE=10)
    result = Array[]

    for _ in 1:SET_SIZE
        temp = posterior_samples[:, (sample(1:size(posterior_samples,2), N_SAMPLES))]  # get N_SAMPLES from posterior_samples
        temp = average(collect(eachcol(temp)))  # take the average of these samples
        push!(result, temp)
    end
    
    return result
end


# LBFGS
Theta_list = sample_from_mcmc_posteriors(posterior_samples, N_SAMPLES, SET_SIZE)

function optimize_Theta_list(Theta_list, gpla_WF)
    optim_results = []

    for Theta in tqdm(Theta_list)
        new_gpla = deepcopy(gpla_WF)
        GaussianProcesses.set_params!(new_gpla, Theta)
        GaussianProcesses.fit!(new_gpla, X_obs, Y_obs)

        @time optimize!(new_gpla)
        optim_params = GaussianProcesses.get_params(new_gpla)
        push!(optim_results, (new_gpla.mll, optim_params))
    end

    return optim_results
end


# Retrieve the Theta value with the lowest overall mll
optim_Theta_vals = optimize_Theta_list(Theta_list, gpla_WF)
Theta_star = sort(optim_Theta_vals)[end][end]
insert_Theta_to_GPLA!(gpla_WF, Theta_star)

# gpla predictions
μ_gpla_WF, Σ_gpla_WF = predict_f(gpla_WF, Xs_gp)




### PLOT STUFF ###
using Plots

X0 = []
Y0 = Float64[]

for h in [150]
    append!(X0, [[i, j, Float64(h)] for j in 0.0:grid_dist:(ny-1)*grid_dist for i in 0.0:grid_dist:(nx-1)*grid_dist])
    append!(Y0, vec(Map[h][1:nx,1:ny]))
end

p1 = heatmap(reshape(Y0, (nx,ny)))
Plots.savefig(p1, "p1")

nx_star = length(0.0:grid_dist_Xs:(nx-1)*grid_dist)
ny_star = length(0.0:grid_dist_Xs:(ny-1)*grid_dist)

# predicted values
# p2 = heatmap(reshape(μ_exact_WF, (nx_star,ny_star)))
p3 = heatmap(reshape(μ_gpla_WF, (nx_star,ny_star)))
Plots.savefig(p3, "p3")