using Distributions
using LinearAlgebra: diag, isposdef
using Random
using GaussianProcesses
import Statistics
using Optim
include("../src/dataparser_GWA.jl")
include("../src/utils/misc.jl")
include("../src/utils/GP_MCMC_with_tqdm.jl")
include("../src/GPLA.jl")

include("../src/utils/WLK_Matern32.jl")

#=========== PARAMS ==========#
# Parsing Farm Data
farm = "AltamontCA"
grid_dist = 220
grid_dist_obs = 220
altitudes = [100, 150, 200]
nx = 90
ny = nx

# Training Set
nx_start = div(nx, 5) * 2
nx_end = div(nx, 5) * 3
ny_start = nx_start
ny_end = nx_end

# Observation Set
grid_dist_obs = grid_dist * 2

# GPLA
NUM_NEIGHBORS = 5

# MCMC
N_ITER = 20
BURN = div(N_ITER, 2)
HMC_ϵ = 0.003

# LBFGS
N_SAMPLES = div(N_ITER, 5)
SET_SIZE = 10
#=============================#


# Load wind farm data
Map = get_3D_data(farm; altitudes=altitudes)

# Create field set
X_field0 = []
Y_field0 = Float64[]

for h in [150]
    append!(X_field0, [[i, j, Float64(h)] for j in 0.0:grid_dist:(ny-1)*grid_dist for i in 0.0:grid_dist:(nx-1)*grid_dist])
    append!(Y_field0, vec(Map[h][1:nx,1:ny]))
end

X_field = transform4GPjl(X_field0)
Y_field = Y_field0


# Create training set
X_train0 = []
Y0 = Float64[]

for h in [100, 200]
    append!(X_train0, [[i, j, Float64(h)] for j in Float64(ny_start-1)*grid_dist : grid_dist : Float64(ny_end-1)*grid_dist for i in Float64(nx_start-1)*grid_dist : grid_dist : Float64(nx_end-1)*grid_dist])
    append!(Y0, vec(Map[h][nx_start : nx_end, ny_start : ny_end]))
end

X_train = transform4GPjl(X_train0)
Y_train = Y0


# Create observation set
function get_Y_from_farm_location(loc, Map, grid_dist, nx, ny, h)
    idx = loc[1:2]./ grid_dist .+ 1.0
    return Map[h][Int.(idx)...]
end

X_obs0 = []

for h in [100, 200]
    append!(X_obs0, [[i, j, Float64(h)] for j in 0.0:grid_dist_obs:(ny-1)*grid_dist for i in 0.0:grid_dist_obs:(nx-1)*grid_dist])
end

X_obs = transform4GPjl(X_obs0)
Y_obs = map(lambda -> get_Y_from_farm_location(lambda, Map, grid_dist, nx, ny, 150), collect(eachcol(X_obs)))   # Retrieve the Y values on the Map for these coordinates.

σy = 10.0

# SquaredExponentialKernel PARAMS
l_sq = exp(1)^0.5 * grid_dist
σs_sq = exp(2)

# LinearExponentialKernel PARAMS
l_lin = 10000.0
σs_lin = 1.0

# WindLogLawKernel PARAMS
d = 0.0
zₒ = 0.05




# GPLA

# kern_gp = SEIso(log(l_sq), log(σs_sq))
# @time gpla_WF = GPLA(X_train, Y_train, NUM_NEIGHBORS, 0, 0, MeanConst(mean(Y_train)), kern_gp, log(σy))
# Theta_star = [-1.6885736184181186, 6.995882114315501 , 6.958324180613349 , 0.6929262639505916]
# GaussianProcesses.set_params!(gpla_WF, Theta_star)
# GaussianProcesses.fit!(gpla_WF, X_obs, Y_obs)

kern_gp_wll = WLK_SEIso(log(l_sq), log(σs_sq), log(l_lin), log(σs_lin), d, zₒ)
@time gpla_WF = GPLA(X_train, Y_train, NUM_NEIGHBORS, 0, 0, MeanConst(mean(Y_train)), kern_gp_wll, log(σy))
# Theta_star_wll = [-1.6885736184181186, 6.995882114315501 , 6.958324180613349 , 0.6929262639505916, 4.605170185988092, 0.0, 0.0, 0.05]
# GaussianProcesses.set_params!(gpla_WF, Theta_star_wll)
# GaussianProcesses.fit!(gpla_WF, X_obs, Y_obs)

# # exact GP
# gp_full_WF = GPE(X_train, Y_train, MeanConst(mean(Y_train)), kern_gp, log(σy))

# # extract predictions
# μ_exact_WF, Σ_exact_WF = predict_f(gp_full_WF, Xs_gp; full_cov=true)


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
Theta_list = sample_from_mcmc_posteriors(posterior_samples, N_SAMPLES, SET_SIZE)
optim_Theta_vals = optimize_Theta_list(Theta_list, gpla_WF)
Theta_star = sort(optim_Theta_vals)[end][end]

GaussianProcesses.set_params!(gpla_WF, Theta_star)
GaussianProcesses.fit!(gpla_WF, X_obs, Y_obs)

# # gpla predictions
# μ_gpla_WF, Σ_gpla_WF = predict_f(gpla_WF, X_field)

# Sample from the posterior
Y_sample = rand(gpla_WF, X_field)



### PLOT STUFF ###
using Plots


p1 = heatmap(reshape(Y0, (nx,ny)))
Plots.savefig(p1, "p1")

nx_star = length(0.0:grid_dist_obs:(nx-1)*grid_dist)
ny_star = length(0.0:grid_dist_obs:(ny-1)*grid_dist)

# predicted values
# p2 = heatmap(reshape(μ_exact_WF, (nx_star,ny_star)))
p3 = heatmap(reshape(μ_gpla_WF, (nx, ny)))
p4 = heatmap(reshape(Y_sample, (nx, ny)))
Plots.savefig(p4, "p4")