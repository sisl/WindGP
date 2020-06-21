using Distributions
using LinearAlgebra: diag, isposdef
using Random
using GaussianProcesses
import Statistics
using Optim
using Sobol
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

# SOBOL
NUM_POINTS = 1000

# GPLA
NUM_NEIGHBORS = 10

# MCMC
N_ITER = 500
BURN = div(N_ITER, 2)
HMC_ϵ = 0.001

# LBFGS
N_SAMPLES = div(N_ITER, 5)
SET_SIZE = 10
#=============================#


# Load wind farm data
Map = get_3D_data(farm; altitudes=altitudes)

# Sample from the wind farm
s = SobolSeq(2)  # Create 2-dimensional hyperspace
p = hcat([next!(s) for i = 1:NUM_POINTS]...)'  # Generate quasi-random samples
p = nearestRound(p .* grid_dist * (nx-1), grid_dist)  # Round to the nearest coordinates on the farm
p = unique(eachrow(p))  # Fetch the unique ones only
p = hcat(p...)


function get_Y_from_farm_location(loc, Map, grid_dist, nx, ny)
    idx = loc[1:2]./ grid_dist .+ 1.0
    return Map[150][Int.(idx)...]
end

# Plot the SobolSeq sampling plan:
using PyPlot
subplot(111, aspect="equal")
plot(p[1,:], p[2,:], "r.")
PyPlot.savefig("p2")

h = [150.0 for _ in 1:size(p,2)]
X_gp = Array(vcat(p,h'))
X_gp = hcat(sort(collect(eachcol(X_gp)))...)   # Sort by x,y,z order.
Y_gp = map(lambda -> get_Y_from_farm_location(lambda, Map, grid_dist, nx, ny), collect(eachcol(X_gp)))   # Retrieve the Y values on the Map for these coordinates.

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

function insert_Theta_to_GPLA!(new_gpla, Theta)
    logNoise_param = Theta[1]
    new_gpla.logNoise.value = logNoise_param
    
    mean_param = Theta[2]
    new_gpla.mean.β = mean_param
    
    kernel_params = Theta[3:end]
    new_gpla.kernel = SEIso(kernel_params...)
end

function optimize_Theta_list(Theta_list, gpla_WF)
    optim_results = []

    for Theta in tqdm(Theta_list)
        new_gpla = deepcopy(gpla_WF)
        insert_Theta_to_GPLA!(new_gpla, Theta)
        
        @time optimize!(new_gpla)
        push!(optim_results, (new_gpla.mll, Theta))
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