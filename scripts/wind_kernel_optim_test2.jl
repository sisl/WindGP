using Optim
using LinearAlgebra: isposdef
using Distributions
include("./dataparser_GWA.jl")
include("./utils/wind_kernel.jl")
include("./utils/misc.jl")


function objFunctionValue(X_gp, Y, d, opt_init)
    """ Notice that this we are trying to minimize this function, therefore the negative of mll is returned. """

    l_sq, σs_sq, l_lin, σs_lin, zₒ = opt_init

    if l_lin<100.0 return Inf end
    if !(0.01 < zₒ < 0.5) return Inf end

    j = CustomWindKernel(l_sq, σs_sq, l_lin, σs_lin, d, zₒ)

    try 
        gp = GPE(X_gp, Y, MeanConst(0.0), j)
        return -gp.mll
    catch
        return Inf
    end
end

function fetchOptimizedGP(X_gp, Y, d, opt_final)
    l_sq, σs_sq, l_lin, σs_lin, zₒ = opt_final
    j = CustomWindKernel(l_sq, σs_sq, l_lin, σs_lin, d, zₒ)
    gp = GPE(X_gp, Y, MeanConst(0.0), j)
    return gp
end



farm = "AltamontCA"
grid_dist = 220

Map = get_3D_data(farm; altitudes=[10, 50, 100, 150, 200])
Map_150 = Map[150]

nx = 31
ny = 30

X = []
Y = []

for h in [150]
    append!(X, [[j, i, Float64(h)] for i in 0.0:grid_dist:(nx-1)*grid_dist for j in 0.0:grid_dist:(ny-1)*grid_dist])
    # append!(X,[[i,j,Float64(h)] for i in 1.0:nx for j in 1.0:ny])
    append!(Y, vec(Map[h][1:nx,1:ny]))
end

X_star = []

for h in [150]
    append!(X_star, [[j, i, Float64(h)] for i in 0.0:grid_dist:(nx-1)*grid_dist for j in 0.0:grid_dist:(ny-1)*grid_dist])
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
opt_settings = Optim.Options(show_trace=true, iterations = 1)

opt_init = [l_sq, σs_sq, l_lin, σs_lin, zₒ]
opt_init0 = deepcopy(opt_init)

# result = Optim.optimize(lambda -> objFunctionValue(X_gp, Y, d, lambda), opt_init, opt_method, opt_settings)
# opt_final = result.minimizer

# gp_final = fetchOptimizedGP(X_gp, Y, d, opt_final)

gp_final = fetchOptimizedGP(X_gp, Y, d, opt_init0)


pred_mu, pred_sigma = predict_f(gp_final, Xs_gp)

# Get mean predicted and ground truth values.
Y_star = Array(reshape(pred_mu, Int(nx), Int(ny)))
ground_truth = Map_150[1:Int(nx),1:Int(ny)]

# Calculate error percentage.
ErrorPct = (Y_star - ground_truth) ./ ground_truth *100
AvgErrorPct = average(abs.(ErrorPct))

# Sample randomly.
rand_sample = rand(gp_final, Xs_gp)
Y_rand = Array(reshape(rand_sample, Int(nx), Int(ny)))
