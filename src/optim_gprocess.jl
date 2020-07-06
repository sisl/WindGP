### GWA tests with Optim.

include("../src/custom_gprocess.jl")
include("../src/optimfunctions_gprocess.jl")
include("../src/dataparser_GWA.jl")
using DataStructures
using LinearAlgebra
using Optim

########## Initial values & Settings ##########

# Dataset PARAMS
# farms = ["AltamontCA", "CantonMtME", "DelawareMtTX", "WoodwardMtTX", "BullHillME", "SmokyHillsKS"]
farms = ["AltamontCA"]
altitudes = [10, 50, 100, 150, 200]     # meters
grid_dist = 220                         # meters
nx_step = 10
ny_step = 10

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
fₓ_val = 6.0                    # you will get division by zero if you set this to zero.

# Optim.jl PARAMS
opt_method = NelderMead()
opt_settings = Optim.Options(show_trace=true, iterations = 1)

opt_init = opt_final = [σn_gp, gp_mean_val, l_sq, σs_sq, l_lin, σs_lin, zₒ, fₓ_val]
opt_init0 = deepcopy(opt_init)

lower_constraint = zeros(length(opt_init))
upper_constraint = [Inf, 10.0, Inf, Inf, Inf, Inf, 1.0, 10.0]

###############################################

# Notice that you can stop anytime to retrieve `opt_final`.

farm = "AltamontCA"

Map = get_3D_data(farm; altitudes=altitudes)
MapPiecesX, MapPiecesY = splitFarm(Map, nx_step, ny_step, altitudes)


for (idx,pieceX) in enumerate(MapPiecesX)
    @show (idx,farm)

    global X = pieceX
    global Y = MapPiecesY[idx]

    result = Optim.optimize(objFunctionValue, opt_init, opt_method, opt_settings)
    opt_new = result.minimizer
    
    if kFoldTest(opt_new, idx) < kFoldTest(opt_init, idx)
        opt_init[:] = opt_new 
    end

end











# for farm in farms
#     Map = get_3D_data(farm; altitudes=altitudes)
#     nx_max, ny_max = size(Map[10])

#     for nx in  0 : nx_step : nx_max - nx_step
#         for ny in  0 : ny_step : ny_max - ny_step

#             @show (nx,ny,farm)

#             global X = []
#             global Y = []

#             for h in altitudes
#                 append!(X, [[j, i, Float64(h)] for i in nx*grid_dist:grid_dist:(nx+nx_step-1)*grid_dist for j in ny*grid_dist:grid_dist:(ny+ny_step-1)*grid_dist])
#                 append!(Y, vec(Map[h][nx+1:nx+nx_step, ny+1:ny+ny_step]))
#             end

#             result = Optim.optimize(objFunctionValue, opt_init, opt_method, opt_settings)
#             opt_final[:] = result.minimizer

#             opt_init[:] = opt_final
#         end
#     end
# end

## Compare initial and optimal points
# finalVal = objFunctionValue(opt_final)

