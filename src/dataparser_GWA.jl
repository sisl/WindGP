# Parses the CFD data provided in ../data/DWA/

using DelimitedFiles
using GaussianProcesses
using Plots
include("./utils/misc.jl"),
include("./utils/linexp_kernel.jl")

######## Parameters ########

datapath = "../data/GWA/AltamontCA/"
filename = "custom_wind-speed_100m.xyz"

############################


struct data_2D
    x:: Array{Float64}
    y:: Array{Float64}

    avgSpeed:: Array{Float64}
    function data_2D(datapath, filename)
        data_read = readdlm(datapath*filename, ' ', Float64)
        
        x = data_read[:,1]
        y = data_read[:,2]
        avgSpeed = data_read[:,3]

        new(x,y,avgSpeed)
    end
end

# Load data
D = data_2D(datapath, filename)
Map = reshape(D.avgSpeed, length(unique(D.x)), length(unique(D.y)))

# Create GP
# x_nm = stretch(D.x,1,2)
# y_nm = stretch(D.y,1,2)
# coords = vcat(x_nm',y_nm')
# gp_2D_nm = GP(coords, D.avgSpeed, MeanZero(), SEIso(0.0,0.0))


coords = vcat(D.x',D.y')
gp_2D = GP(coords, D.avgSpeed, MeanZero(), SEIso(0.0,0.0))

# julia> extrema(coords[1,:])
# (-121.80967731781246, -121.55717731781245)

# julia> extrema(coords[2,:])
# (37.63701874773503, 37.87451874773504)


loc = coords[1:2,[4,70,800]]
predict_f()