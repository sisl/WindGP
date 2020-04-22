# Parses the CFD data provided in ../data/DWA/

using DelimitedFiles
# using GaussianProcesses
# using Plots
include("./utils/misc.jl")

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
Map_100 = deepcopy(Map)

# Create GP
# x_nm = stretch(D.x,1,2)
# y_nm = stretch(D.y,1,2)
# coords = vcat(x_nm',y_nm')
# gp_2D_nm = GP(coords, D.avgSpeed, MeanZero(), SEIso(0.0,0.0))


# coords = vcat(D.x',D.y')
# gp_2D = GP(coords, D.avgSpeed, MeanZero(), SEIso(0.0,0.0))

# julia> extrema(coords[1,:])
# (-121.80967731781246, -121.55717731781245)

filename = "custom_wind-speed_150m.xyz"
# Load data
D = data_2D(datapath, filename)
Map = reshape(D.avgSpeed, length(unique(D.x)), length(unique(D.y)))
Map_150 = deepcopy(Map)

# julia> extrema(coords[1,:])
# (-121.80967731781246, -121.55717731781245)

# julia> extrema(coords[2,:])
# (37.63701874773503, 37.87451874773504)


# loc = coords[1:2,[4,70,800]]
# predict_f()


filename = "custom_wind-speed_150m.xyz"
# Load data
D = data_2D(datapath, filename)
Map = reshape(D.avgSpeed, length(unique(D.x)), length(unique(D.y)))
Map_150 = deepcopy(Map)


filename = "custom_wind-speed_200m.xyz"
# Load data
D = data_2D(datapath, filename)
Map = reshape(D.avgSpeed, length(unique(D.x)), length(unique(D.y)))
Map_200 = deepcopy(Map)


# Test my Gaussian Processes module.
include("custom_gprocess.jl")

l = exp(1)
σs = exp(2)
σn = 0
gp_mean = ConstantMean(x->0)
kernel_xy = SquaredExponentialKernel(l,σs)


# nx,ny = size(Map_100)
nx = 5
ny = 5

X = []
Y = []
append!(X,[[i,j,100] for i in 1.0:nx for j in 1.0:ny])
append!(Y,Array(reshape(Map_100'[1:Int(nx),1:Int(ny)], Int(nx*ny),1)))
# append!(X,[[i,j,200] for i in 1.0:nx for j in 1.0:ny])
# append!(Y,Array(reshape(Map_200'[1:Int(nx),1:Int(ny)], Int(nx*ny),1)))

d = 0.0
zₒ = 0.05
fₓ = z -> average(Y)   # you will get NaN in K2 if you set this to zero.
kernel_z = WindLogLawKernel(gp_mean,d,zₒ,fₓ)

X_star = [[i,j,150] for i in 1.0:nx for j in 1.0:ny]

gp_kernel = CompositeWindKernel(kernel_xy,kernel_z)
gp = GaussianProcess(X,Y,gp_mean,gp_kernel,0.0)
gp_dist = predictPosterior(X_star,gp)


ground_truth = Map_150[1:Int(nx),1:Int(ny)]
Y_star = Array(reshape(gp_dist.μ, Int(nx), Int(ny))')