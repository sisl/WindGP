include("../src/custom_gprocess.jl")

### SquaredExponentialKernel 1D

mean = ConstantMean(x->0)
kernel = SquaredExponentialKernel(1.0,1.0)
X = collect(0:5)
Y = [2, 1, 2, 3, 2, 3]

gp = GaussianProcess(X,Y,mean,kernel,0.0)

X_star = collect(0.5:1.0:4.5)
# X_star = [1,4]


### SquaredExponentialKernel 2D

X = [[i,j] for i in 1.0:4.0 for j in 1.0:4.0]
Y = [7,9,8,6,9,8,10,11,10,11,12,13,11,14,13,15]

# X_star = [[i,j] for i in 1.5:1.0:3.5 for j in 1.5:1.0:3.5]
X_star = []
push!(X_star,[5.0,5.0])
push!(X_star,[6.0,5.0])

l = exp(1)
σs = exp(2)
σn = 0.0

gp_mean = ConstantMean(x->0)
gp_kernel = SquaredExponentialKernel(l,σs)
gp = GaussianProcess(X,Y,gp_mean,gp_kernel,σn)
gp_dist = predictPosterior(X_star,gp)


### WindLogLawKernel test1

d = 0.0
zₒ = 0.05
fₓ = z -> z^2 / 10

gp_mean = ConstantMean(x->0)
gp_kernel = WindLogLawKernel(gp_mean,d,zₒ,fₓ)

X = [5,7,9]
Y = [2.5,4.9,8.1]

X_star = [5,6,7,8,9,10,11,12,13]

gp = GaussianProcess(X,Y,gp_mean,gp_kernel,0.0)
gp_dist = predictPosterior(X_star,gp)


### WindLogLawKernel test2

d = 0.0
zₒ = 0.05
fₓ = z -> z^2 / 10

gp_mean = ConstantMean(x->0)
gp_kernel = WindLogLawKernel(gp_mean,d,zₒ,fₓ)

X = [5.0, 4.0]
Y = [7.0, 6.66]

X_star = [6.0]

σn = 0.0
gp = GaussianProcess(X,Y,gp_mean,gp_kernel,σn)
gp_dist = predictPosterior(X_star,gp)

# Expected result:
exactLogLaw(X_star[1], X[1], zₒ, d) * Y


### WindLogLawKernel test3

d = 0.0
zₒ = 0.05
fₓ = z -> z^2 / 10

gp_mean = ConstantMean(x->0)
gp_kernel = WindLogLawKernel(gp_mean,d,zₒ,fₓ)

X = [3, 5, 6]
Y = [6.22, 7, 7.27]

X_star = [2.0, 7.0]
σn = 0.0
gp = GaussianProcess(X,Y,gp_mean,gp_kernel,σn)
gp_dist = predictPosterior(X_star,gp)

# # Expected result:
# exactLogLaw(X_star[1], X[1], zₒ, d) * Y


### CompositeWindKernel test1: Only z is different (single altitude to co-vary with).

l = exp(1)
σs = exp(1)
σn = 0
gp_mean = ConstantMean(x->0)
kernel_xy = SquaredExponentialKernel(l,σs)

d = 0.0
zₒ = 0.05
fₓ = z -> average(Y)
kernel_z = WindLogLawKernel(gp_mean,d,zₒ,fₓ)

X = [[1.1, 2.0, 4.1],[1.1, 1.0, 4.2],[1.1, 3.0, 3.9]]
Y = [9,8,15]
X_star = [[1.0,1.0,4.0],[1.0,2.0,4.0],[1.0,3.0,4.0],[1.0,1.0,6.0],[1.0,2.0,6.0],[1.0,3.0,6.0]]
# Y_star should have mean: [6.66, 8.56, 7.61, 7.27, 9.35, 8.31] according to wind log law.

gp_kernel = CompositeWindKernel(kernel_xy,kernel_z)
gp = GaussianProcess(X,Y,gp_mean,gp_kernel,0.0)
gp_dist = predictPosterior(X_star,gp)





### CompositeWindKernel test2: Only z is different (multiple altitudes to co-vary with).

X = [[1.0,1.0,3.0],[1.0,2.0,3.0],[1.0,3.0,3.0],[1.0,1.0,5.0],[1.0,2.0,5.0],[1.0,3.0,5.0]]
Y = [6.22,6.22,6.22,7,7,7]
X_star = [[1.0,1.0,4.0],[1.0,2.0,4.0],[1.0,3.0,4.0],[1.0,1.0,6.0],[1.0,2.0,6.0],[1.0,3.0,6.0]]

l = exp(1)
σs = exp(2)
σn = 0
gp_mean = ConstantMean(x->0)
kernel_xy = SquaredExponentialKernel(l,σs)

d = 0.0
zₒ = 0.05
fₓ = z -> average(Y)
kernel_z = WindLogLawKernel(gp_mean,d,zₒ,fₓ)

gp_kernel = CompositeWindKernel(kernel_xy,kernel_z)
gp = GaussianProcess(X,Y,gp_mean,gp_kernel,0.0)
gp_dist = predictPosterior(X_star,gp)



### CompositeWindKernel test2b (smaller): Only z is different (multiple altitudes to co-vary with).

X = [[1.0,1.0,50], [1.0,1.0,150]]
Y = [6, 7]
X_star = [[1.0,1.0,50], [1.0,1.0,100], [1.0,1.0,150]]
# Y_star should have mean: 6.66

l = exp(1)
σs = exp(2)
σn = 0
gp_mean = ConstantMean(x->0)
kernel_xy = SquaredExponentialKernel(l,σs)

d = 0.0
zₒ = 0.05
fₓ = z -> average(Y)
kernel_z = WindLogLawKernel(gp_mean,d,zₒ,fₓ)

gp_kernel = CompositeWindKernel(kernel_xy,kernel_z)
gp = GaussianProcess(X,Y,gp_mean,gp_kernel,0.0)
gp_dist = predictPosterior(X_star,gp)


### CompositeWindKernel test3

l = exp(1)
σs = exp(2)
σn = 0
gp_mean = ConstantMean(x->0)
kernel_xy = SquaredExponentialKernel(l,σs)

X = [[i,j,5] for i in 1.0:4.0 for j in 1.0:4.0]
Y = [7,9,8,6,9,8,10,11,10,11,12,13,11,14,13,15]
X_star = [[i,j,5] for i in 1.5:1.0:3.5 for j in 1.5:1.0:3.5]

d = 0.0
zₒ = 0.05
fₓ = z -> average(Y)   # you will get NaN in K2 if you set this to zero.
kernel_z = WindLogLawKernel(gp_mean,d,zₒ,fₓ)

gp_kernel = CompositeWindKernel(kernel_xy,kernel_z)
gp = GaussianProcess(X,Y,gp_mean,gp_kernel,0.0)
gp_dist = predictPosterior(X_star,gp)


### CompositeWindKernel test4

X = [[1.0, 1.0, 3.0], [1.0, 3.0, 3.0]]
Y = [7, 10]
X_star = [[1.0, 2.0, 2.0]]

l = exp(1)
σs = exp(2)
σn = 0
gp_mean = ConstantMean(x->0)
kernel_xy = SquaredExponentialKernel(l,σs)

d = 0.0
zₒ = 0.05
fₓ = z -> average(Y)
kernel_z = WindLogLawKernel(gp_mean,d,zₒ,fₓ)

gp_kernel = CompositeWindKernel(kernel_xy,kernel_z)
gp = GaussianProcess(X,Y,gp_mean,gp_kernel,0.0)
gp_dist = predictPosterior(X_star,gp)



### GWA tests.

include("../src/custom_gprocess.jl")
include("../src/dataparser_GWA.jl")
using DataStructures

farm = "AltamontCA"
grid_dist = 220

Map = get_3D_data(farm; altitudes=[10, 50, 100, 150, 200])
Map_150 = Map[150]

nx = 50
ny = 50

X = []
Y = []

for h in [10, 50, 100, 200]
    append!(X, [[j, i, Float64(h)] for i in 0.0:grid_dist:(nx-1)*grid_dist for j in 0.0:grid_dist:(ny-1)*grid_dist])
    # append!(X,[[i,j,Float64(h)] for i in 1.0:nx for j in 1.0:ny])
    append!(Y, vec(Map[h][1:nx,1:ny]))
end

X_star = []
append!(X_star, [[j, i, Float64(150)] for i in 0.0:grid_dist:(nx-1)*grid_dist for j in 0.0:grid_dist:(ny-1)*grid_dist])
# append!(X_star, [[i,j,150] for i in 1.0:nx for j in 1.0:ny])

X_gp = transform4GPjl(X)
Xs_gp = transform4GPjl(X_star)

## Create the final GP.
# GaussianProcess PARAMS
σn_gp = 0.0
gp_mean = CustomMean(DefaultDict(0.0))   # you will get division by zero if you set this equal to fₓ.

# SquaredExponentialKernel PARAMS
l_sq = exp(1) * grid_dist^2
σs_sq = exp(2)

# LinearExponentialKernel PARAMS
l_lin = 10000.0
σs_lin = 1.0

# WindLogLawKernel PARAMS
d = 0.0
zₒ = 0.05
fₓ = DefaultDict(average(Y))             # you will get division by zero if you set this to zero.

# kernel_xy = Kernel[SquaredExponentialKernel(l_sq,σs_sq)]
# kernel_z = Kernel[LinearExponentialKernel(l_lin,σs_lin), WindLogLawKernel(gp_mean,d,zₒ,fₓ)]
# kernel_xyz = Kernel[]
# gp_kernel = CompositeWindKernel(kernel_xy,kernel_z,kernel_xyz)


gp_kernel = CustomTripleKernel(l_sq, σs_sq, l_lin, σs_lin, d, zₒ, fₓ, gp_mean)
gp = GaussianProcess(X, Y, gp_mean, gp_kernel, σn_gp)

# Predict from the final GP.
# gp_dist = predictPosterior(X_star,gp)
gp_dist = predictPosteriorFaster(X_star,gp)

# Y_star = Array(reshape(μ_star, Int(nx), Int(ny)))
Y_star = Array(reshape(gp_dist.μ, Int(nx), Int(ny)))
ground_truth = Map_150[1:Int(nx),1:Int(ny)]

# Calculate error percentage.
ErrorPct = (Y_star - ground_truth) ./ ground_truth *100
AvgErrorPct = average(abs.(ErrorPct))