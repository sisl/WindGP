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

include("../src/dataparser_GWA.jl")
using DataStructures

farm = "AltamontCA"
Map = get_3D_data(farm, heights=[100,150,200])

Map_100, Map_150, Map_200 = Map[100], Map[150], Map[200]

l = exp(1)
σs = exp(2)
σn = 0
gp_mean = CustomMean(DefaultDict(0.0))


# nx,ny = size(Map_100)
nx = 10
ny = 10

X = []
Y = []
# append!(X,[[i,j,100] for i in 1.0:nx for j in 1.0:ny])
# append!(Y,Array(reshape(Map_100[1:Int(nx),1:Int(ny)], Int(nx*ny),1)))
# append!(X,[[i,j,200] for i in 1.0:nx for j in 1.0:ny])
# append!(Y,Array(reshape(Map_200[1:Int(nx),1:Int(ny)], Int(nx*ny),1)))
append!(X,[[i,j,150] for i in 1.0:nx for j in 1.0:ny])
append!(Y,Array(reshape(Map_150'[1:Int(nx),1:Int(ny)], Int(nx*ny),1)))

d = 0.0
zₒ = 0.05
fₓ = DefaultDict(average(Y))            # you will get NaN in K2 if you set this to zero.

X_star = []
append!(X_star, [[i,j,150] for i in 1.0:nx for j in 1.0:ny])
# append!(X_star, [[i,j,170] for i in 0.5:nx+0.5 for j in 0.5:ny+0.5])


kernel_xy = Kernel[SquaredExponentialKernel(l,σs)]
kernel_z = Kernel[LinearExponentialKernel(10000.0, 1.0), WindLogLawKernel(gp_mean,d,zₒ,fₓ)]
gp_kernel = CompositeWindKernel(kernel_xy,kernel_z)


# Create the final GP.
gp_kernel = CustomTripleKernel(exp(1),exp(2),10000.0,1.0,gp_mean,d,zₒ,fₓ)
gp = GaussianProcess(X,Y,gp_mean,gp_kernel,0.0)

# Predict from the final GP.
# gp_dist = predictPosterior(X_star,gp)
gp_dist = predictPosteriorFaster(X_star,gp)

Y_star = Array(reshape(gp_dist.μ, Int(nx), Int(ny))')
# Y_star = Array(reshape(μ_star, Int(nx), Int(ny))')
ground_truth = Map_150[1:Int(nx),1:Int(ny)]

# Calculate error percentage.
ErrorPct = (Y_star - ground_truth) ./ ground_truth *100
AvgErrorPct = average(abs.(ErrorPct))