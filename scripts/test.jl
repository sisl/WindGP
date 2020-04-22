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

X_star = [[i,j] for i in 1.5:1.0:3.5 for j in 1.5:1.0:3.5]
# append!(X_star,X)

l = exp(1)
σs = exp(2)
σn = 0

gp_mean = ConstantMean(x->0)
gp_kernel = SquaredExponentialKernel(l,σs)
gp = GaussianProcess(X,Y,gp_mean,gp_kernel,σn)
gp_dist = predictPosterior(X_star,gp)


### WindLogLawKernel

d = 0.0
zₒ = 0.05
fₓ = z->z^2/10

gp_mean = ConstantMean(x->0)
gp_kernel = WindLogLawKernel(gp_mean,d,zₒ,fₓ)

X = [5,7,9]
Y = [2.5,4.9,8.1]

X_star = [5,6,7,8,9,10,11,12,13]

gp = GaussianProcess(X,Y,gp_mean,gp_kernel,0.0)
gp_dist = predictPosterior(X_star,gp)


### CompositeWindKernel test1: Only z is different.

l = exp(1)
σs = exp(2)
σn = 0
gp_mean = ConstantMean(x->0)
kernel_xy = SquaredExponentialKernel(l,σs)

d = 0.0
zₒ = 0.05
fₓ = z -> average(Y)
kernel_z = WindLogLawKernel(gp_mean,d,zₒ,fₓ)

# X = [[i,j,5] for i in 1.0:4.0 for j in 1.0:4.0]
# Y = [7,9,8,6,9,8,10,11,10,11,12,13,11,14,13,15]
X = [[1.0,1.0,5.0],[1.0,2.0,5.0],[1.0,3.0,5.0]]
Y = [7,9,8]
X_star = [[1.0,1.0,4.0],[1.0,2.0,4.0],[1.0,3.0,4.0],[1.0,1.0,6.0],[1.0,2.0,6.0],[1.0,3.0,6.0]]
# Y_star should have mean: [6.66, 8.56, 7.61, 7.27, 9.35, 8.31] according to wind log law.

gp_kernel = CompositeWindKernel(kernel_xy,kernel_z)
gp = GaussianProcess(X,Y,gp_mean,gp_kernel,0.0)
gp_dist = predictPosterior(X_star,gp)


### CompositeWindKernel test2

l = exp(1)
σs = exp(2)
σn = 0
gp_mean = ConstantMean(x->0)
kernel_xy = SquaredExponentialKernel(l,σs)

d = 0.0
zₒ = 0.05
fₓ = z -> average(Y)
kernel_z = WindLogLawKernel(gp_mean,d,zₒ,fₓ)

X = [[i,j,5] for i in 1.0:4.0 for j in 1.0:4.0]
Y = [7,9,8,6,9,8,10,11,10,11,12,13,11,14,13,15]
X_star = [[i,j,5] for i in 1.5:1.0:3.5 for j in 1.5:1.0:3.5]

gp_kernel = CompositeWindKernel(kernel_xy,kernel_z)
gp = GaussianProcess(X,Y,gp_mean,gp_kernel,0.0)
gp_dist = predictPosterior(X_star,gp)


