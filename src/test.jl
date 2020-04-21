mean = ConstantMean(x->0)
kernel = SquaredExponentialKernel(1.0,1.0)
X = collect(0:5)
Y = [2, 1, 2, 3, 2, 3]

gp = GaussianProcess(X,Y,mean,kernel,0.0)

X_star = collect(0.5:1.0:4.5)
# X_star = [1,4]

###

X = [[i,j] for i in 1.0:4.0 for j in 1.0:4.0]
Y = [7,9,8,6,9,8,10,11,10,11,12,13,11,14,13,15]

X_star = [[i,j] for i in 1.5:1.0:3.5 for j in 1.5:1.0:3.5]
append!(X_star,X)
kernel = SquaredExponentialKernel(2.0,1.0)
gp = GaussianProcess(X,Y,mean,kernel,0.0)
gp_dist = predictPosterior(X_star,gp)