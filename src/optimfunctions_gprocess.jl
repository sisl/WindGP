function marginalLogLikelihood(X, Y, σn_gp, gp_mean::Mean, gp_kernel)
    # Normally, we'd like to maximize the variable `margLogLhood`.
    # However, notice we are returning the negative of it for Optim.jl.
    # Therefore, we are trying to minimize the output of this function.
    
    n = length(Y)
    Σ = σn_gp .* eye(n)

    m_dict = gp_mean.M
    m(x) = m_dict[x]
    M_X = [m(x) for x in X]
    
    K_X = getKernelMatrix(X,X,gp_kernel)
    K_X += 1e-6 .* eye(length(X))

    detK_X = det(K_X + Σ)
    if detK_X <= 0 return Inf end   # unacceptable solution.
    
    invK_X = inv(K_X + Σ)

    modelComplexityTerm = -0.5 * log(detK_X)
    if isequal(modelComplexityTerm, Inf) return Inf end   # unacceptable solution.
    
    dataFitTerm = -0.5 * ((Y - M_X)' * invK_X * (Y - M_X))
    if dataFitTerm > 0 return Inf end   # unacceptable solution.
    
    constantTerm = -0.5 * n * log(2*pi)
    
    margLogLhood = modelComplexityTerm + dataFitTerm + constantTerm

    @show detK_X
    @show modelComplexityTerm
    @show dataFitTerm
    @show constantTerm
    @show margLogLhood
    println("")

    return -margLogLhood
end

function objFunctionValue(opt_init)

    """ Notice that X,Y,d are retrieved from the main scope """
    global X, Y, d

    σn_gp, gp_mean_val, l_sq, σs_sq, l_lin, σs_lin, zₒ, fₓ_val = opt_init

    if gp_mean_val > 10.0 || zₒ > 1.0 || fₓ_val > 10.0 return Inf end   # unacceptable solution.

    gp_mean = CustomMean(DefaultDict(gp_mean_val))
    fₓ = DefaultDict(fₓ_val)
    gp_kernel = CustomTripleKernel(l_sq, σs_sq, l_lin, σs_lin, d, zₒ, fₓ, gp_mean)

    return marginalLogLikelihood(X, Y, σn_gp, gp_mean, gp_kernel)
end

function splitFarm(Map, nx_step, ny_step, altitudes)
    nx_max, ny_max = size(Map[10])

    MapPiecesX = []
    MapPiecesY = []

    for nx in  0 : nx_step : nx_max - nx_step
        for ny in  0 : ny_step : ny_max - ny_step

            SemiPiecesX = []
            SemiPiecesY = []

            for h in altitudes
                append!(SemiPiecesX, [[j, i, Float64(h)] for i in nx*grid_dist:grid_dist:(nx+nx_step-1)*grid_dist for j in ny*grid_dist:grid_dist:(ny+ny_step-1)*grid_dist])
                append!(SemiPiecesY, vec(Map[h][nx+1:nx+nx_step, ny+1:ny+ny_step]))
            end

            push!(MapPiecesX, SemiPiecesX)
            push!(MapPiecesY, SemiPiecesY)

        end
    end
    return (MapPiecesX, MapPiecesY)
end

function kFoldTest(opt_final, idx_to_exclude)
    kFoldVal = 0

    for (idx,val) in enumerate(MapPiecesX)
        global X = MapPiecesX[idx]
        global Y = MapPiecesY[idx]
        idx != idx_to_exclude ? kFoldVal += objFunctionValue(opt_final) : nothing
    end

    return kFoldVal/length(MapPiecesY)
end

### MANUAL DEBUG ###

# gp_mean = CustomMean(DefaultDict(gp_mean_val))
# fₓ = DefaultDict(fₓ_val)
# ## gp_kernel = CustomTripleKernel(l_sq, σs_sq, l_lin, σs_lin, d, zₒ, fₓ, gp_mean)

# kernel_xy = Kernel[SquaredExponentialKernel(l_sq, σs_sq)]
# kernel_z = Kernel[LinearExponentialKernel(l_lin, σs_lin), WindLogLawKernel(gp_mean,d,zₒ,fₓ)]
# kernel_xyz = Kernel[]
# gp_kernel = CompositeWindKernel(kernel_xy, kernel_z, kernel_xyz)

# n = length(Y)
# Σ = σn_gp .* eye(n)

# m_dict = gp_mean.M
# m(x) = m_dict[x]
# M_X = [m(x) for x in X]

# ## K_X = getKernelMatrix(X,X,gp_kernel)
# ## K_X += 1e-6 .* eye(length(X))

# kernel = gp_kernel                                      # composite kernel.
# k(x_star, x) = getKernelValue(x_star, x, kernel)        # function of kernel.

# K_X = [k(x1,x2) for x1 in X, x2 in X]
# K_X += 1e-6 .* eye(length(X))


# @show l_sq
# @show det(K_X)
# @show modelComplexityTerm = -0.5 * log(det(K_X + Σ))
# @show dataFitTerm = -0.5 * ((Y - M_X)' * inv(K_X + Σ) * (Y - M_X))
# @show constantTerm = -0.5 * n * log(2*pi)

# modelComplexityTerm + dataFitTerm + constantTerm