function marginalLogLikelihood(X, Y, σn_gp, gp_mean::Mean, gp_kernel)
    # We'd like to maximize the return of this function (Notice the negative signs).
    
    n = length(Y)
    Σ = σn_gp .* eye(n)

    m_dict = gp_mean.M
    m(x) = m_dict[x]
    M_X = [m(x) for x in X]
    
    K_X = getKernelMatrix(X,X,gp_kernel)
    K_X += 1e-6 .* eye(length(X))

    modelComplexityTerm = -0.5 * log(det(K_X + Σ))
    dataFitTerm = -0.5 * ((Y - M_X)' * inv(K_X + Σ) * (Y - M_X))
    constantTerm = -0.5 * n * log(2*pi)

    return modelComplexityTerm + dataFitTerm + constantTerm
end

function objFunctionValue(X, Y, σn_gp, gp_mean_val, l_sq, σs_sq, l_lin, σs_lin, d, zₒ, fₓ_val)

    gp_mean = CustomMean(DefaultDict(gp_mean_val))
    fₓ = DefaultDict(fₓ_val)
    gp_kernel = CustomTripleKernel(l_sq, σs_sq, l_lin, σs_lin, d, zₒ, fₓ, gp_mean)

    return marginalLogLikelihood(X, Y, σn_gp, gp_mean, gp_kernel)
end
