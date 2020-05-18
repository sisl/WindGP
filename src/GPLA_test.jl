# using Revise
# using BOMCTS
using GaussianProcesses

mean_0 = 0.0
nobs = 2000
x = rand(Float64, 3, nobs)
# y = rand(Float64, nobs)
y = sum(x, dims=1)[1,:]
y += rand(Float64, nobs)*0.1
gpla = GPLA(x, y, 10, 2, 1, MeanConst(mean_0), SEIso(0.0, 0.0), -2.0)
gpe = GPE(x, y, MeanConst(mean_0), SEIso(0.0, 0.0), -2.0)
println("MLL GPLA: $(gpla.mll)")
println("MLL GPE: $(gpe.mll)")


update_dmll!(gpla)
GaussianProcesses.update_target_and_dtarget!(gpe)
println("dMLL GPLA (Analytical): $(gpla.dmll)")
println("dMLL GPE (Analytical): $(gpe.dmll)")
function dmll_num(gp::GPLA)
    eps = 1e-6
    noise_0 = gp.logNoise.value
    noise_1 = noise_0 - eps
    noise_2 = noise_0 + eps
    gp.logNoise = GaussianProcesses.wrap_param(noise_2)
    BOMCTS.initialize!(gp)
    dmll_noise = gp.mll
    gp.logNoise = GaussianProcesses.wrap_param(noise_1)
    BOMCTS.initialize!(gp)
    dmll_noise -= gp.mll
    dmll_noise /= (2.0*eps)
    println("DMLL Noise: $dmll_noise")
    gp.logNoise = GaussianProcesses.wrap_param(noise_0)
    BOMCTS.initialize!(gp)
    mean_0 = GaussianProcesses.get_params(gp.mean)[1]
    mean_1 = mean_0 - eps
    mean_2 = mean_0 + eps
    gp.mean = GaussianProcesses.MeanConst(mean_2)
    BOMCTS.initialize!(gp)
    dmll_mean = gp.mll
    gp.mean = GaussianProcesses.MeanConst(mean_1)
    BOMCTS.initialize!(gp)
    dmll_mean -= gp.mll
    dmll_mean /= (2.0*eps)
    println("DMLL Mean: $dmll_mean")
    gp.mean = GaussianProcesses.MeanConst(mean_0)
    BOMCTS.initialize!(gp)
    ### Kernel l ###
    kernel_0 = GaussianProcesses.get_params(gp.kernel)[1]
    kernel_static = GaussianProcesses.get_params(gp.kernel)[2]
    kernel_1 = kernel_0 - eps
    kernel_2 = kernel_0 + eps
    gp.kernel = GaussianProcesses.SEIso(kernel_2, kernel_static)
    BOMCTS.initialize!(gp)
    dmll_kernel = gp.mll
    gp.kernel = GaussianProcesses.SEIso(kernel_1, kernel_static)
    BOMCTS.initialize!(gp)
    dmll_kernel -= gp.mll
    dmll_kernel /= (2.0*eps)
    println("DMLL Kernel 1: $dmll_kernel")
    gp.kernel = GaussianProcesses.SEIso(kernel_0, kernel_static)
    BOMCTS.initialize!(gp)
    #### Kernel Parameter sigma ####
    kernel_0 = GaussianProcesses.get_params(gp.kernel)[2]
    kernel_static = GaussianProcesses.get_params(gp.kernel)[1]
    kernel_1 = kernel_0 - eps
    kernel_2 = kernel_0 + eps
    gp.kernel = GaussianProcesses.SEIso(kernel_static, kernel_2)
    BOMCTS.initialize!(gp)
    dmll_kernel = gp.mll
    gp.kernel = GaussianProcesses.SEIso(kernel_static, kernel_1)
    BOMCTS.initialize!(gp)
    dmll_kernel -= gp.mll
    dmll_kernel /= (2.0*eps)
    println("DMLL Kernel 2: $dmll_kernel")
    gp.kernel = GaussianProcesses.SEIso(kernel_static, kernel_0)
    BOMCTS.initialize!(gp)
end
function dmll_num(gp::GPE)
    eps = 1e-6
    noise_0 = gp.logNoise.value
    noise_1 = noise_0 - eps
    noise_2 = noise_0 + eps
    gp.logNoise = GaussianProcesses.wrap_param(noise_2)
    GaussianProcesses.update_mll!(gp)
    dmll_noise = gp.mll
    gp.logNoise = GaussianProcesses.wrap_param(noise_1)
    GaussianProcesses.update_mll!(gp)
    dmll_noise -= gp.mll
    dmll_noise /= (2.0*eps)
    println(dmll_noise)
    gp.logNoise = GaussianProcesses.wrap_param(noise_0)
    GaussianProcesses.update_mll!(gp)
    mean_0 = GaussianProcesses.get_params(gp.mean)[1]
    mean_1 = mean_0 - eps
    mean_2 = mean_0 + eps
    gp.mean = GaussianProcesses.MeanConst(mean_2)
    GaussianProcesses.update_mll!(gp)
    dmll_mean = gp.mll
    gp.mean = GaussianProcesses.MeanConst(mean_1)
    GaussianProcesses.update_mll!(gp)
    dmll_mean -= gp.mll
    dmll_mean /= (2.0*eps)
    println(dmll_mean)
    gp.mean = GaussianProcesses.MeanConst(mean_0)
    GaussianProcesses.update_mll!(gp)
    ### Kernel l ###
    kernel_0 = GaussianProcesses.get_params(gp.kernel)[1]
    kernel_static = GaussianProcesses.get_params(gp.kernel)[2]
    kernel_1 = kernel_0 - eps
    kernel_2 = kernel_0 + eps
    gp.kernel = GaussianProcesses.SEIso(kernel_2, kernel_static)
    GaussianProcesses.update_mll!(gp)
    dmll_kernel = gp.mll
    gp.kernel = GaussianProcesses.SEIso(kernel_1, kernel_static)
    GaussianProcesses.update_mll!(gp)
    dmll_kernel -= gp.mll
    dmll_kernel /= (2.0*eps)
    println(dmll_kernel)
    gp.kernel = GaussianProcesses.SEIso(kernel_0, kernel_static)
    GaussianProcesses.update_mll!(gp)
    #### Kernel Parameter sigma ####
    kernel_0 = GaussianProcesses.get_params(gp.kernel)[2]
    kernel_static = GaussianProcesses.get_params(gp.kernel)[1]
    kernel_1 = kernel_0 - eps
    kernel_2 = kernel_0 + eps
    gp.kernel = GaussianProcesses.SEIso(kernel_static, kernel_2)
    GaussianProcesses.update_mll!(gp)
    dmll_kernel = gp.mll
    gp.kernel = GaussianProcesses.SEIso(kernel_static, kernel_1)
    GaussianProcesses.update_mll!(gp)
    dmll_kernel -= gp.mll
    dmll_kernel /= (2.0*eps)
    println(dmll_kernel)
    gp.kernel = GaussianProcesses.SEIso(kernel_static, kernel_0)
    GaussianProcesses.update_mll!(gp)
end
# dmll_num(gpla)
# dmll_num(gpe)
GaussianProcesses.optimize!(gpla)