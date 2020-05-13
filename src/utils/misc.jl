# Some helper functions. Some simple functions are re-defined here to avoid dependencies to other modules.

average(a::AbstractArray) = sum(a::AbstractArray)/length(a::AbstractArray)

eye(x::Int) = [a==b ? 1.0 : 0.0 for a in 1:x, b in 1:x]   # create identity matrix (no dependencies).

function makeHermitian!(A; inflation=1e-6)
    A[:,:] = 0.5 .* (A + A')                        # average with transpose.
    A[:,:] = A[:,:] + inflation .* eye(size(A,1))   # prevent singularity.
end    

function dropBelowThreshold!(A; threshold=eps(Float64))
    for idx in eachindex(A)
        if abs(A[idx]) < threshold
           A[idx] = 0
        end
    end 
end

function transform4GPjl(X)
    X_gp = Array{Float64,2}(undef, 3, length(X))
    for idx in 1:size(X,1)
        X_gp[:,idx] = X[idx]
    end
    return X_gp
end

function exactLogLaw(z_star::Number, z::Number, zₒ, d)
    return log((z_star-d)/zₒ) / log((z-d)/zₒ)
end

function euclidean_dist(a::CartesianIndex,b::CartesianIndex)
    diff = abs(a-b).I
    return sqrt(diff[1]^2 + diff[2]^2)
end

function dot_product(a::Union{Number,AbstractArray}, b::Union{Number,AbstractArray})    # Supports numbers, vectors and matrices.
    @assert size(a) == size(b) "Dimensions of the two inputs do not match."
    c = 0
    for i in eachindex(a)
        c += a[i]*b[i]
    end
    return c
end

function normalizeArray(a::AbstractArray)  # normalizes any-size Array.
    norma = maximum(a)
    b = a./norma
    return b
end

function normalizeArray!(a::AbstractArray)  # normalizes any-size Array in-place.
    norma = maximum(a)
    for i = firstindex(a):lastindex(a)
        a[i] /= norma
    end
end

function stretch(a::AbstractArray, lower_lim::Real, upper_lim::Real)  # stretches (forces given extremas, interpolates others in-between) any-size Array.
    mini, maxi = minimum(a), maximum(a)
    norma = (maxi-mini)/(upper_lim-lower_lim)

    b = (a .- mini) ./ norma .+ lower_lim
    return b
end

function stretch!(a::AbstractArray, lower_lim::Real, upper_lim::Real)  # stretches (forces given extremas, interpolates others in-between) any-size Array in-place.
    mini, maxi = minimum(a), maximum(a)
    norma = (maxi-mini)/(upper_lim-lower_lim)

    b = (a .- mini) ./ norma .+ lower_lim

    for i = firstindex(a):lastindex(a)
        a[i] = b[i]
    end
end

function deg2NSEW(x_deg,y_deg)
    # Converts degress to direction, degrees, minutes, seconds.
    # E.g. +43.040958∘, -77.241711∘  ->  N 43∘2’27.45”, W 77◦14’30.16”.
    x_NSEW = Dict()
    y_NSEW = Dict()

    if x_deg>=0
        push!(x_NSEW, (:dir=>'N'))
    else
        push!(x_NSEW, (:dir=>'S'))
    end
    
    if y_deg>=0
        push!(y_NSEW, (:dir=>'E'))
    else
        push!(y_NSEW, (:dir=>'W'))
    end

    function getNSEW(val)
        val = abs(val)
        deg = floor(val)
        min = floor(abs((val - deg) * 60))
        sec = (abs(val - deg) * 60 - min) * 60
        return Dict(:deg=>deg, :min=>min, :sec=>sec)
    end

    merge!(x_NSEW, getNSEW(x_deg))
    merge!(y_NSEW, getNSEW(y_deg))

    return x_NSEW, y_NSEW
end

function savefig_recursive(plt_obj, filename, reset; dir="Figures")
    
    # If the filename exists, add a number to it and then save. Prevents overwrite.
    # filename: Name of file. Don't put filetype (e.g. "png") to the end of filename! Defaults to ".png".
    # reset: Delete everything in folder before saving? Set to true or false.
    
    if reset
        run(`rm -rf ./$dir`)
        mkdir("$dir")  # recreate the folder.
        filedir = dir*"/"*filename
        savefig(plt_obj, filedir)
    else
        dir in readdir() ? nothing : mkdir("$dir")  # if the folder exists, do nothing, otherwise, create it.
        i = 1
        itemname = filename
        while true in [occursin(itemname, item) for item in readdir(dir)]
            itemname = filename*string(i)
            i += 1
        end
        filedir = dir*"/"*itemname
        savefig(plt_obj, filedir)
    end

    return nothing
end

