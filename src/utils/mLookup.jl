# Lookup mean function

"""
    MeanLookup <: GaussianProcesses.Mean
Lookup mean function
"""
mutable struct MeanLookup <: GaussianProcesses.Mean
    kdtree::NearestNeighbors.KDTree    # Lookup K-Dimensional Tree of X vals
    Y::AbstractVector{Float64}         # Lookup Y vals

    MeanLookup(X::AbstractMatrix{Float64}, Y::AbstractVector{Float64}) = new(NearestNeighbors.KDTree(X), Y)
end

function GaussianProcesses.mean(mLookup::MeanLookup, x::AbstractVector)
    nn, _ = knn(mLookup.kdtree, x, 1)
    return mLookup.Y[nn][1]
end

function GaussianProcesses.mean(mLookup::MeanLookup, x::AbstractMatrix)
    nn, _ = knn(mLookup.kdtree, x, 1)
    nn = [item for sublist in nn for item in sublist]    # extract list of lists
    return mLookup.Y[nn]
end

GaussianProcesses.get_params(mLookup::MeanLookup) = [mLookup.kdtree, mLookup.Y]
GaussianProcesses.get_param_names(::MeanLookup) = [:kdtree, :Y]
GaussianProcesses.num_params(mLookup::MeanLookup) = length(mLookup.Y)    # returns number of datapoints in Lookup

function GaussianProcesses.set_params!(mLookup::MeanLookup, hyp::AbstractVector)
    (isa(hyp[1], NearestNeighbors.KDTree) && isa(hyp[2], AbstractVector{Float64})) || throw(ArgumentError("Hyp params do not match MeanLookup format."))
    copyto!(mLookup.β, hyp)
end