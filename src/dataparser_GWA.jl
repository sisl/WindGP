# Parses the data provided in ../data/DWA/

using DelimitedFiles
include("./utils/misc.jl")


struct data_2D
    x:: Array{Float64}
    y:: Array{Float64}
    avgSpeed:: Array{Float64}

    function data_2D(datapath, filename)
        data_read = DelimitedFiles.readdlm(datapath*filename, ' ', Float64)
        
        x = data_read[:,1]
        y = data_read[:,2]
        avgSpeed = data_read[:,3]

        new(x,y,avgSpeed)
    end
end

function get_3D_data(farm; heights = [10, 50, 100, 150, 200])
    
    filename_t(h) = "custom_wind-speed_$(h)m.xyz"
    datapath_t(loc) = "../data/GWA/$(loc)/"
    datapath = datapath_t(farm)

    data_3D = Dict{Int, Array{Float64,2}}()

    for h in heights
        filename = filename_t(h)
        D = data_2D(datapath, filename)
        h_data = reshape(D.avgSpeed, length(unique(D.x)), length(unique(D.y)))
        
        push!(data_3D, h=>h_data)
    end

    return data_3D
end
