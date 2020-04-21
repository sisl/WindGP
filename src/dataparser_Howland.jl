# Parses the CFD data provided in ../data/Howland1/

using CSV
using DataFrames

######## Parameters ########

data_path = "../data/Howland1"
N = (Nx=962, Ny=450, Nz=384)

############################

struct data
    x
    y
    z
    angle
    speed
    function data(data_path, N)
        
        x = CSV.read("$data_path/domain/x.csv", delim=',', header=false)
        y = CSV.read("$data_path/domain/y.csv", delim=',', header=false)
        z = CSV.read("$data_path/domain/z.csv", delim=',', header=false)

        angle = Array{Float64,3}(undef, N.Nx, N.Ny, N.Nz)
        speed = Array{Float64,3}(undef, N.Nx, N.Ny, N.Nz)
        for idx in range(1, N.Nz)
            println("## Loading data: $idx")
            file_idx = lpad(idx,3,'0')  # puts extra zeros in front when required

            angle_idx = CSV.read("$data_path/angle/$file_idx.csv", header=false, delim=',', missingstring="")
            angle_idx = DataFrame(colwise(col -> recode(col, missing=>0.0), angle_idx), names(angle_idx))       # there is literally no built-in function to replace missing values to something else.
            angle[:,:,idx] = Matrix(angle_idx)

            speed_idx = CSV.read("$data_path/speed/$file_idx.csv", header=false, delim=',', missingstring="")
            speed_idx = DataFrame(colwise(col -> recode(col, missing=>1.0), speed_idx), names(speed_idx))
            speed[:,:,idx] = Matrix(speed_idx)
        end

        new(vec(Matrix(x)), vec(Matrix(y)), vec(Matrix(z)), angle, speed)
    end
end

D = data(data_path, N)


# Plotting the data

using Plots

function f(x,y,h)
    # Notice that D is pulled from the global workspace.

    # h=10

    x_idx = findfirst(e->e==x, D.x)
    y_idx = findfirst(e->e==y, D.y)
    z = Dict(:angle => D.angle[x_idx,y_idx,h], :speed => D.speed[x_idx,y_idx,h])
    
    # z_spd = z[:speed]
    
    return z
end



# p_speed = plot(D.x, D.y, f, st=:surface, camera = (45,45))
# p_speed = plot(D.x, D.y, f, st=:contour)

height_idx = 10

p_speed = plot(D.x, D.y, ((a,b)->f(a,b,height_idx)[:speed]), zlim=(0,1), st=:surface, camera = (45,45))
p_speed = plot(D.x, D.y, ((a,b)->f(a,b,height_idx)[:speed]), st=:contour)

p_angle = plot(D.x, D.y, ((a,b)->f(a,b,height_idx)[:angle]), st=:surface, camera = (45,45))
p_angle = plot(D.x, D.y, ((a,b)->f(a,b,height_idx)[:angle]), st=:contour)

contour(D.angle[:,:,9], title="Wind Angle at 100 m.", fill=true)

@gif for height_idx in range(1, 50)
    println("## Plotting data: $height_idx")
    h = D.z[height_idx]
    p_speed = plot(D.x, D.y, ((a,b)->f(a,b,height_idx)[:speed]),
                   zlim=(0.3, 1),
                   st=:surface,
                   camera = (45,45),
                #    xlabel="Length",
                #    ylabel="Width",
                #    zlabel="Wind Speed",
                   title="Normalized Wind Speed at Height $h",)
end
