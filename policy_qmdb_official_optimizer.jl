import Pkg
if !haskey(Pkg.installed(), "AA228FinalProject")
    jenv = joinpath(dirname(@__FILE__()), ".") # this assumes the notebook is in the same dir
    # as the Project.toml file, which should be in top level dir of the project. 
    # Change accordingly if this is not the case.
    Pkg.activate(jenv)
end
using AA228FinalProject
using POMDPs
using ParticleFilters
using JLD
using Statistics
using LinearAlgebra
using QMDP
using FIB

# P = DiscreteLidarRoombaPOMDP(cont_m, 50, 50, 20, 5.0, 0.5, 1:0.5:80)
function DiscreteLidarRoombaPOMDP(cont_m::RoombaPOMDP, num_x_pts::Int64, num_y_pts::Int64, num_th_pts::Int64, v_step::Float64, om_step::Float64, cutoff_range)
    m = cont_m.mdp
    config = m.config
    vlist = collect(0.0:v_step:m.v_max)
    omlist = collect(-m.om_max:om_step:m.om_max)

    aspace = vec(collect(RoombaAct(v, om) for v in vlist, om in omlist))
    sspace = DiscreteRoombaStateSpace(num_x_pts, num_y_pts, num_th_pts)

    cut_points = collect(cutoff_range)
    discrete_sensor = DiscreteLidar(cut_points)

    return RoombaPOMDP(sensor=discrete_sensor, mdp=RoombaMDP(config=config, sspace=sspace, aspace=aspace))
end

struct AlphaVectorPolicy <: Policy
    discrete_P::RoombaPOMDP
    Γ::Array{Float64, 2}
end

function save_policy(save_path::String, p::AlphaVectorPolicy)
    println("Saving policy to $(save_path)")
    save(save_path, "policy", p)
end

function load_policy(save_path::String)
    println("Loading policy from $(save_path)")
    return load(save_path, "policy")
end

function main()
    sensor = Lidar() # or Bumper() for the bumper version of the environment
    config = parse(Int64, ARGS[1])
    cont_m = RoombaPOMDP(sensor=sensor, mdp=RoombaMDP(config=config))

    P = DiscreteLidarRoombaPOMDP(cont_m, 25, 25, 10, 5.0, 0.5, 1:5:80)
    
    solver = QMDPSolver(max_iterations=30, belres=1e-3)
    println("Starting to solve")
    policy = solve(solver, P) # async as shown in https://github.com/JuliaPOMDP/DiscreteValueIteration.jl/blob/master/src/vanilla.jl#L132
    println("Finished solving, got policy")

    alphas = hcat(policy.alphas...)
    save_path = "qmdb_discrete_$(config)_official.jld"
    save_policy(save_path, AlphaVectorPolicy(P, alphas))
end

if abspath(PROGRAM_FILE) == @__FILE__ # equivalent to if __name__ == "__main__"
    println("Running main")
    main()
end
