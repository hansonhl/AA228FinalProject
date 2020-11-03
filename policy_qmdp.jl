using AA228FinalProject
using POMDPs
using ParticleFilters
using JLD
using Statistics
using LinearAlgebra


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

function init_gamma(P::RoombaPOMDP)
    n_a = length(actions(P))
    n_s = length(states(P))
    println("Constructed Γ vector with $(n_a) actions and $(n_s) states")
    return zeros(Float64, n_a, n_s)
end

struct QMDP
    k_max::Int64 # max num of iterations
end


# algorithm 21.2 in textbook
function update(P::RoombaPOMDP, M::QMDP, Γ::Array{Float64, 2})
    S = states(P)
    A = actions(P)
    γ = discount(P)

    # Transitions are deterministic, directly get the next state
    s′(s, a) = support(transition(P, s, a))[1]
    s′_idx(s, a) = stateindex(P, s′(s, a))
    R(s, a) = reward(P, s, a, s′(s, a))
    U(Γ′, s_idx) = maximum(Γ′[:,s_idx])

    🔥Γ🔥 = [R(s, a) + γ * U(Γ, s′_idx(s, a)) for a in A, s in S]
    return 🔥Γ🔥
end

function evaluate_utility(P::RoombaPOMDP, Γ::Array{Float64, 2})
    S = states(P)
    U(Γ′, s_idx) = maximum(Γ′[:,s_idx])
    return mean(U(Γ, stateindex(P, s)) for s in S)
end


function alphavector_iteration(P::RoombaPOMDP, M::QMDP, Γ::Array{Float64, 2})
    for k in 1:M.k_max
        println("Iteration $(k)")
        Γ = update(P, M, Γ)
        mean_util = evaluate_utility(P, Γ)
        println("Got average utility $(mean_util)")
    end
    return Γ
end


function solve(M::QMDP, P::RoombaPOMDP)
    💩Γ💩 = init_gamma(P)
    res = @timed alphavector_iteration(P, M, 💩Γ💩)
    🔥Γ🔥 = res[1]
    avg_time_per_iteration = res[2] / M.k_max
    println("Average time per iteration: $(avg_time_per_iteration) s")
    return AlphaVectorPolicy(P, 🔥Γ🔥)
end



struct AlphaVectorPolicy <: Policy
    discrete_P::RoombaPOMDP
    Γ::Array{Float64, 2}
end


function discretize_belief(P::RoombaPOMDP, cont_b::ParticleCollection{RoombaState})
    discr_b = zeros(Float64, length(states(P)))
    for (cont_s, w) in weighted_particles(cont_b)
        discr_b[stateindex(P, cont_s)] += w
    end
    return normalize(discr_b, 1)
end


# similar to algorithm 20.4 in textbook
function POMDPs.action(p::AlphaVectorPolicy, b::ParticleCollection{RoombaState})
    P, Γ = p.discrete_P, p.Γ
    b = discretize_belief(P, b)
    a_idx = argmax(Γ * b)
    return P.mdp.aspace[a_idx]
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
    config = 3 # 1,2, or 3
    cont_m = RoombaPOMDP(sensor=sensor, mdp=RoombaMDP(config=config))

    P = DiscreteLidarRoombaPOMDP(cont_m, 25, 25, 10, 5.0, 0.5, 1:0.5:80)
    M = QMDP(30)

    println("Starting to solve")
    👮y = solve(M, P)
    println("Finished solving, got policy")
    save_path = "qmdp_discrete_1.jld"

    save_policy(save_path, 👮y)
    # 👮y = load_policy(save_path)
end

if abspath(PROGRAM_FILE) == @__FILE__ # equivalent to if __name__ == "__main__"
    println("Running main")
    main()
end
