using AA228FinalProject
using POMDPs
using ParticleFilters
using JLD
using Statistics
using LinearAlgebra
using BasicPOMCP
using DiscreteValueIteration

include("policy_qmdp.jl")

function POMDPs.action(p::AlphaVectorPolicy, s::RoombaState)
    P, Γ = p.discrete_P, p.Γ
    a_idx = argmax(Γ[:, stateindex(P, s)])
    return P.mdp.aspace[a_idx]
end

function discrete_pomdp()
    sensor = Lidar() # or Bumper() for the bumper version of the environment
    config = 3 # 1,2, or 3
    m = RoombaPOMDP(sensor=sensor, mdp=RoombaMDP(config=config));
    m_disc = DiscreteLidarRoombaPOMDP(m, 25, 25, 10, 5.0, 0.5, 1:0.5:80);
    m = RoombaPOMDP(sensor=sensor, mdp=RoombaMDP(config=config, aspace=actions(m_disc))); # BasicPOMCP only supports discrete actions
    return m_disc, m
end

struct DiscreteValueIterationPolicy <: Policy
    policy::ValueIterationPolicy
    discrete_P::RoombaPOMDP
end

function POMDPs.action(p::DiscreteValueIterationPolicy, s::RoombaState)
    println("Fuck")
    P = p.discrete_P
    sidx = stateindex(P, s)
    aidx = p.policy.policy[sidx]
    return p.policy.action_map[aidx]
    # return action(p.policy, discrete_s)
end

function estimate_value_mdp(p::DiscreteValueIterationPolicy)
    P = p.discrete_P
    function v(pomdp::RoombaPOMDP, s::RoombaState,  h::BeliefNode, steps)
        discrete_s = states(P)[stateindex(P, s)] 
        value(p.policy, discrete_s)
    end
    return v
end

# Computing and saving value iteration policy of the underlying MDP
# For value estimation in MCTS
function value_iteration()
    m_disc, _ = discrete_pomdp()
    solver = ValueIterationSolver(max_iterations=100, belres=1e-6, verbose=true)
    policy = DiscreteValueIteration.solve(solver, m_disc.mdp)
    policy = DiscreteValueIterationPolicy(policy, m_disc)
    save_path = "vi_mdp_discrete.jld"
    save(save_path, "policy", policy)
end


function get_mcts_policy()
    _, m = discrete_pomdp()
    save_path = "qmdp_discrete_3.jld"
    p_qmdp = load_policy(save_path)

    solver = POMCPSolver(tree_queries=100, c=10)
    p_mcts = BasicPOMCP.solve(solver, m)

    solver = POMCPSolver(tree_queries=2000, c=1, estimate_value=FORollout(p_qmdp), default_action=p_qmdp)
    p_mcts_qmdp = BasicPOMCP.solve(solver, m);

    save_path = "vi_mdp_discrete.jld"
    p_mdp = load_policy(save_path)
    # solver = POMCPSolver(tree_queries=100, c=1, estimate_value=FORollout(p_mdp), default_action=p_mdp)
    solver = POMCPSolver(tree_queries=2000, c=10, estimate_value=estimate_value_mdp(p_mdp), default_action=p_qmdp)
    println(solver)
    # action(p_mdp, first(states(p_mdp.discrete_P)))

    p_mcts_mdp = BasicPOMCP.solve(solver, m);
    return p_mcts, p_mcts_qmdp, p_mcts_mdp
end