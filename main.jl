using AA228FinalProject
using POMDPs
using POMDPPolicies
using BeliefUpdaters
using Statistics
using ParticleFilters
using POMDPSimulators
using Cairo
using Gtk
using Random
using Printf

include("policy_baseline.jl")
include("policy_qmdp.jl")
include("policy_mcts.jl")

sensor = Lidar()
config = 3
m = RoombaPOMDP(sensor=sensor, mdp=RoombaMDP(config=config))

num_particles = 5000

resampler = LidarResampler(num_particles, LowVarianceResampler(num_particles))
# for the bumper environment
# resampler = BumperResampler(num_particles)

spf = BasicParticleFilter(m, resampler, num_particles)
# spf = SIRParticleFilter(m, num_particles)

v_noise_coefficient = 2.0
om_noise_coefficient = 0.5

belief_updater = RoombaParticleFilter(spf, v_noise_coefficient, om_noise_coefficient)

total_rewards = []

save_path = "qmdp_discrete_3.jld"
p_qmdp = load_policy(save_path)

p_mcts, p_mcts_qmdp, p_mcts_mdp = get_mcts_policy()

# p = p_to_end
# p = p_qmdp
p = p_mcts_mdp

for exp = 1:50
    println(string(exp))

    Random.seed!(exp)

    traj_rewards = sum([step.r for step in stepthrough(m,p,belief_updater, max_steps=100)])
    println(traj_rewards)
    push!(total_rewards, traj_rewards)
end

# Baseline: -1.541
# QMDP:2.011
# QMDP + MCTS, c=10, iter=2000: mean 2.732, stderr 6.397732
# MDP value estimation + MCTS, c=10, iter=2000: mean 2.990 stderr 2.998
# in 50 trials
@printf("Mean Total Reward: %.3f, StdErr Total Reward: %.3f\n", mean(total_rewards), std(total_rewards)/sqrt(5))
