using AA228FinalProject
using POMDPs
using POMDPPolicies
using BeliefUpdaters
using Statistics
using ParticleFilters
using POMDPSimulators
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

p_mcts, p_mcts_qmdp, p_mcts_mdp = get_mcts_policy(config)

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


@printf("Mean Total Reward: %.3f, StdErr Total Reward: %.3f\n", mean(total_rewards), std(total_rewards)/sqrt(length(total_rewards)))
# Config 3, 50 trials
# QMDP + MCTS, c=10, iter=2000: mean 2.732, stderr 0.904 
# MDP value estimation + MCTS, c=10, iter=2000: mean 2.990 stderr 0.948

# Config 1
# ToEnd Mean Total Reward: -2.045, StdErr Total Reward: 3.870
# QMDP Mean Total Reward: 2.389, StdErr Total Reward: 3.194

# Config 2
# ToEnd Mean Total Reward: -6.004, StdErr Total Reward: 3.811
# QMDP Mean Total Reward: 1.054, StdErr Total Reward: 3.858

# Config 3
# ToEnd Mean Total Reward: -1.879, StdErr Total Reward: 3.877
# QMDP Mean Total Reward: 1.393, StdErr Total Reward: 3.690

