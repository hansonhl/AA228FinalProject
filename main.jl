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

sensor = Lidar()
config = 3
m = RoombaPOMDP(sensor=sensor, mdp=RoombaMDP(config=config))

num_particles = 5000
resampler = LidarResampler(num_particles, LowVarianceResampler(num_particles))
# for the bumper environment
# resampler = BumperResampler(num_particles)

spf = SimpleParticleFilter(m, resampler)

v_noise_coefficient = 2.0
om_noise_coefficient = 0.5

belief_updater = RoombaParticleFilter(spf, v_noise_coefficient, om_noise_coefficient)

total_rewards_to_end = []
total_rewards_qmdp = []

save_path = "qmdp_discrete_1.jld"
p_qmdp = load_policy(save_path)

for exp = 1:75
    println(string(exp))

    Random.seed!(exp)

    p_to_end = ToEnd(0, get_goal_xy(m))
    traj_rewards_to_end = sum([step.r for step in stepthrough(m,p_to_end,belief_updater, max_steps=100)])
    traj_rewards_qmdp = sum([step.r for step in stepthrough(m,p_qmdp,belief_updater, max_steps=100)])

    push!(total_rewards_to_end, traj_rewards_to_end)
    push!(total_rewards_qmdp, traj_rewards_qmdp)
end

# -1.541
# 2.011
# in 50 trials
@printf("ToEnd Mean Total Reward: %.3f, StdErr Total Reward: %.3f\n", mean(total_rewards_to_end), std(total_rewards_to_end)/sqrt(5))
@printf("QMDP Mean Total Reward: %.3f, StdErr Total Reward: %.3f\n", mean(total_rewards_qmdp), std(total_rewards_qmdp)/sqrt(5))
