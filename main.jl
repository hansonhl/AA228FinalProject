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
# include("policy_mcts.jl")


function main()
    config = parse(Int64, ARGS[1])
    num_trials = parse(Int64, ARGS[2])

    sensor = Lidar()
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

    total_rewards_to_end = []
    total_rewards_qmdp = []

    # p_mcts, p_mcts_qmdp, p_mcts_mdp = get_mcts_policy(config)

    # p = p_to_end
    # p = p_qmdp
    # p = p_mcts_mdp

    # total_rewards = []


    save_path = "qmdp_discrete_$(config).jld"
    p_qmdp = load_policy(save_path)


    for exp = 1:num_trials
        println(string(exp))

        Random.seed!(exp)

        p_to_end = ToEnd(0, get_goal_xy(m))
        traj_rewards_to_end = sum([step.r for step in stepthrough(m,p_to_end,belief_updater, max_steps=100)])
        traj_rewards_qmdp = sum([step.r for step in stepthrough(m,p_qmdp,belief_updater, max_steps=100)])

        push!(total_rewards_to_end, traj_rewards_to_end)
        push!(total_rewards_qmdp, traj_rewards_qmdp)

        #=
        traj_rewards = sum([step.r for step in stepthrough(m,p,belief_updater, max_steps=100)])
        println(traj_rewards)
        push!(total_rewards, traj_rewards)
        @printf("Mean Total Reward: %.3f, StdErr Total Reward: %.3f\n", mean(total_rewards), std(total_rewards)/sqrt(length(total_rewards)))
        =#
    end


    # Config 3, 50 trials
    # QMDP + MCTS, c=10, iter=2000: mean 2.732, stderr 0.904
    # MDP value estimation + MCTS, c=10, iter=2000: mean 2.990 stderr 0.948

    # Config 1, 100 trials
    # ToEnd Mean Total Reward: -2.045, StdErr Total Reward: 0.865
    # QMDP (50) Mean Total Reward: 2.389, StdErr Total Reward: 0.714
    # QMDP (25) Mean Total Reward: 1.312, StdErr Total Reward: 0.841

    # Config 2, 100 trials
    # ToEnd Mean Total Reward: -6.004, StdErr Total Reward: 0.852
    # QMDP (50) Mean Total Reward: 1.054, StdErr Total Reward: 0.863
    # QMDP (25) Mean Total Reward: -0.576, StdErr Total Reward: 0.854

    # Config 3, 100 trials
    # ToEnd Mean Total Reward: -1.879, StdErr Total Reward: 0.867
    # QMDP (50) Mean Total Reward: 1.393, StdErr Total Reward: 0.825
    # QMDP (25) Mean Total Reward: 0.848, StdErr Total Reward: 0.868

    @printf("ToEnd Mean Total Reward: %.3f, StdErr Total Reward: %.3f\n", mean(total_rewards_to_end), std(total_rewards_to_end)/sqrt(num_trials))
    @printf("QMDP Mean Total Reward: %.3f, StdErr Total Reward: %.3f\n", mean(total_rewards_qmdp), std(total_rewards_qmdp)/sqrt(num_trials))
end

if abspath(PROGRAM_FILE) == @__FILE__ # equivalent to if __name__ == "__main__"
    println("Running main")
    main()
end
