using AA228FinalProject
using POMDPs
using ParticleFilters

# Define the policy to test
mutable struct ToEnd <: Policy
    ts::Int64 # to track the current time-step.
    goal_xy::Vector{Float64}
end

# define a new function that takes in the policy struct and current belief,
# and returns the desired action
function POMDPs.action(p::ToEnd, b::ParticleCollection{RoombaState})

    # spin around to localize for the first 25 time-steps
    if p.ts < 25
        p.ts += 1
        return RoombaAct(0.,1.0) # all actions are of type RoombaAct(v,om)
    end
    p.ts += 1

    # after 25 time-steps, we follow a proportional controller to navigate
    # directly to the goal, using the mean belief state

    # compute mean belief of a subset of particles
    s = mean(b)

    # compute the difference between our current heading and one that would
    # point to the goal
    goal_x, goal_y = p.goal_xy
    x,y,th = s[1:3]
    ang_to_goal = atan(goal_y - y, goal_x - x)
    del_angle = wrap_to_pi(ang_to_goal - th)

    # apply proportional control to compute the turn-rate
    Kprop = 1.0
    om = Kprop * del_angle

    # always travel at some fixed velocity
    v = 5.0

    return RoombaAct(v, om)
end
