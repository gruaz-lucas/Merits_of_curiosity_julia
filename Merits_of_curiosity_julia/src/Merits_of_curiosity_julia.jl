module Merits_of_curiosity_julia
    using Distributed
    using JLD
    include("room_environment.jl")
    include("QAgent.jl")
    include("simulation.jl")
    export RoomEnvironment, reset!, step!, render, EnvParams
    export QAgent, update!, sample_action
    export create_intr_ag_from_env, run_step, partial_measure, compute_rewards_under_given_policy, all_time_spent, perf_evolution, score_intr_param, score_intr_param_nov_ig
    include("plotting.jl")
    export plot_all_early, plot_perf_across_envs, plot_avg_normalized, time_spent_stackplot_full
end # module Merits_of_curiosity_julia
