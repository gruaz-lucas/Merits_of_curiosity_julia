import Pkg
Pkg.activate(".")

#Pkg.instantiate()
using Distributed
n_workers = 55
# Add workers
if nprocs() < n_workers+1
    addprocs(n_workers+1 - nprocs())
end

@everywhere using Merits_of_curiosity_julia
@everywhere using Random
@everywhere using SharedArrays
@everywhere using JLD

seed=42
Random.seed!(seed)

env_params = EnvParams(40, 0.2, 4, 0.1, 1/4, 1/4, 1/4, 50, 50, 1.0)
env_name = "sink50-source50-stoc1"
n_envs = 50
intr_types = ["novelty_eps", "surprise", "information_gain", "empowerment", "MOP", "SP", "extrinsic"]
n_steps = 10000

n_rooms = Int(floor(env_params.n_s * env_params.p_room))
n_states = env_params.n_s + n_rooms*(env_params.room_size^2-1)
ϵ = 1/n_states
λ = (1/2)^(2/n_states) #n_states/2 th root of 0.5

T_PS = 100
path = "" # time spent path

β_factor = 1.0

compute_rewards_under_policy = true

model_known=true

if compute_rewards_under_policy
    for intr_type in intr_types
        result_path = path * intr_type * ".jld"
        compute_rewards_under_given_policy(env_params, env_name, n_envs, intr_type, intr_types, n_steps, ϵ, λ, T_PS; path=result_path, model_known=model_known)
    end
else
    all_time_spent(env_params, env_name, n_envs, intr_types, n_steps, ϵ, λ, T_PS, path=path, model_known=model_known, β_factor=β_factor, compute_rewards=compute_rewards)
end




