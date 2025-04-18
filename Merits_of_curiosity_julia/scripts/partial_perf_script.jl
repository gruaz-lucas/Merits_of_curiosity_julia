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
@everywhere using Graphs
@everywhere using Random
@everywhere using JLD

n_repet = 100 # CHANGE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
seed = 1

Random.seed!(seed)

data_path = ""
partial_perf_path = "" # path to store performance
opti_prefix = "" # path where optimized parameters are stored
envs_path = "" # path where the environments are stored (= nothing to create new environments)
store_envs = false
novig=true
intr_types = nothing



branch_rates = [0.0, 0.5, 1.0]
n_rooms = [1,2,4]
room_size= [2,3,4]
n_states = 100
setups = []
for br in branch_rates, nr in n_rooms, rs in room_size
    n_room_states = nr*rs*rs
    n_initial = n_states - n_room_states + nr #+ n_rooms because states are TRANSFORMED into rooms
    push!(setups, [EnvParams(n_initial, br, rs, nr/n_initial, 0.0, 0.0, 0.0, 0,  0, 0.0), "default_br"*string(br)*"_nr"*string(nr)*"_rs"*string(rs), 20, 100])
    push!(setups, [EnvParams(n_initial, br, rs, nr/n_initial, 0.0, 1/nr, 0.0, 50,  0, 0.0), "sink50_br"*string(br)*"_nr"*string(nr)*"_rs"*string(rs), 20, 100])
    push!(setups, [EnvParams(n_initial, br, rs, nr/n_initial, 0.0, 0.0, 1/nr, 0,  50, 0.0), "source50_br"*string(br)*"_nr"*string(nr)*"_rs"*string(rs), 20, 100])
    push!(setups, [EnvParams(n_initial, br, rs, nr/n_initial, 1/nr, 0.0, 0.0, 0,  0, 1.0), "stoc1_br"*string(br)*"_nr"*string(nr)*"_rs"*string(rs), 20, 100])
    frac=(1/4, 1/4, 1/4)
    if nr==2
        frac=(0.0, 1/2, 1/2)
    end
    push!(setups, [EnvParams(n_initial, br, rs, nr/n_initial, frac..., 50,  50, 1.0), "mixed_br"*string(br)*"_nr"*string(nr)*"_rs"*string(rs), 20, 100])

end


for setup in setups
    env_params, env_name, step_size, n_steps = setup
    println("Starting setup: "*env_name)
    
    opti_filename = data_path * opti_prefix * env_name *".jld"
    partial_perf_filename = partial_perf_path * env_name *"_"* string(step_size*n_steps) *".jld"
    envs_file = envs_path === nothing ? nothing : envs_path * env_name *"_"* string(step_size*n_steps) *".jld"

    perf_evolution(env_params, opti_filename, step_size, n_steps, n_repet, partial_perf_filename, seed=4, show_degree=false, novig=novig, showrandom=true, intr_types=intr_types, envs_file=envs_file, store_envs=store_envs)
    println("End of setup: "*env_name)
end

println("END OF ALL *dramatic music*")
