import Pkg
Pkg.activate(".")
#Pkg.instantiate()
using Distributed
intr_type = "nov_ig"
measures = ["state_discovery", "model_accuracy", "uniform_state_visitation"]
n_workers = 55
# Add workers
if nprocs() < n_workers+1
    addprocs(n_workers+1 - nprocs())
end

@everywhere using Merits_of_curiosity_julia
@everywhere using Random
@everywhere using SharedArrays
@everywhere using JLD


opti_path = "" # path to store optimization results


branch_rates = [0.0, 0.5, 1.0]
n_rooms = [1,2,4]
room_size= [2,3,4]
n_states = 100
setups = []
for br in branch_rates, nr in n_rooms, rs in room_size
    n_room_states = nr*rs*rs
    frac=(1/4, 1/4, 1/4)
    if nr==2
        frac=(0.0, 1/2, 1/2)
    end
    n_initial = n_states - n_room_states + nr #+ n_rooms because states are TRANSFORMED into rooms
    push!(setups, [EnvParams(n_initial, br, rs, nr/n_initial, 0.0, 0.0, 0.0, 0,  0, 0.0), "default_br"*string(br)*"_nr"*string(nr)*"_rs"*string(rs)])
    push!(setups, [EnvParams(n_initial, br, rs, nr/n_initial, 0.0, 1/nr, 0.0, 50,  0, 0.0), "sink50_br"*string(br)*"_nr"*string(nr)*"_rs"*string(rs)])
    push!(setups, [EnvParams(n_initial, br, rs, nr/n_initial, 0.0, 0.0, 1/nr, 0,  50, 0.0), "source50_br"*string(br)*"_nr"*string(nr)*"_rs"*string(rs)])
    push!(setups, [EnvParams(n_initial, br, rs, nr/n_initial, 1/nr, 0.0, 0.0, 0,  0, 1.0), "stoc1_br"*string(br)*"_nr"*string(nr)*"_rs"*string(rs)])
    push!(setups, [EnvParams(n_initial, br, rs, nr/n_initial, frac..., 50,  50, 1.0), "mixed_br"*string(br)*"_nr"*string(nr)*"_rs"*string(rs)])

end

measure_params = Dict("state_discovery"=>[500, 100], "model_accuracy"=>[500, 100], "uniform_state_visitation"=>[500, 100], "all"=>[0, 0])
n_envs = 50
T_PS = 100

seed=42
b_e=0.0
l_e=0.0
Rhat_0=0.0
update_reward=true

Random.seed!(seed)

n_params = 4 # epsilon, lambda, β
p1 = setups[1][1]
n_rooms = Int(floor(p1.n_s * p1.p_room))
n_states = p1.n_s + n_rooms*(p1.room_size^2-1)
ϵ = 1/n_states
λ = (1/2)^(2/n_states) #n_states/2 th root of 0.5

n_βs = 100
βs = exp10.(range(-2, stop=2, length=n_βs))

αs = [0.0, 0.25, 0.5 ,0.75, 1.0]

envs = Array{RoomEnvironment}(undef, length(setups), n_envs)

for (is, setup) in enumerate(setups)
    env_params, setup_name = setup
    for e = 1:n_envs
        env = RoomEnvironment(env_params)
        envs[is, e] = env
    end
end

# List of all settings (environment, measure, intrinsic type)
settings = [(is,setup[2],im,m,ib,β,ia,α) for (is,setup) in enumerate(setups), (im,m) in enumerate(measures), (ib,β) in enumerate(βs), (ia,α) in enumerate(αs)]

# For each setting (measure, intrinsic type), store the optimized parameters (epsilon, lambda_i, β_i)
opti_params = SharedArray{Float64}(length(setups), length(measures), length(αs), n_params)
all_βs = SharedArray{Float64}(length(setups), length(measures), length(αs), n_βs)
fill!(all_βs, 100.0) # bad initial value


# Distribute optimization for each setting
@sync @distributed for (is, setup_name, im, measure, ib, β, ia, α) in settings
    m_param = measure_params[measure] # parameter of the measure
    setup_envs = envs[is, :]
    # Function to optimize
    x = (ϵ, λ, β, α)
    score = score_intr_param_nov_ig(x, 0, setup_envs, intr_type, measure, m_param, b_e, l_e, T_PS, Rhat_0, update_reward)

    # Path to save intermediate results
    path_save = "data/intermediate"

    # Save results
    min_score, min_β = findmin(all_βs[is,im,ia,:])
    if score < min_score
        opti_params[is,im,ia,:] .= x
        save(string(path_save, "/", setup_name, "-", measure, "-",intr_type, "-intermed_save.jld"),
                        "x_min", x, "y_min", score)
        println("---------------------------------------------")
        println("Setup: ", setup_name, ", Measure: ", measure, ", Intrinsic type: ", intr_type)
        println("-------")
        @show x
        @show score
    end
    all_βs[is, im, ia, ib] = score    
end

# Save scores for all βs
save(opti_path*"all_βs.jld", "results", all_βs.s, "βs", βs, "setups", setups, "intr_type", intr_type, "measures", measures)
# Save final results, one file per setup
for (is, setup) in enumerate(setups)
    env_params, setup_name = setup
    out_filename = opti_path * setup_name * ".jld"
    setup_envs = envs[is,:]
    setup_results = opti_params.s[is,:,:,:]
    save(out_filename, "env_params", env_params, "intr_type", intr_type, "measures", measures, "measure_params", measure_params, "opti_params", setup_results,
        "b_e", b_e, "l_e", l_e, "T_PS", T_PS, "Rhat_0", Rhat_0, "update_reward", update_reward, "αs", αs
    )
end

