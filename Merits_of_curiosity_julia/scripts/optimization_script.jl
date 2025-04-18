println("Starting script!")
import Pkg
println("pkg imported")
Pkg.activate(".")
println("pkg activated")

using Distributed
println("distributed loaded")
intr_types = ["novelty_eps", "surprise", "information_gain", "empowerment", "MOP", "SP"]
measures = ["state_discovery", "model_accuracy", "uniform_state_visitation"]
n_workers = 55
# Add workers
println("starting to add workers")
if nprocs() < n_workers+1
    addprocs(n_workers+1 - nprocs())
end
println("workers added")

@everywhere using Merits_of_curiosity_julia
@everywhere using Random
@everywhere using SharedArrays
@everywhere using JLD


println("Packages downloaded")

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

println("setups defined")

model_known=false
measure_params = Dict("state_discovery"=>[500, 100], "model_accuracy"=>[500, 100], "uniform_state_visitation"=>[500, 100], "all"=>[0, 0], "Phat_KL"=>[500, 100], "uniform_state_visitation_KL"=>[500, 100])
n_envs = 50
T_PS = 100

seed=42
b_e=0.0
l_e=0.0
Rhat_0=0.0
update_reward=true

Random.seed!(seed)

n_params = 3 # ε, λ, β
ϵs = zeros(length(setups))
λs = zeros(length(setups))
for (is,s) in enumerate(setups)
    p1 = s[1]
    n_rooms = Int(floor(p1.n_s * p1.p_room))
    n_states = p1.n_s + n_rooms*(p1.room_size^2-1)
    ϵs[is] = 1/n_states
    λs[is] = (1/2)^(2/n_states) #n_states/2 th root of 0.5
end

n_βs = 100
βs = exp10.(range(-2, stop=2, length=n_βs))

envs = Array{RoomEnvironment}(undef, length(setups), n_envs)

for (is, setup) in enumerate(setups)
    env_param, setup_name = setup
    for e = 1:n_envs
        env = RoomEnvironment(env_param)
        envs[is, e] = env
    end
end

println("env generated")

# List of all settings (environment, measure, intrinsic type)
settings = [(is,setup[2],im,m,it,t,ib,β) for (is,setup) in enumerate(setups), (im,m) in enumerate(measures), (it,t) in enumerate(intr_types), (ib,β) in enumerate(βs)]

# For each setting (measure, intrinsic type), store the optimized parameters (ε, λ_i, β_i)
opti_params = SharedArray{Float64}(length(setups), length(measures), length(intr_types), n_params)
all_βs = SharedArray{Float64}(length(setups), length(measures), length(intr_types), n_βs)
fill!(all_βs, 100.0) # bad initial value


println("Starting main loop:")
# Distribute optimization for each setting
@sync @distributed for (is, setup_name, im, measure, it, intr_type, ib, β) in settings
    
    m_param = measure_params[measure] # parameter of the measure
    setup_envs = envs[is, :]

    ϵ = ϵs[is]
    λ = λs[is]
    # Function to optimize
    x = (ϵ, λ, β)
    score = 0.0
    if intr_type != "MOP" # no need to optimize for MOP
        score = score_intr_param(x, 0, setup_envs, intr_type, measure, m_param, b_e, l_e, T_PS, Rhat_0, update_reward, model_known=model_known)
    end

    # Path to save intermediate results
    path_save = "data/intermediate"

    # Save results
    min_score, min_β = findmin(all_βs[is,im,it,:])
    if score < min_score
        opti_params[is,im,it,:] .= (ϵ, λ, β)
        save(string(path_save, "/", setup_name, "-", measure, "-",intr_type, "-intermed_save.jld"),
                        "x_min", x, "y_min", score)
        println("---------------------------------------------")
        println("Setup: ", setup_name, ", Measure: ", measure, ", Intrinsic type: ", intr_type)
        println("-------")
        @show x
        @show score
    end
    all_βs[is, im, it, ib] = score
end

println("END OF OPTIMIZATION, SAVING RESULTS")
# Save scores for all βs
save(opti_path*"all_βs.jld", "results", all_βs.s, "βs", βs, "setups", setups, "intr_types", intr_types, "measures", measures)
# Save final results, one file per setup
for (is, setup) in enumerate(setups)
    env_params, setup_name = setup
    out_filename = opti_path * setup_name * ".jld"
    setup_envs = envs[is,:]
    setup_results = opti_params.s[is,:,:,:]
    save(out_filename, "env_params", env_params, "intr_types", intr_types, "measures", measures, "measure_params", measure_params, "opti_params", setup_results,
        "b_e", b_e, "l_e", l_e, "T_PS", T_PS, "Rhat_0", Rhat_0, "update_reward", update_reward, "model_known", model_known
    )
end

println("RESULTS SAVED")
