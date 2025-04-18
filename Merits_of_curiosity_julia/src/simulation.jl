using Merits_of_curiosity_julia

"""
Execute one step of the agent in the environment. t is the current time (step).
"""
function step_update!(ag, env::RoomEnvironment, t)
    at = sample_action(ag)
    s_prime, r = step!(env, at)

    # Update agent depending on its type (QAgent of RandomAgent)
    if typeof(ag) == QAgent
        update!(ag, s_prime, r, t)
        update_policy!(ag)
    else
        throw(DomainError(ag, "Unknown agent type"))
    end
    return r
end

"""First measure"""
function visit_error(ag)
    unvisited = count(x->(x==0), ag.C_s)
    return unvisited / ag.n_states
end

"""Second measure"""
function phat_error(ag, env::RoomEnvironment)
    err = 0
    for s in 1:env.n_tot_states
        for a in 1:env.n_actions_per_state[s]
            err += sum((env.transition_matrix[s,a,:] .- ag.Phat_sa_s[s,a,:]).^2)
        end
    end
    err = err / (env.n_tot_states * sum(env.n_actions_per_state))
    return err
end

"""Third measure"""
function uniform_state_visitation_error(ag)
    n_steps = sum(ag.C_s)
    uniform_state_visitation = ag.C_s ./ n_steps
    # mean squared error
    err = sum((uniform_state_visitation .- (1/ag.n_states)).^2) / ag.n_states
    return err
end

"""
create_intr_ag_from_env(e, l_i, b_i, intrinsic_type, env, T_PS = 10, b_e = 0.0, l_e = 0.0, Rhat_0 = 0.0, update_reward = true)

creates an agent with the given intrinsic parameters (e, l_i, b_i, intrinsic_type)
that can be then used on the given environment (n_states and n_actions correspond, and initial state is picked)
Other parameters of the agent are optional
"""
function create_intr_ag_from_env(e, l_i, b_i, intrinsic_type, env, T_PS = 50, b_e = 0.0, l_e = 0.0, Rhat_0 = 0.0, update_reward = true; nov_ig_alpha=0.5, sp_α=0.0)
    n_states = env.n_tot_states
    n_actions_per_state = env.n_actions_per_state
    st = env.state
    return QAgent(e, b_i, b_e, l_i, l_e, T_PS, Rhat_0, st, intrinsic_type, n_states, n_actions_per_state, update_reward, nov_ig_alpha=nov_ig_alpha, sp_α=sp_α)
end


function run_step(ag, env, n_steps, t_init; store_time_spent=false, meas=nothing)
    t = t_init
    states = zeros(Int64, n_steps)
    for i in 1:n_steps
        t += 1
        add_extr_reward_if_necessary(ag, env, t, meas)
        step_update!(ag, env, t)
        if store_time_spent
            states[i] = ag.st
        end
    end
    return ag, states
end

function add_extr_reward_if_necessary(ag, env, t, meas)
    if meas !== nothing && occursin("extr", meas)
        parts = split(meas, "_")
        if parts[2] == "start"
            if t == 1
                add_random_goal!(env)
            end
        elseif parse(Int, parts[2]) == t
            println("Adding extrinsic reward at time: ", t)
            add_random_goal!(env)
            if parts[3] == "known" && typeof(ag) == QAgent
                ag.Rhat_sa_s = env.reward_matrix # agent knows the reward function
                ag.update_reward = false
                ag.intrinsic_type = "extrinsic"
                ag.β_i = 0.0 # to ensure non-zero beta
                ag.β_e = 1.0
                ag.λ_e = ag.λ_i
                ag.is_greedy = true # exploitation
                #intrinsic_reward!(ag, ag.intrinsic_type, t, -1)
                #PSA!(ag, true)
                PSA!(ag, false)
                update_policy!(ag)
            end
        end
    end
end

function partial_measure(ag, env, measure)
    if measure == "state_discovery"
        # Return fraction of unvisited states
        return visit_error(ag)

    elseif measure == "model_accuracy"
        # Return MSE between its estimated transition probabilities
        # and the environment's true transition probabilities
        return phat_error(ag, env)

    elseif measure == "uniform_state_visitation"
        # Return MSE between the state frequencies
        # and the uniform distribution
        return uniform_state_visitation_error(ag)
    else
        throw(DomainError(measure_type, "Unknown measure type"))
    end
end


function measure_performance(ag_original, env::RoomEnvironment, measure_type::String, measure_params)
    # Deep copy of agent to not modify the original one
    ag = copy(ag_original)
    n_iter, eval_every = measure_params
    perfs = []
    for is in 1:div(n_iter,eval_every)
        # Run step_size steps
        ag, _ = run_step(ag, env, eval_every, (is-1)*eval_every)
        # Store partial measure
        push!(perfs, partial_measure(ag, env, measure_type))
    end
    return mean(perfs)
end

"""
Compute the score of the agent
- x: parameters of the agent (epsilon, lambda_i, beta_i)
- g: gradient of the function (not used here)
- envs: environments to test the agent on
- intrinsic_type: type of intrinsic reward
- measure_type: performance measure ("state_discovery", "model_accuracy", "uniform_state_visitation")
"""
function score_intr_param(x, g, envs, intrinsic_type, measure_type, measure_params, b_e, l_e, T_PS, Rhat_0, update_reward; sp_α=0.0, model_known=false)
    e, l_i, b_i = x
    score = 0
    # compute the performance of the agent on the environments and returns the mean score
    for env in envs
        reset!(env)
        ag = QAgent(e, b_i, b_e, l_i, l_e, T_PS, Rhat_0, env.state, intrinsic_type, env.n_tot_states, env.n_actions_per_state, update_reward, sp_α=sp_α)
        if model_known
            fix_model!(ag, env.transition_matrix)
        end

        s = measure_performance(ag, env, measure_type, measure_params)
    
        score += s
    end
    return score/length(envs)
end

"""
Same as score_intr_params but for a combination of Novelty and Information gain. Alpha is the weighting factor.
"""
function score_intr_param_nov_ig(x, g, envs, intrinsic_type, measure_type, measure_params, b_e, l_e, T_PS, Rhat_0, update_reward)
    e, l_i, b_i, alpha = x
    score = 0
    # compute the performance of the agent on the environments and returns the mean score
    for env in envs
        reset!(env)
        ag = QAgent(e, b_i, b_e, l_i, l_e, T_PS, Rhat_0, env.state, intrinsic_type, env.n_tot_states, env.n_actions_per_state, update_reward, nov_ig_alpha=alpha)

        s = measure_performance(ag, env, measure_type, measure_params)
    
        score += s
    end
    return score/length(envs)
end


function perf_evolution(env_params, opti_filename, step_size, n_steps, n_repet, out_filename; seed=4, show_degree=false, novig=false, showrandom=false, intr_types=nothing, envs_file=nothing, store_envs=false, measure_transfer_dict=nothing)
    opti_data = load(opti_filename)
    # agents_params: array of dimension (n_measures, n_intr_types, n_params)
    agents_params = opti_data["opti_params"]
    T_PS = opti_data["T_PS"]

    measures = opti_data["measures"]

    if isnothing(intr_types)
        intr_types = novig ? ["nov_ig"] : opti_data["intr_types"]
    end

    n_alphas = novig ? 5 : 0

    model_known = get(opti_data, "model_known", false) # Whether the env was known during optimization

    if showrandom
        intr_types = [intr_types; "random"]
    end

    Random.seed!(seed)

    scores = novig ? SharedArray{Float64}(length(measures), n_alphas, n_steps, n_repet) : SharedArray{Float64}(length(measures), length(intr_types), n_steps, n_repet)
  
    time_spent = SharedArray{Int64}(length(measures), length(intr_types), n_repet, 5) # 5 is for sink / source / stoc / neutral room / corridor
    
    envs = Array{RoomEnvironment}(undef, n_repet)
    if envs_file === nothing
        for ie = 1:n_repet
            env = RoomEnvironment(env_params)
            envs[ie] = env
        end
    else
        stored_envs = load(envs_file)["envs"][1:n_repet] # TODO change that

        length(stored_envs) != n_repet && throw(ArgumentError(string(n_repet)))
        for (i,e) in enumerate(stored_envs)
            envs[i] = e
        end
    end

    if measure_transfer_dict !== nothing # we may want to measure something else
        for (im,meas) in enumerate(measures)
            measures[im] = measure_transfer_dict[meas]
        end
    end

    @sync @distributed for ir in 1:n_repet
 
        for (im, meas) in enumerate(measures)
            
            if novig
                for ia in 1:n_alphas
                    env = copy(envs[ir])
                    env.reward_matrix .= 0.0
                    
                    eps, l_i, b_i, nov_ig_alpha = agents_params[im, ia, :]
                    ag = create_intr_ag_from_env(eps, l_i, b_i, "nov_ig", env, T_PS, nov_ig_alpha=nov_ig_alpha)
                    if model_known
                        fix_model!(ag, env.transition_matrix)
                    end
                                        
                    for is in 1:n_steps
                        # Run step_size steps
                        ag,_ = run_step(ag, env, step_size, (is-1)*step_size, meas=meas)
                        # Store partial measure
                        scores[im, ia, is, ir] = partial_measure(ag, env, meas)
                        
                    end
                end
            else
                for (it, intr_type) in enumerate(intr_types)
                    # set rewards to 0
                    env = copy(envs[ir])
                    reset!(env)
                    env.reward_matrix .= 0.0
                    ag = nothing
                    n_states = env.n_tot_states
                    if intr_type == "random"
                        ag = RandomAgent(env.n_actions_per_state, env.n_tot_states, env.state, 1/n_states)
                    else

                        
                        eps, l_i, b_i = 1/n_states, (1/2)^(2/n_states), 0.0
                        if intr_type != "extrinsic"
                            eps, l_i, b_i = agents_params[im, it, :]
                        end

                        nov_ig_alpha = 0.5

                        ag = create_intr_ag_from_env(eps, l_i, b_i, intr_type, env, T_PS, nov_ig_alpha=nov_ig_alpha)
                        if model_known
                            fix_model!(ag, env.transition_matrix)
                        end
                    end
                        
                    for is in 1:n_steps
                        # Run step_size steps
                        ag,_ = run_step(ag, env, step_size, (is-1)*step_size, meas=meas)
                        # Store partial measure
                        scores[im, it, is, ir] = partial_measure(ag, env, meas)
                    end
                end
            end
        end
    end

    if !store_envs
        envs = nothing
    end

    save(out_filename, "scores", scores.s, "env_params", env_params, "envs", envs, "step_size", step_size, "measures", measures, "intr_types", intr_types, "time_spent", time_spent.s)
end

function grid_to_number(grid::Grid)
    grid.is_sink && return 1
    grid.is_source && return 2
    grid.is_stoc && return 3
    return 4
end

function compute_time_spent(intr_type, env, grids, n_steps, ϵ, λ, β, T_PS; model_known=false)
    rewards = zeros(n_steps)
    time_spent = zeros(Int64, n_steps)
    r_min, r_max = Inf, -Inf

    ag = create_intr_ag_from_env(ϵ, λ, β, intr_type, env, T_PS, nov_ig_alpha=false)
    if model_known
        fix_model!(ag, env.transition_matrix)
    end
    for is in 1:n_steps
        r = step_update!(ag, env, is)
        rewards[is] = r
        r < r_min && (r_min = r)
        r > r_max && (r_max = r)
        # which grid contains st ?
        location = 5 # corridor
        for grid in grids
            if ag.st in grid.node_ids
                location = grid_to_number(grid)
            end
        end
        time_spent[is] = location
    end

    std_reward = std(rewards)
    return std_reward, time_spent
end

function save_time_spent(intr_type, envs, grids_vec, n_steps, ϵ, λ, β, T_PS, out_filename; model_known=false)
    n_envs = length(envs)
    times = SharedArray{Int64}(n_envs, n_steps)
    std_rewards = SharedArray{Float64}(n_envs)

    @sync @distributed for ie in 1:n_envs
        env = envs[ie]
        grids = grids_vec[ie]
        r, time = compute_time_spent(intr_type, env, grids, n_steps, ϵ, λ, β, T_PS, model_known=model_known)
        times[ie,:] .= time
        std_rewards[ie] = r
    end
    std_reward = mean(std_rewards.s)
    save(out_filename, "time_spent", times.s, "std_reward", std_reward)
end

function all_time_spent(env_params, env_name, n_envs, intr_types, n_steps, ϵ, λ, T_PS; path="data/perfs/time_spent/env3_4rooms/", model_known=false, β_factor=1)
    #n_intr = length(intr_types)
    
    grids_vec = []
    envs = Array{RoomEnvironment}(undef, n_envs)
    for e = 1:n_envs
        env = RoomEnvironment(env_params)
        grids = env.grids
        envs[e] = env
        push!(grids_vec, grids)
    end

    @sync @distributed for intr in intr_types
        β0_path = path*env_name*"_"*intr*"_β0.jld"
        if !isfile(β0_path)
            println("IM: ", intr, ", computing reward for β=0")
            save_time_spent(intr, envs, grids_vec, n_steps, ϵ, λ, 0.0, T_PS, β0_path, model_known=model_known)
        end
        d = load(β0_path)
        β = β_factor/d["std_reward"]
        #intr == "MOP" && (β = min(0.5, β))
        
        println("IM: ", intr, ", β: ", β)
        save_path = path*env_name*"_"*intr*"_βopti.jld"
        save_time_spent(intr, envs, grids_vec, n_steps, ϵ, λ, β, T_PS, save_path, model_known=model_known)
        println("IM: ", intr, ", DONE")
    end
end

function compute_rewards_under_given_policy(env_params, env_name, n_envs, intr_type, intr_types, n_steps, ϵ, λ, T_PS; path=nothing, model_known=false)
    grids_vec = []
    envs = Array{RoomEnvironment}(undef, n_envs)
    for e = 1:n_envs
        env = RoomEnvironment(env_params)
        grids = env.grids
        #g, grids, stoc_v, _ = generate_graph(env_params)
        #env = graphToEnv(g, stoc_v, env_params)
        envs[e] = env
        push!(grids_vec, grids)
    end

    if path === nothing
        throw(ArgumentError("path is not defined"))
    end
    println("Path: ", path)
    

    # run random policy in env, compute all rewards at each step
    n_states = env_params.n_s
    rewards = SharedArray{Float64}(n_envs, n_steps, length(intr_types))
    states = SharedArray{Int64}(n_envs, n_steps)

    @sync @distributed for ie in 1:n_envs
        env = envs[ie]
        grids = grids_vec[ie]
        ag = create_intr_ag_from_env(ϵ, λ, 0.0, intr_type, env, T_PS)
        if model_known
            fix_model!(ag, env.transition_matrix)
        end
        # perform steps
        for is in 1:n_steps
            # select action
            at = sample_action(ag)
            s_prime, r = transition(env, ag.st, at)
            # compute all rewards
            for it in 1:length(intr_types)
                intrinsic_reward!(ag, intr_types[it], is, s_prime)
                rewards[ie, is, it] = ag.Ri_sa_s[ag.st, at, s_prime]
            end
            states[ie, is] = ag.st

            # Update agent
            update!(ag, s_prime, r, is)
            if "SP" in intr_types
                ag.sp_M = update_sp(ag)
            end
            update_policy!(ag)
        end
    
    end

    println("types of variables to save: ", typeof(rewards.s), typeof(states.s), typeof(envs))

    save(path, "rewards", rewards.s, "states", states.s)#, "envs", envs)
end