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
            err += sum((env.Psa_s[s,a,:] .- ag.Phat_sa_s[s,a,:]).^2)
        end
    end
    err = err / (env.n_tot_states * sum(env.n_actions_per_state))
    return err
end

"""Third measure"""
function state_freq_error(ag)
    n_steps = sum(ag.C_s)
    state_freq = ag.C_s ./ n_steps
    # mean squared error
    err = sum((state_freq .- (1/ag.n_states)).^2) / ag.n_states
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


function run_step(ag, env, n_steps, t_init)
    t = t_init
    for i in 1:n_steps
        t += 1
        step_update!(ag, env, t)
    end
    return ag
end

function partial_measure(ag, env, measure)
    if measure == "state_discovery"
        # Return fraction of unvisited states
        unvisited = count(x->(x==0), ag.C_s)
        return unvisited / ag.n_states

    elseif measure == "model_accuracy"
        # Return MSE between its estimated transition probabilities
        # and the environment's true transition probabilities
        return phat_error(ag, env)

    elseif measure == "uniform_state_visitation"
        # Return MSE between the state frequencies
        # and the uniform distribution
        return state_freq_error(ag)
    else
        throw(DomainError(measure_type, "Unknown measure type"))
    end
end
