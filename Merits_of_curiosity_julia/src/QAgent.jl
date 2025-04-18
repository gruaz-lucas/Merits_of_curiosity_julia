

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Structure of the Q-agent and functions applicable to it
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

mutable struct QAgent
    ε::Float64                # Prior for Phat
    β_i::Float64                 # Inverse temperature for intrinsic term in softmax policy
    β_e::Float64                 # Inverse temperature for extrinsic term softmax policy
    λ_i::Float64               # Discount factor for intrinsic rewards
    λ_e::Float64               # Discount factor for extrinsic rewards
    T_PS::Int64                     # Number of iteration of Prioritized Sweeping (function PSA)

    Rhat_0::Array{Float64,3}        # Prior belief for the extrinsic rewards
    
    st::Int64                       # Current state of the agent
    Qe_sa::Matrix{Float64}          # Extrinsic Q-values
    Qi_sa::Matrix{Float64}          # Intrinsic Q-values
    Ue_t_s::Vector{Float64}         # Max of Qe_sa over actions at time t
    Ui_t_s::Vector{Float64}         # Max of Qi_sa over actions at time t
    C_sa_s::Array{Float64,3}          # Counts of transition (s,a -> s')
    C_sa::Matrix{Float64}             # Counts of pairs (s,a)
    C_s::Vector{Float64}              # Counts of states
    CumReward_sa_s::Array{Float64,3}# Cumulative reward for transition (s,a -> s')
    Ri_sa_s::Array{Float64,3}       # Intrinsic reward of every transitions (s,a -> s') (depends on intrinsic_type)
    intrinsic_type::String          # type of intrinsic motivation : novelty, surprise or information_gain (or none)

    Phat_sa_s::Array{Float64,3}     # estimation of the environment's transition probabilities
    Rhat_sa_s::Array{Float64,3}     # estimation of the extrinsic rewards for each transition

    pi_a::Vector{Float64}           # Agent's policy for the current state
    at::Int64                       # Last action taken by agent

    n_states::Int64                 # Number of states of the environment
    n_actions_per_state::Vector{Int64}                # Number of actions for each state (constant)
    max_actions::Int64

    update_reward::Bool             # Decide whether to update Rhat_sa_s or not (can be set to false when no extrinsic rewards)
    nov_ig_alpha::Float64           # Factor for combination of novelty and information gain rewards

    mop_α::Float64
    mop_β::Float64
    mop_V::Vector{Float64}
    mop_π::Matrix{Float64}

    sp_M::Matrix{Float64}
    sp_α::Float64
    sp_TPS::Int64

    model_fixed::Bool               # Whether the environment is known or not
    extr_reward_encounter_time::Int64 # First time the agent encounters an extrinsic reward
    is_greedy::Bool
end


"""
QAgent(e::Float64, b_i::Float64, b_e::Float64, l_i::Float64, l_e::Float64, T_PS::Int64, Rhat_0, st::Int64, intrinsic_type::String, n_states::Int64, n_actions_per_state::Vector{Int64}, update_reward::Bool)

Create a QAgent with the given attributes and default value for others
"""
function QAgent(e::Float64, b_i::Float64, b_e::Float64, l_i::Float64, l_e::Float64, T_PS::Int64, Rhat_0, st::Int64, intrinsic_type::String, n_states::Int64, n_actions_per_state::Vector{Int64}, update_reward::Bool; nov_ig_alpha=0.5, sp_α=0.0, sp_TPS=50)
    max_actions = maximum(n_actions_per_state)
    # Initialize all counts
    C_s = zeros(Float64, n_states)
    C_sa = zeros(Float64, n_states, max_actions)
    C_sa_s = zeros(Float64, n_states, max_actions, n_states)
    C_s[st] = 1

    # Rhat_0 can be either Vector of Float or a single Float. It must be extended to 3D matrix
    if typeof(Rhat_0) == Vector{Float64} && length(Rhat_0) == n_states
        mean_reward = sum(Rhat_0)/n_states
        Qe_sa = fill(mean_reward/(1-l_e), (n_states, max_actions))
        Ue_1_s = fill(mean_reward/(1-l_e), n_states)
        Rhat_0 = repeat(Rhat_0, inner=(1, max_actions, n_states))
        Rhat_sa_s = Rhat_0
    elseif typeof(Rhat_0) == Float64
        Qe_sa = fill(Rhat_0/(1-l_e), (n_states, max_actions))
        Ue_1_s = fill(Rhat_0/(1-l_e), n_states)
        Rhat_0 = fill(Rhat_0, (n_states, max_actions, n_states))
        Rhat_sa_s = Rhat_0
    else
        println("Wrong type of Rhat_0")
    end

    # Initialize Q-values
    Ui_1_s = zeros(Float64, n_states) #fill(log(n_states)/(1-l_i), n_states)
    Qi_sa = zeros(Float64, n_states, max_actions) #fill(Ui_1_s[1], (n_states, max_actions))

    # Initialize rewards and probabilities
    CumReward_sa_s = zeros(n_states, max_actions, n_states)
    Ri_sa_s = zeros(n_states, max_actions, n_states)
    Phat_sa_s = fill(1/n_states, (n_states, max_actions, n_states))

    pi_a = fill(1/n_actions_per_state[st], n_actions_per_state[st])
    at = 0

    mop_α = 1.0
    mop_β = 1.0
    # Initialize state-value function
    mop_V = zeros(n_states)

    # Initialize policy
    mop_π = zeros(n_states, maximum(n_actions_per_state))
    for s in 1:n_states, a in 1:n_actions_per_state[s]
        mop_π[s,a] = 1/n_actions_per_state[s]
    end

    sp_M = zeros(Float64, n_states, n_states)

    model_fixed = false
    extr_reward_encounter_time = -1
    is_greedy = false
    # Create the agent and return it
    return QAgent(e, b_i, b_e, l_i, l_e, T_PS, Rhat_0, st, Qe_sa, Qi_sa, Ue_1_s, Ui_1_s, C_sa_s, C_sa, C_s, CumReward_sa_s, Ri_sa_s, intrinsic_type, Phat_sa_s, Rhat_sa_s, pi_a, at, n_states, n_actions_per_state, max_actions, update_reward, nov_ig_alpha, mop_α, mop_β, mop_V, mop_π, sp_M, sp_α, sp_TPS, model_fixed, extr_reward_encounter_time, is_greedy)
end


"""
sample_action(agent::QAgent)

Sample an action following the policy pi_a of the agent
"""
function sample_action(agent::QAgent)
    try
        actions = 1:agent.n_actions_per_state[agent.st]         # All possible actions
        weights = Weights(agent.pi_a)       # weighted according to policy
        agent.at = sample(actions, weights) # sample one
        return agent.at
    catch e
        println("sample_action error with agent:")
        println(agent)
        println("Intr type: ", agent.intrinsic_type)
        println("Policy: ")
        println(agent.pi_a)
        println("Qvalues: ")
        println(agent.Qi_sa)
        rethrow(e)
    end
end

function compute_policy(agent::QAgent, s)
    # Q = β_i Q_i + β_e Q_e
    Q_s = agent.β_i * agent.Qi_sa[s, 1:agent.n_actions_per_state[s]] + agent.β_e * agent.Qe_sa[s, 1:agent.n_actions_per_state[s]]
    # subtract the minimum of Q_values to all Q-values to avoid overflows
    m = min(Q_s...)
    Q_s_resized = Q_s .- m
    # Take softmax on Q-values as policy
    expo = exp.(Q_s_resized)
    pi_a = expo/sum(expo)
    if any(x->isnan(x), pi_a)
        # Deal with NaNs when expo is too large for example
        pi_a = replace(pi_a, NaN=>1)
        pi_a = pi_a ./ sum(pi_a)
    end
    if agent.is_greedy
        max_pi_ids = findall(x->x==maximum(pi_a), pi_a)
        pi_a .= 0
        pi_a[max_pi_ids] .= 1/length(max_pi_ids)
    end
    return pi_a
end


"""
update_policy!(agent::QAgent)

Update the policy using softmax on the Q-Values, for state st
"""
function update_policy!(agent::QAgent)
    if agent.intrinsic_type == "MOP"
        mop_optimize_policy!(agent)
        #pi_a = 
        agent.pi_a = agent.mop_π[agent.st, 1:agent.n_actions_per_state[agent.st]]
    else
        agent.pi_a = compute_policy(agent, agent.st)
    end
    
end


"""
update!(agent::QAgent, s_prime::Int64, r::Float64, t::Int64)

Update the agent after receiving next state s_prime and reward r at time t
"""
function update!(agent::QAgent, s_prime::Int64, r::Float64, t::Int64)
    # Update counts
    agent.C_sa_s[agent.st, agent.at, s_prime] += 1
    agent.C_sa[agent.st, agent.at] += 1
    agent.C_s[s_prime] += 1

    if agent.extr_reward_encounter_time == -1 && r > 0
        agent.extr_reward_encounter_time = t
    end

    # Update Phat, i.e the estimation of the transition probabilities
    if !agent.model_fixed
        alpha_sa_s = agent.C_sa_s .+ agent.ε
        agent.Phat_sa_s = alpha_sa_s ./ sum(alpha_sa_s, dims=3)
    end

    # Update the cumulative rewards
    agent.CumReward_sa_s[agent.st, agent.at, s_prime] += r

    # Update Rhat = cumreward/count
    if agent.update_reward
        agent.Rhat_sa_s[agent.st, agent.at, s_prime] = agent.CumReward_sa_s[agent.st, agent.at, s_prime] / agent.C_sa_s[agent.st, agent.at, s_prime]
    end

    if agent.intrinsic_type == "SP"
        agent.sp_M = update_sp(agent)
    end

    if agent.intrinsic_type != "MOP"
        if agent.intrinsic_type != "none"
            # Update intrinsic rewards
            intrinsic_reward!(agent, agent.intrinsic_type, t, s_prime)
            # Update intrinsic U and Q-values
            PSA!(agent, true)
        end

        # Update extrinsic U and Q-values
        PSA!(agent, false)
    end

    # Update current state
    agent.st = s_prime

end


"""
intrinsic_reward!(agent::QAgent, reward_type::String, t::Int64, s_prime)

Compute the intrinsic reward of the agent (i.e update Ri_sa_s), depending on the reward type, the time t and the next state s_prime
"""
@views function intrinsic_reward!(agent::QAgent, reward_type::String, t::Int64, s_prime::Int64; eps=1e-10)
    if reward_type == "novelty"
        # Compute novelty of each state s
        # one entry for each state s
        nov_s = log.((t+agent.n_states)./(agent.C_s.+1))
        # repeat to have an entry for each (s,a,s')
        agent.Ri_sa_s = permutedims(repeat(nov_s, inner=(1, agent.max_actions, agent.n_states)), [3,2,1])
        return agent.Ri_sa_s

    elseif reward_type == "surprise"
        # Compute surprise for each transition (s,a,s')
        sur_sa_s = -log.(agent.Phat_sa_s)
        if agent.model_fixed && any(x->isinf(x), sur_sa_s)
            # Deal with Inf
            sur_sa_s = replace(sur_sa_s, Inf=>0.0)
        end
        agent.Ri_sa_s = sur_sa_s
        return agent.Ri_sa_s

    elseif reward_type == "information_gain"
        # Compute information gain for each transition (s,a,s')
        # i.e. how much Phat would change (in terms of KL divergence) if we observe transition (s,a,s')
        if agent.model_fixed
            agent.Ri_sa_s[:,:,:] .= 0.0
        else
            for s in 1:agent.n_states
                for a in 1:agent.n_actions_per_state[s]
                    alpha_sa = agent.ε * agent.n_states + agent.C_sa[s,a]
                    for s_prime in 1:agent.n_states
                        alpha_sa_s = agent.ε + agent.C_sa_s[s,a,s_prime]
                        agent.Ri_sa_s[s,a,s_prime] = log((alpha_sa+1)/alpha_sa) + (alpha_sa_s/alpha_sa) * log(alpha_sa_s/(alpha_sa_s+1))
                    end
                end
            end
        end
        
        return agent.Ri_sa_s

    elseif reward_type == "empowerment"
        for s_prime in 1:agent.n_states
            empow = Func_R_sas_Empow(agent, s_prime)
            empow = abs(empow) < eps ? 0.0 : empow # round to zero
            agent.Ri_sa_s[:,:,s_prime] .= empow
        end
        return agent.Ri_sa_s
    
    elseif reward_type == "nov_ig"
        Ri_nov = copy(intrinsic_reward!(agent, "novelty", t, s_prime))
        Ri_ig = copy(intrinsic_reward!(agent, "information_gain", t, s_prime))
        agent.Ri_sa_s = Ri_nov * agent.nov_ig_alpha + Ri_ig * (1-agent.nov_ig_alpha)
        return agent.Ri_sa_s

    elseif reward_type == "MOP"
        logp = log.(agent.Phat_sa_s)
        if agent.model_fixed && any(x->isinf(x), logp)
            # Deal with NaNs when expo is too large for example
            logp = replace(logp, -Inf=>0.0)
        end
        for s in 1:agent.n_states
            logpi = log.(compute_policy(agent, s))
            agent.Ri_sa_s[s,1:agent.n_actions_per_state[s],:] .= (-agent.mop_α .* logpi) .- (agent.mop_β * logp[s,1:agent.n_actions_per_state[s],:])
        end

    elseif reward_type == "SP"
        #update_sp!(agent, agent.st, s_prime)
        for sp in 1:agent.n_states
            for s in 1:agent.n_states
                #sp_M = update_sp(agent, s, sp)
                retro = sum(abs.(agent.sp_M[:,sp]))
                agent.Ri_sa_s[s,:,sp] .= agent.sp_M[s,sp] - retro
            end
        end


    elseif reward_type == "extrinsic"
        # Intrinsic rewards = extrinsic rewards
        # Just cumreward / count for each transition
        agent.Ri_sa_s[agent.st, agent.at, s_prime] = agent.CumReward_sa_s[agent.st, agent.at, s_prime] / agent.C_sa_s[agent.st, agent.at, s_prime]
        return agent.Ri_sa_s
    else
        throw(DomainError(reward_type, "Unknown intrinsic reward type"))
    end
end

"""
PSA!(agent::QAgent, is_intrinsic::Bool)

Update the Q-values and the U-values of the agent using Prioritized Sweeping Algorithm
"""
@views function PSA!(agent::QAgent, is_intrinsic::Bool, ΔV_thresh = 1e-2, θ_thresh = 1e-3)
    
    Phat_sa_s = agent.Phat_sa_s
    # Choose whether to update intrinsic or extrinsic values
    if is_intrinsic
        λ = agent.λ_i
        Qt = agent.Qi_sa
        Ut = agent.Ui_t_s
        R = agent.Ri_sa_s
    else
        λ = agent.λ_e
        Qt = agent.Qe_sa
        Ut = agent.Ue_t_s
        R = agent.Rhat_sa_s
    end

    # Compute R + (λ * U)
    # Set Ut in a correct format to add to R (repeat the vector for every (s,a))
    R_Ut = R + λ * permutedims(repeat(Ut, inner=(1, agent.max_actions, agent.n_states)), [3,2,1])
    # Multiply then sum over s' to get line 4 of alg D of page 11 of https://doi.org/10.1371/journal.pcbi.1009070.s001
    Qt = dropdims(sum(Phat_sa_s .* R_Ut, dims=3), dims=3)

    V = zeros(size(Ut))
    # Making the priority queue
    Prior = zeros(size(Ut))
    for s in 1:agent.n_states
        if agent.intrinsic_type == "MOP"
            pi_a = compute_policy(agent, s)
            V[s] = sum(pi_a .* Qt[s,1:agent.n_actions_per_state[s]])
        else
            V[s] = maximum(Qt[s,1:agent.n_actions_per_state[s]])
        end
        Prior[s] = abs(Ut[s] - V[s])
    end
     
    # Updating U-values for T_PS step
    for i in 1:agent.T_PS
        s_prime = findmax(Prior)[2] # Element with largest priority
        delta_V = V[s_prime] - Ut[s_prime]
        if (abs(delta_V)/abs(maximum(V) - minimum(V))) <= ΔV_thresh
            break
        else
            Ut[s_prime] = V[s_prime]
            # Applying the effect of the update of U-values on Q-values
            for s in 1:agent.n_states
                if (θ_thresh==0)||
                    ((sum(Phat_sa_s[s,:,s_prime]))*abs(delta_V) > θ_thresh)
                    for a in 1:agent.n_actions_per_state[s]
                        Qt[s,a] += λ * Phat_sa_s[s,a,s_prime] * delta_V
                    end
                    if agent.intrinsic_type == "MOP"
                        pi_a = compute_policy(agent, s)
                        V[s] = sum(pi_a .* Qt[s,1:agent.n_actions_per_state[s]])
                    else
                        V[s] = maximum(Qt[s,1:agent.n_actions_per_state[s]])
                    end
                    Prior[s] = abs(Ut[s] - V[s])
                end
            end
        end
    end


    # Update U and Q-values in QAgent
    if is_intrinsic
        agent.Qi_sa = Qt
        agent.Ui_t_s = Ut
    else
        agent.Qe_sa = Qt
        agent.Ue_t_s = Ut
    end

end 

@views function update_sp_alpha(ag::QAgent, st, st1)
    sp_M = copy(ag.sp_M)
    td = ag.λ_i * sp_M[st1,:] - sp_M[st,:]
    td[st] += 1
    sp_M[st,:] .= sp_M[st,:] + ag.sp_α * td
    return sp_M
end

@views function update_sp(ag::QAgent, ΔV_thresh = 1e-2, θ_thresh = 1e-3)
    n = ag.n_states
    P = zeros(n,n) #P(s' from s) = sum_a(pi(a|s) * P(s'|s,a))
    for s in 1:n
        pis = compute_policy(ag, s)
        for s2 in 1:n
            P[s,s2] = sum(pis .* ag.Phat_sa_s[s,1:ag.n_actions_per_state[s],s2])
        end
    end

    λ = ag.λ_i
    # Applying the effect of the latest observation
    M_old = copy(ag.sp_M)
    M = zeros(n,n)
    for s in 1:n, s2 in 1:n
        M[s,s2] = Int(s == s2) + λ * sum(P[s,:] .* M_old[:,s2])
    end

    # Making the priority queue
    Prior = zeros(size(M))
    for s in 1:n, s2 in 1:n        
        Prior[s,s2] = abs(M[s,s2] - M_old[s,s2])
    end
     
    # Updating U-values for T_PS step
    for i in 1:ag.sp_TPS
        id = findmax(Prior)[2] # Element with largest priority
        s3,s2 = id[1], id[2]
        delta_M = M[s3,s2] - M_old[s3,s2]
        if (abs(delta_M)/abs(maximum(M) - minimum(M))) <= ΔV_thresh
            break
        else
            M_old[s3,s2] = M[s3,s2]
            # Applying the effect of the update of U-values on Q-values
            for s1 in 1:n
                if (θ_thresh==0)||
                    (P[s1,s3] * abs(delta_M) > θ_thresh)
                    
                    M[s1,s2] += λ * P[s1,s3] * delta_M
                                        
                    Prior[s1,s2] = abs(M[s1,s2] - M_old[s1,s2])
                end
            end
        end
    end

    return M
end

function fix_model!(agent::QAgent, P_sa_s)
    agent.Phat_sa_s = P_sa_s
    agent.model_fixed = true
    return agent
end


# Deep copy of a QAgent
Base.copy(a::QAgent) = QAgent(a.ε, a.β_i, a.β_e, a.λ_i, a.λ_e, a.T_PS, a.Rhat_0, a.st, copy(a.Qe_sa), copy(a.Qi_sa), copy(a.Ue_t_s), copy(a.Ui_t_s), copy(a.C_sa_s), copy(a.C_sa), copy(a.C_s), copy(a.CumReward_sa_s), copy(a.Ri_sa_s), a.intrinsic_type, copy(a.Phat_sa_s), copy(a.Rhat_sa_s), copy(a.pi_a), a.at, a.n_states, a.n_actions_per_state, a.max_actions, a.update_reward, a.nov_ig_alpha, a.mop_α, a.mop_β, copy(a.mop_V), copy(a.mop_π), copy(a.sp_M), a.sp_α, a.sp_TPS, a.model_fixed)

######################################
################ MOP #################
######################################


function entropy_mop(probabilities)
    return -sum(p * log(p) for p in probabilities if p > 0)
end


# Function to compute the partition function Z(s)
function mop_Z(ag, state, V, H, α, β, γ)
    return sum(exp((α^(-1)) * (β * H[state, action] + γ * sum(ag.Phat_sa_s[state, action, s_prime] * V[s_prime] for s_prime in 1:ag.n_states))) for action in 1:ag.n_actions_per_state[state])
end

# Value iteration
function mop_value_iteration!(ag, V, π, H, α, β, γ; max_iters=1000, tol=1e-6)
    for iter in 1:max_iters
        V_prev = deepcopy(V)
        for s in 1:ag.n_states
            #println(s)
            #println(V[s])
            #println(Z(s))
            V[s] = α * log(mop_Z(ag, s, V, H, α, β, γ))
        end
        #println(V_prev)
        #println(V)
        if maximum(abs.(V_prev .- V)) < tol
            #println("Converged after $iter iterations")
            break
        end
    end
end

# Policy update
function mop_update_policy!(ag, π, V, H, α, β, γ)
    for s in 1:ag.n_states
        Zs = mop_Z(ag, s, V, H, α, β, γ)
        for a in 1:ag.n_actions_per_state[s]
            π[s, a] = exp((α^(-1)) * (β * H[s, a] + γ * sum(ag.Phat_sa_s[s, a, s_prime] * V[s_prime] for s_prime in 1:ag.n_states))) / Zs
        end
    end
end

# Main optimization loop
function mop_optimize_policy!(ag; max_iters=500, v_tol=1e-6, π_tol=1e-6)
    α, β, γ, V, π = ag.mop_α, ag.mop_β, ag.λ_i, ag.mop_V, ag.mop_π

    H = zeros(ag.n_states, maximum(ag.n_actions_per_state))
    for s in 1:ag.n_states, a in 1:ag.n_actions_per_state[s]
        H[s,a] = entropy_mop(ag.Phat_sa_s[s,a,:])
    end

    π_old = copy(π)
    for iter in 1:max_iters
        mop_value_iteration!(ag, V, π, H, α, β, γ, tol=v_tol)
        mop_update_policy!(ag, π, V, H, α, β, γ)
        if maximum(abs.(π_old .- π)) < π_tol

            break
        end
        π_old = copy(π)
    end
    if any(x->isnan(x), π)
        # Deal with NaNs
        π = replace(π, NaN=>1)
        π = π ./ sum(π, dims=2)
    end

    ag.mop_V, ag.mop_π = V, π
end