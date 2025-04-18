# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Structure of a Random Agent and functions applicable to it
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

mutable struct RandomAgent
    n_states::Int64                 # Number of states in the environment
    n_actions_per_state::Vector{Int64}      # Number of action for each state (constant)
    st::Int64                       # Current state
    at::Int64                       # Last action taken

    Ri_sa_s::Array{Float64, 3}
    Phat_sa_s::Array{Float64, 3}    # Estimation of the transition probabilities
    C_s::Array{Int64, 1}          # Counts of each state s
    C_sa_s::Array{Int64, 3}       # Counts of each transition (s,a -> s')
    ε::Float64                # Prior for Phat_sa_s (often set to 0)
    extr_reward_encounter_time::Int64 # First time the agent encounters an extrinsic reward
end


"""
RandomAgent(n_actions, n_states, st, ε=0.0)

Constructor for Random Agent
"""
function RandomAgent(n_actions_per_state, n_states, st, ε=0.025)
    max_actions = maximum(n_actions_per_state)
    # Initialize Phat and the counts
    Phat_sa_s = fill(1/n_states, (n_states, max_actions, n_states))
    C_sa_s = zeros(Int64, n_states, max_actions, n_states)
    C_s = zeros(Int64, n_states)
    C_s[st] += 1
    at = 0
    Ri_sa_s = zeros(n_states, max_actions, n_states)
    extr_reward_encounter_time = -1
    return RandomAgent(n_states, n_actions_per_state, st, at, Ri_sa_s, Phat_sa_s, C_s, C_sa_s, ε, extr_reward_encounter_time)
end


"""
sample_action(agent::RandomAgent)

Sample an action at random
"""
function sample_action(agent::RandomAgent)
    # Sample actions uniformly at random
    a = sample(1:agent.n_actions_per_state[agent.st])
    agent.at = a
    return a
end


"""
update!(agent::RandomAgent)

Updates st
"""
function update!(agent::RandomAgent, s_prime::Int64, r::Float64, t::Int64)
    # Update the counts
    agent.C_s[s_prime] += 1
    agent.C_sa_s[agent.st, agent.at, s_prime] += 1
    if agent.extr_reward_encounter_time == -1 && r > 0
        agent.extr_reward_encounter_time = t
    end
    # Update Phat(s,a,s') as counts(s,a,s')+ε divided by counts(s,a) + n_states*ε
    alpha_sa_s = agent.C_sa_s .+ agent.ε
    agent.Phat_sa_s = alpha_sa_s ./ sum(alpha_sa_s, dims=3)
    # If counts(s,a) = 0 -> division by 0 gives NaN
    replace!(agent.Phat_sa_s, NaN=>1/agent.n_states)
    # Update current state
    agent.st = s_prime
end

# Deep copy of a Random Agent
Base.copy(ag::RandomAgent) = RandomAgent(ag.n_states, ag.n_actions_per_state, ag.st, ag.at, copy(ag.Ri_sa_s), copy(ag.Phat_sa_s), copy(ag.C_s), copy(ag.C_sa_s), ag.ε, ag.extr_reward_encounter_time)
