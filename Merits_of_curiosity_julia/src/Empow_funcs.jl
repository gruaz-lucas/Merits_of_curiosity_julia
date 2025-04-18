# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Functions used for the Blahut–Arimoto algorithm
# See https://en.wikipedia.org/wiki/Blahut-Arimoto_algorithm
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_blahut_arimoto(P_YX; thresh = 1e-6, max_iter = 100,
    pass_all = false)
    # P_YX: the conditional probabilities of y (state s') given x (action a) for 
    #       a given state s
    # In the RL setting: P_YX[i,j] is equal to P(s' = j | s, a = i)
    N = size(P_YX)[1]   # size of alphabets fo x (= number of actions)
    M = size(P_YX)[2]   # size of alphabets fo y (= number of states)

    r = [ones(N) ./ N]  # Initializiation of r (= prior policy)
    c = [0.]

    qi = zeros(M,N)
    ri = zeros(N)

    inds = zeros(Bool, M)
    logr = zeros(N)

    for i = 1:max_iter
        for m = 1:M
            qi[m,:] = @views r[end] .* P_YX[:,m]
            if sum(@view qi[m,:]) == 0
                qi[m,:] .= 1 / N
            else
                qi[m,:] .= @views qi[m,:] ./ sum(qi[m,:])
            end
        end

        for n=1:N
            inds[:] .= @views (qi[:, n] .!= 0)
            ri[n] = @views prod( qi[inds, n] .^ P_YX[n,inds])
        end
        ri[:] = @views ri ./ sum(ri)

        tolerance = @views sum((ri - r[end]).^2)

        if pass_all
            push!(r, ri)
            push!(c, Func_Capacity(P_YX,ri,qi))
        else
            r[1] = ri
            #c[1] = Func_Capacity(P_YX,ri,qi)
            # try something

            for n = 1:N
                inds[:] .= @views (P_YX[n,:] .!= 0)
                #temp1 = P_YX[n,inds]
                #temp2 = q[inds, n]
                logr[n] = @views sum(P_YX[n,inds] .* log.( qi[inds, n] ./ ri[n] ) )
            end
            c[1] = sum(ri .* logr)
        end
        if tolerance < thresh
            break
        end
    end
    return r,c
end
export Func_blahut_arimoto


function Func_Capacity(P_YX,
r::Array{Float64,1},q::Array{Float64,2})
N = size(P_YX)[1]   # size of alphabets fo x (= number of actions)
M = size(P_YX)[2]   # size of alphabets fo y (= number of states)
logr = zeros(N)
for n = 1:N
inds = @views (P_YX[n,:] .!= 0)
#temp1 = P_YX[n,inds]
#temp2 = q[inds, n]
logr[n] = @views sum(P_YX[n,inds] .* log.( q[inds, n] ./ r[n] ) )
end
c = @views sum(r .* logr)
end
export Func_Capacity

function Func_Capacity(P_YX,r::Array{Float64,1})
N = size(P_YX)[1]   # size of alphabets fo x (= number of actions)
M = size(P_YX)[2]   # size of alphabets fo y (= number of states)

P_Y = @views (r' * P_YX)[:]
H_Y = - @views sum(P_Y[P_Y .!= 0] .* log.(P_Y[P_Y .!= 0]))
H_YX = zeros(N)
for n = 1:N
P_YX_n = @views P_YX[n,:]
H_YX[n] = @views - sum(P_YX_n[P_YX_n .!= 0] .* log.(P_YX_n[P_YX_n .!= 0]))
end
H_YX = @views sum(r .* H_YX)
c = H_Y - H_YX
end
export Func_Capacity

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Empowerment
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Func_R_sas_Empow computes the empowerment of state sp as the intrinsic reward 
# of transition for (s,a) → sp for an agent Agent.
function Func_R_sas_Empow(agent,sp; test_if_sp_dep=false,
         test_if_limited_2_sa=false,
         thresh = 1e-6, max_iter = 100)
if test_if_sp_dep
# Indicating whether the reward function depends on sp
return true
end
if test_if_limited_2_sa
# Indicating whether the reward function should be updated only for (s_t,a_t)
return true
end
#State_Set = agent.State_Set     # set of all states
#state_num = agent.state_num     # set of all actions
n_actions_sp = agent.n_actions_per_state[sp]

#sp_ind = findfirst(isequal(sp),State_Set) # index of sp in the state set
P_YX = @view agent.Phat_sa_s[sp,1:n_actions_sp,:]            # P(:| s = sp, a = :)
r,c = Func_blahut_arimoto(P_YX; thresh=thresh, max_iter=max_iter,
     pass_all=false)
return c[end]

end
export Func_R_sas_Empow
