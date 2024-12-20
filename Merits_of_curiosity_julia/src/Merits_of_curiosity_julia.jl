module Merits_of_curiosity_julia
    include("room_environment.jl")
    include("QAgent.jl")
    include("simulation.jl")
    export RoomEnvironment, reset!, step!, render
    export QAgent, update!, sample_action
    export create_intr_ag_from_env, run_step, partial_measure
end # module Merits_of_curiosity_julia
