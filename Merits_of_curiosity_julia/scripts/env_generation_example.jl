# Add the src directory to the LOAD_PATH so Julia can find the package
push!(LOAD_PATH, "src")

using Random
using Printf
using Graphs
using Merits_of_curiosity_julia

# ------------------------------------------------------------------------------
# The script below demonstrates how to render the environment
# ------------------------------------------------------------------------------

function main()
    # Instantiate the environment with specified parameters
    env = RoomEnvironment(
        n_init_states=13,
        branching_rate=0.5,
        room_size=3,
        p_room=4 / 13,
        p_sink=0.25,
        p_source=0.25,
        p_stoc=0.25,
        n_edges_per_sink=10,
        n_edges_per_source=10,
        uncontrollability=1.0
    )

    # Reset the environment to start a new episode
    reset!(env)
    println("Initial state: $(env.state)")
    done = false
    step_count = 0
    max_steps = 50  # Maximum number of steps to run
    while !done && step_count < max_steps
        # Get valid actions (assuming it's the number of neighbors)
        neighbors = outneighbors(env.g, env.state)
        if isempty(neighbors)
            println("No valid actions from state $(env.state).")
            break
        end
        action = rand(1:length(neighbors))  # Random valid action index

        # Take a step in the environment
        next_state, reward = step!(env, action)
        println(@sprintf("Step %d: Action %d, State %d", step_count, action, next_state))

        # Optionally render the environment (if render method is defined)
        # Uncomment the following line if `render` is implemented
        #render(env)

        step_count += 1
    end

    # Close the environment at the end
    # Uncomment if `close` method is implemented
    # env.close()
    
end

main()

