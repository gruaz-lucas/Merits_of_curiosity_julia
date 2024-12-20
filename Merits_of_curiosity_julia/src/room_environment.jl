
using Random
using Graphs
using GraphPlot
using Colors
using Distributions
using DataStructures
using NetworkLayout
using StatsBase
using PyPlot

# ---------------------------------------------------------------------------- #
# Utils
# ---------------------------------------------------------------------------- #

"""
Returns true with probability p.
"""
function rand_prob(p::Float64)::Bool
    return rand() < p
end

"""
Removes and returns a random element from the array `v`.
"""
function pop_random!(v::Vector)
    idx = rand(1:length(v))
    return popat!(v, idx)
end

"""
Return the number of neighbors of s. s can be a single state or a room (grid)
"""
function n_neighbors(graph, s)
    if s isa Grid
        return 4-length(s.boundaries)
    else
        return length(outneighbors(graph,s))
    end
end

"""
Grid structure representing a room in the environment.
"""
mutable struct Grid
    node_ids::Vector{Int}
    boundaries::Vector{Int}
    is_sink::Bool
    is_source::Bool
    is_stoc::Bool

    function Grid(node_ids, boundaries; is_sink=false, is_source=false, is_stoc=false)
        new(node_ids, boundaries, is_sink, is_source, is_stoc)
    end
end

"""
RoomEnvironment: A custom environment simulating a graph-based room structure.
"""
mutable struct RoomEnvironment
    n_init_states::Int
    branching_rate::Float64
    room_size::Int
    p_room::Float64
    p_sink::Float64
    p_source::Float64
    p_stoc::Float64
    n_edges_per_sink::Int
    n_edges_per_source::Int
    uncontrollability::Float64
    grids::Vector{Grid}
    g::SimpleDiGraph
    x_coords::Vector{Float64}
    y_coords::Vector{Float64}
    sink_edges::Vector{Tuple{Int, Int}}
    source_edges::Vector{Tuple{Int, Int}}
    transition_matrix::Array{Float64, 3}
    reward_matrix::Array{Float64, 3}
    n_actions_per_state::Vector{Int}
    max_actions::Int
    n_tot_states::Int
    stoc_v::Vector{Bool}
    state::Int

    function RoomEnvironment(; 
        n_init_states::Int, branching_rate::Float64, room_size::Int,
        p_room::Float64, p_sink::Float64, p_source::Float64, p_stoc::Float64,
        n_edges_per_sink::Int, n_edges_per_source::Int, uncontrollability::Float64
    )
        grids = Grid[]
        g = SimpleDiGraph()

        env = new(
            n_init_states, branching_rate, room_size, p_room, p_sink, p_source, 
            p_stoc, n_edges_per_sink, n_edges_per_source, uncontrollability, grids, g, 
            Float64[], Float64[], Tuple{Int, Int}[], Tuple{Int, Int}[],
            Array{Float64}(undef, 0, 0, 0), Array{Float64}(undef, 0, 0, 0), Int[], 0, 0, Bool[], 0
        )

        generate_graph!(env)
        @assert env.n_tot_states == nv(env.g)
        env.n_actions_per_state = [length(outneighbors(env.g,s)) for s in 1:env.n_tot_states]
        env.max_actions = maximum(env.n_actions_per_state)
        env.transition_matrix, env.reward_matrix = generate_transition_matrix!(env)
        return env
    end
end

function reset!(env::RoomEnvironment)
    env.state = rand(1:env.n_tot_states)
end

function step!(env::RoomEnvironment, action::Int)
    probabilities = env.transition_matrix[env.state, action, :]
    next_state = sample(1:env.n_tot_states, pweights(probabilities))
    reward = env.reward_matrix[env.state, action, next_state]
    env.state = next_state
    return next_state, reward
end

function render(env::RoomEnvironment)
    colors = PyPlot.cm.Accent.colors
    room_colors=Dict(
        "sink"=>RGB(colors[1]...),
        "source"=>RGB(colors[2]...),
        "stoc"=>RGB(colors[3]...),
        "neutral"=>RGB(colors[4]...),
        "corridor"=>RGB(colors[5]...),
    )
    # Compute node properties
    node_rooms = compute_node_properties(env)
    node_colors = [room_colors[node_rooms[i+1]] for i in 0:env.n_tot_states-1]
    NODESIZE = 1 / (3*sqrt(env.n_tot_states))
    node_sizes = [1.0 for _ in 0:env.n_tot_states-1]

    # Highlight the current state
    node_colors[env.state+1] = RGB(1,0,0)
    node_sizes[env.state+1] *= 1.5

    # Prepare edge colors
    edge_list = collect(edges(env.g))
    base_colors = fill("gray", ne(env.g))
    edge_width = 2.0 / sqrt(env.n_tot_states)

    # Assign colors to sink edges
    sink_edge_indices = findall(e -> e in env.sink_edges, edge_list)
    for idx in sink_edge_indices
        base_colors[idx] = room_colors["sink"]
    end

    # Assign colors to source edges
    source_edge_indices = findall(e -> e in env.source_edges, edge_list)
    for idx in source_edge_indices
        base_colors[idx] = room_colors["source"]
    end

    # Create labels for nodes
    labels = Dict(s => string(s) for s in 0:env.n_tot_states-1)
    font_size = 50.0 / sqrt(env.n_tot_states)
    
    base_colors = fill(colorant"white", ne(env.g))
    if ne(env.g) > 100
        # For some reason edge colors are not displayed properly when the number of edges is large
        base_colors = "gray"
    end
    gplot(env.g, env.x_coords, env.y_coords, nodefillc=node_colors, edgestrokec=base_colors, nodesize=node_sizes, EDGELINEWIDTH=edge_width, arrowlengthfrac=0.05, NODESIZE=NODESIZE)
end

function generate_graph!(env::RoomEnvironment)
    generate_base_structure!(env)
    assign_grid_properties!(env)
    env.x_coords, env.y_coords = generate_coordinates(env)
    env.sink_edges, env.source_edges = add_sink_source_edges!(env)
    compute_stochastic_nodes!(env)
end

function add_grid!(env, previous_node, start_node; is_sink=false, is_source=false, is_stoc=false)
    """
    Create a grid room in the graph g, connected to previous_node, starting from start_node (first node id in the grid is start_node).
    """
    n = env.room_size
    node_ids = Int[]
    boundaries = Int[]

    # Iterate over rows and columns to create nodes and connect them
    for i in 1:n
        for j in 1:n
            # i is row, j is column
            state = start_node + (i - 1) * n + (j - 1)
            t = start_node + (i - 2) * n + (j - 1)  # Top state
            d = start_node + i * n + (j - 1)  # Down state
            r = state + 1                     # Right state
            l = state - 1                     # Left state

            # Connect up, right, down, left directions if within bounds
            if i != 1
                add_edge!(env.g, state, t)
            end
            if j != n
                add_edge!(env.g, state, r)
            end
            if i != n
                add_edge!(env.g, state, d)
            end
            if j != 1
                add_edge!(env.g, state, l)
            end

            push!(node_ids, state)

            # Connect to the previous node at the middle of the top boundary
            if i == 1 && j == div(n - 1, 2) + 1 && previous_node !== nothing
                add_edge!(env.g, state, previous_node)
                add_edge!(env.g, previous_node, state)
            end

            # Track boundaries
            if (i == n && j == div(n - 1, 2) + 1) || 
            (previous_node === nothing && i == 1 && j == div(n - 1, 2) + 1) ||
            ((j == 1 || j == n) && i == div(n - 1, 2) + 1)
                push!(boundaries, state)
            end
        end
    end

    # Create a new Grid object and append it to grids
    grid = Grid(node_ids, boundaries; is_sink=is_sink, is_source=is_source, is_stoc=is_stoc)
    push!(env.grids, grid)

    # Return the grid and the id of the last state in the grid
    last_state = start_node + (n - 1) * n + (n - 1)
    return grid, last_state
end


function generate_base_structure!(env::RoomEnvironment)
    """
    Create the graph structure with branches and grid rooms.
    No grid properties are assigned at this stage.
    """
    # Calculate the number of rooms to create
    n_rooms = Int(floor(env.n_init_states * env.p_room))
    room_ids = sample(1:env.n_init_states, n_rooms, replace=false)  # IDs of nodes that will be transformed into rooms
    env.n_tot_states = env.n_init_states + n_rooms * (env.room_size^2 - 1)
    env.grids = []
    env.g = DiGraph()
    add_vertices!(env.g, env.n_tot_states)
    
    D = Deque{Any}()
    push!(D,nothing) # Initialize deque with a `nothing` value
    next_node = 1  # Next node id to be used

    for i in 1:env.n_init_states
        s = popfirst!(D)
        
        while isa(s, Grid) && length(s.boundaries) == 0
            # Drop grid with no boundaries
            if isempty(D)
                s = nothing
                break
            end
            s = popfirst!(D)
        end

        # With some probability, put it back in the queue (if not already full neighbors)
        if rand_prob(env.branching_rate) && s != nothing && n_neighbors(env.g, s) < 3
            pushfirst!(D, s)
        end

        if isa(s, Grid)
            if length(s.boundaries) == 0
                continue
            end
            cur_node = pop_random!(s.boundaries)
        else
            cur_node = s
        end

        if i in room_ids
            # Add a grid room
            grid, last_state = add_grid!(env, cur_node, next_node)
            push!(D, grid)
            next_node = last_state + 1
        else
            # Connect to a new node if not part of a room
            if s != nothing
                add_edge!(env.g, cur_node, next_node)
                add_edge!(env.g, next_node, cur_node)
            end
            push!(D, next_node)
            next_node += 1
        end
    end
    
    return
end


function assign_grid_properties!(env::RoomEnvironment)
    n_rooms = length(env.grids)
    n_sink = Int(round(n_rooms * env.p_sink))
    n_source = Int(round(n_rooms * env.p_source))
    n_stoc = Int(round(n_rooms * env.p_stoc))

    sink_ids = rand(1:n_rooms, n_sink)
    source_ids = setdiff(1:n_rooms, sink_ids) |> x -> rand(x, n_source)
    stoc_ids = setdiff(1:n_rooms, union(sink_ids, source_ids)) |> x -> rand(x, n_stoc)

    for i in sink_ids
        env.grids[i].is_sink = true
    end
    for i in source_ids
        env.grids[i].is_source = true
    end
    for i in stoc_ids
        env.grids[i].is_stoc = true
    end
end

function generate_coordinates(env::RoomEnvironment)
    layout = Stress()
    points = layout(env.g)
    x_coords = [p[1] for p in points]
    y_coords = [p[2] for p in points]
    return x_coords, y_coords
end

function add_random_edges!(g, from, to, n_e)
    n_added=0
    added_edges = []
    while n_added < n_e
        sour = rand(from)
        dest = rand(to)
        if !has_edge(g, sour, dest)
            add_edge!(g, sour, dest)
            n_added += 1
            push!(added_edges, (sour, dest))
        end
    end
    return added_edges
end

function add_sink_source_edges!(env::RoomEnvironment)
    sink_edges_vec = []
    source_edges_vec = []
    g = env.g
    for grid in env.grids
        if grid.is_sink
            from = setdiff(vertices(g), grid.node_ids)
            to = grid.node_ids
            n_e = env.n_edges_per_sink
            edges = add_random_edges!(g, from, to, n_e)
            append!(sink_edges_vec, edges)
        end
        if grid.is_source
            to = setdiff(vertices(g), grid.node_ids)
            from = grid.node_ids
            n_e = env.n_edges_per_source
            edges = add_random_edges!(g, from, to, n_e)
            append!(source_edges_vec, edges)
        end
    end
    
    return sink_edges_vec, source_edges_vec
end

function compute_stochastic_nodes!(env::RoomEnvironment)
    env.stoc_v = falses(env.n_tot_states)
    for grid in env.grids
        if grid.is_stoc
            env.stoc_v[grid.node_ids] .= true
        end
    end
end

function generate_transition_matrix!(env::RoomEnvironment)
    Psa_s = zeros(env.n_tot_states, env.max_actions, env.n_tot_states)
    Rsa_s = zeros(env.n_tot_states, env.max_actions, env.n_tot_states)

    for u in 1:env.n_tot_states
        neighbors = outneighbors(env.g, u)
        for ia in 1:length(neighbors)
            v = neighbors[ia]
            if env.stoc_v[u]
                prob = env.uncontrollability / length(neighbors)
                Psa_s[u, ia, neighbors] .= prob
                Psa_s[u, ia, v] += 1.0 - env.uncontrollability
            else
                Psa_s[u, ia, v] = 1.0
            end
            Rsa_s[u, ia, v] = 0.0
        end
    end

    return Psa_s, Rsa_s
end

function compute_node_properties(env)
    """
    Compute properties of nodes in the graph.
    """
    props = ["corridor" for i in 1:env.n_tot_states]  # Default corridor
    for grid in env.grids
        grid_prop = "neutral"
        if grid.is_stoc
            grid_prop = "stoc"
        elseif grid.is_sink
            grid_prop = "sink"
        elseif grid.is_source
            grid_prop = "source"
        end
        for n in grid.node_ids
            props[n] = grid_prop
        end
    end
    return props
end

export RoomEnvironment, reset!, step!, render
