"""
MCTS solver type

Fields:

    n_iterations::Int64
        Number of iterations during each action() call.
        default: 100

    max_time::Float64
        Maximum amount of CPU time spent iterating through simulations.
        default: Inf

    depth::Int64:
        Maximum rollout horizon and tree depth.
        default: 10

    exploration_constant::Float64:
        Specifies how much the solver should explore.
        In the UCB equation, Q + c*sqrt(log(t/N)), c is the exploration constant.
        default: 1.0

    rng::AbstractRNG:
        Random number generator

    estimate_value::Any (rollout policy)
        Function, object, or number used to estimate the value at the leaf nodes.
        If this is a function `f`, `f(mdp, s, remaining_depth)` will be called to estimate the value (remaining_depth can be ignored).
        If this is an object `o`, `estimate_value(o, mdp, s, remaining_depth)` will be called.
        If this is a number, the value will be set to that number
        default: RolloutEstimator(RandomSolver(rng); max_depth=50, eps=nothing)

    init_Q::Any
        Function, object, or number used to set the initial Q(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_Q(o, mdp, s, a)` will be called.
        If this is a number, Q will be set to that number
        default: 0.0

    init_N::Any
        Function, object, or number used to set the initial N(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_N(o, mdp, s, a)` will be called.
        If this is a numERMCTSSolverber, N will be set to that number
        default: 0

    reuse_tree::Bool:
        If this is true, the tree information is re-used for calculating the next plan.
        Of course, clear_tree! can always be called to override this.
        default: false

    enable_tree_vis::Bool:
        If this is true, extra information needed for tree visualization will
        be recorded. If it is false, the tree cannot be visualized.
        default: false

    timer::Function:
        Timekeeping method. Search iterations ended when `timer() - start_time ≥ max_time`.
"""
mutable struct ERMCTSSolver <: AbstractMCTSSolver
    n_iterations::Int64
    max_time::Float64
    depth::Int64
    exploration_constant::Float64
    λ::Float64
    rng::AbstractRNG
    estimate_value::Any
    init_Q::Any
    init_N::Any
    reuse_tree::Bool
    enable_tree_vis::Bool
    timer::Function
end

"""
    ERMCTSSolver()

Use keyword arguments to specify values for the fields.
"""
function ERMCTSSolver(;n_iterations::Int64=100,
                     max_time::Float64=Inf,
                     depth::Int64=10,
                     exploration_constant::Float64=1.0,
                     λ::Float64=0.1,
                     rng=Random.GLOBAL_RNG,
                     estimate_value=RolloutEstimator(RandomSolver(rng)),
                     init_Q=0.0,
                     init_N=0,
                     reuse_tree::Bool=false,
                     enable_tree_vis::Bool=false,
                     timer=() -> 1e-9 * time_ns())
    return ERMCTSSolver(n_iterations, max_time, depth, exploration_constant, λ, rng, estimate_value, init_Q, init_N,
                      reuse_tree, enable_tree_vis, timer)
end

mutable struct ERMCTSPlanner{P<:Union{MDP,POMDP}, S, A, SE, RNG} <: AbstractMCTSPlanner{P}
	solver::ERMCTSSolver # containts the solver parameters
	mdp::P # model
    tree::Union{Nothing,MCTSTree{S,A}} # the search tree
    solved_estimate::SE
    rng::RNG
end

function ERMCTSPlanner(solver::ERMCTSSolver, mdp::Union{POMDP,MDP})
    # tree = Dict{statetype(mdp), StateNode{actiontype(mdp)}}()
    tree = MCTSTree{statetype(mdp), actiontype(mdp)}(solver.n_iterations)
    se = convert_estimator(solver.estimate_value, solver, mdp)
    return ERMCTSPlanner(solver, mdp, tree, se, solver.rng)
end

"""
Delete existing decision tree.
"""
function clear_tree!(p::ERMCTSPlanner{S,A}) where {S,A} p.tree = nothing end


# no computation is done in solve - the solver is just given the mdp model that it will work with
POMDPs.solve(solver::ERMCTSSolver, mdp::Union{POMDP,MDP}) = ERMCTSPlanner(solver, mdp)


function lse(planner::ERMCTSPlanner, x)
    λ = planner.solver.λ
    return λ*logsumexp(x./λ)
end

function simulate(planner::ERMCTSPlanner, node::StateNode, depth::Int64)
    mdp = planner.mdp
    rng = planner.rng
    s = state(node)
    tree = node.tree

    # once depth is zero return
    if isterminal(planner.mdp, s)
        return 0.0
    elseif depth == 0
        return estimate_value(planner.solved_estimate, planner.mdp, s, depth)
    end

    # pick action using UCT
    sanode = best_sanode_UCB(node, planner.solver.exploration_constant)
    said = sanode.id

    # transition to a new state
    sp, r = @gen(:sp, :r)(mdp, s, action(sanode), rng)

    spid = get(tree.state_map, sp, 0)
    if spid == 0
        spn = insert_node!(tree, planner, sp)
        spid = spn.id
        q = r + discount(mdp) * estimate_value(planner.solved_estimate, planner.mdp, sp, depth-1)
    else
        q = r + discount(mdp) * lse(planner, simulate(planner, StateNode(tree, spid) , depth-1))
    end
    if planner.solver.enable_tree_vis
        record_visit!(tree, said, spid)
    end

    tree.total_n[node.id] += 1
    tree.n[said] += 1
    tree.q[said] += (q - tree.q[said]) / tree.n[said] # moving average of Q value
    #return q
    #qs = map(a->q(a), children(node))
    qs = map(child->node.tree.q[child.id], children(node))
    return qs
end


function insert_node!(tree::MCTSTree, planner::ERMCTSPlanner, s)
    push!(tree.s_labels, s)
    tree.state_map[s] = length(tree.s_labels)
    push!(tree.child_ids, [])
    total_n = 0
    for a in actions(planner.mdp, s)
        n = init_N(planner.solver.init_N, planner.mdp, s, a)
        total_n += n
        push!(tree.n, n)
        push!(tree.q, init_Q(planner.solver.init_Q, planner.mdp, s, a))
        push!(tree.a_labels, a)
        push!(last(tree.child_ids), length(tree.n))
    end
    push!(tree.total_n, total_n)
    return StateNode(tree, length(tree.total_n))
end


"""
Return the best action node based on the UCB score with exploration constant c
"""
function best_sanode_UCB(snode::StateNode, c::Float64)
    if c==0
        return argmax(q, children(snode))
    end

    best_UCB = -Inf
    best = first(children(snode))
    sn = total_n(snode)
    for sanode in children(snode)
        # if action was not used, use it. This also handles the case sn==0, 
        # since sn==0 is possible only when for all available actions n(sanode)==0
        if n(sanode) == 0
            return sanode
        else
            UCB = q(sanode) + c*sqrt(log(sn)/n(sanode))
        end
		
        # if isnan(UCB)
        #     @show sn
        #     @show n(sanode)
        #     @show q(sanode)
        # end
		
        # @assert !isnan(UCB)
        # @assert !isequal(UCB, -Inf)
		
        if UCB > best_UCB
            best_UCB = UCB
            best = sanode
        end
    end
    return best
end
