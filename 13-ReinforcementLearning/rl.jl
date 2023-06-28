"""
Reinforcemente Learning - Markov Decision Processes
"""

# preparando o ambiente
using POMDPs, DiscreteValueIteration
using POMDPModelTools
using QuickPOMDPs, POMDPSimulators
using Random, Distributions, Plots

α = 0.80
β = 0.70
r_search = 3
r_wait = 1

states = ["high", "low"]
actions = ["search", "wait", "recharge"]

m = QuickMDP(
    states = states,
    actions = actions,
    discount = 0.95,
    initialstate=Deterministic("low"),
    transition = function (s, a)
        if s == "high" && a == "search"
            return SparseCat(["high", "low"], [α, 1-α])
        elseif s == "low" && a == "search"
            return SparseCat(["high", "low"], [1-β, β])
        elseif s == "high" && a == "wait"
            return Deterministic("high")
        elseif s == "low" && a == "wait"
            return Deterministic("low")
        elseif a == "recharge"
            return Deterministic("high")
        end
    end,
    reward = function (s, a, sp)
        if s == "high" && a == "search"
            return r_search
        elseif s == "low" && a == "search" && sp == "high"
            return -3
        elseif s == "low" && a == "search" && sp == "low"
            return r_search
        elseif a == "wait"
            return r_wait
        elseif a == "recharge"
            return 0
        else
            return 0
        end
    end,
)

solver = ValueIterationSolver()
policy = solve(solver, m)

for a in actions, s in states
    @show a, s
    @show action(policy, s)
    @show value(policy, s)
    @show value(policy, s, a)
    println()
end

sims = []
actions = []
N = 1000
for sim in 1:N
    rsum = 0.0
    act = []
    for (s,a,r) in stepthrough(m, policy, "s,a,r", max_steps=100)
        rsum += r
        push!(act, a)
    end
    push!(sims, rsum)
    push!(actions, act)
end

mean(sims), std(sims)

histogram(sims, label="", 
title="Distribuição do score total em 1000 simulações")
savefig("fig1.png")