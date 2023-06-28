"""
Implementação do SVM
"""

# preparando o ambiente
using Random, Distributions
using MLUtils
using CSV, DataFrames
using JuMP, HiGHS

Random.seed!(0)

# geração de dados
n, m = 3000, 20
X = rand(Normal(0, 1), (n, m))
y = ones(n)
y[
    ((X[:, 1] .>= 0.3) .&& (X[:, 2] .<= 0.7)) .|
    ((X[:, 5] .<= 0.4) .&& (X[:, 2] .<= 0.5)) .|
    (X[:, 6] .>= 0.8)
] .= -1
# split treino teste
train_obs, test_obs = splitobs(shuffleobs(1:size(X)[1]), at=0.7)
X_train = @view X[train_obs, :]
X_test = @view X[test_obs, :]
y_train = @view y[train_obs]
y_test = @view y[test_obs]

# formulação
"""
Implementação do SVM com modelo JuMP

Parâmetros
----------
X: Matrix(n,m){Float}
y: Vector(n){Int}
c: Float
k: Int
"""
function primal_svm(X, y, c, k, M=10)
    
    n, m = size(X)

    svm = Model(Gurobi.Optimizer)
    @variables(svm, begin
        w[1:m]
        ϵ[1:n] ≥ 0
        b
    end)
    @constraint(svm, y.*(X*w .+ b) .≥ 1 - ϵ)
    # ||w||₀ ≤ k
    @variable(svm, l[1:m], Bin)
    @constraints(svm, begin
        [p=1:m], M*l[p] ≥ w[p]
        [p=1:m], M*l[p] ≥ -w[p]
        sum(l) ≤ k
    end)

    obj = 0.5*sum(w.^2) + c*sum(ϵ)
    @objective(svm, Min, obj)
    optimize!(svm)

    return value.(w), value(b), value.(l), value.(ϵ)

end

# treino
w, b, l, ϵ = primal_svm(X_train, y_train, 1, 4);

# escolha de features
w

# acuracia em treino
mean(sign.(X_train*w .+ b) .== y_train)
# acuracia em teste
mean(sign.(X_test*w .+ b) .== y_test)
