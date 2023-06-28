"""
Optmial Classification Tree
"""

# preparação do ambiente
using CSV, DataFrames
using JuMP
using Gurobi

# leitura do dataset
df = DataFrame(CSV.File("iris.csv"))
data = Array(df)
X = @view data[:, 2:end-1]
y = @view data[:, end]
for i=1:size(X)[2]
    x = X[:, i]
    m = minimum(x)
    M = maximum(x)
    X[:, i] = (x .- m) ./ (M - m)
end
y_space = unique(y)
y = indexin(y, y_space)

# pai do nó
parent(t) = div(t, 2)

# ancestrais do nó
function anc(t)
    i = parent(t)
    r = [i]
    while i != 1
        i = parent(i)
        push!(r, i)
    end
    return r
end

# ancestrais pela esquerda
function anc_left(i)
    if i == 1
        return []
    else
        if i % 2 == 0
            return [parent(i); anc_left(parent(i))]
        else
            return anc_left(parent(i))
        end
    end
end

# ancestrais pela direita
function anc_right(i)
    if i == 1
        return []
    else
        if i % 2 == 1
            return [parent(i); anc_right(parent(i))]
        else
            return anc_right(parent(i))
        end
    end
end

# modelo OCT
function OCT(X, y, D=2, Nmin=1, α=0.0, timeLimit=600)

    n, p = size(X)
    K = size(unique(y))[1]
    T = 2^(D+1) - 1
    nodes = [i for i=1:T]
    root_nodes = [i for i in nodes if i <= T/2]
    leaf_nodes = [i for i in nodes if i > T/2]
    Tb = size(root_nodes)[1]
    Tl = size(leaf_nodes)[1]

    model = Model(Gurobi.Optimizer)

    # splits
    @variables(model, begin
        a[1:p, 1:Tb], Bin
        d[1:Tb], Bin
        b[1:Tb]
    end)
    @constraints(model, begin
        [t=1:Tb], sum(a[:, t]) == d[t]
        [t=1:Tb], b[t] >= 0
        [t=1:Tb], b[t] <= d[t]
        [t=2:Tb], d[t] <= d[parent(t)]
    end)

    # alocação dos pontos
    @variables(model, begin
        z[1:n, 1:Tl], Bin
        l[1:Tl], Bin
    end)
    @constraints(model, begin
        [i=1:n], sum(z[i, :]) == 1
        [i=1:n, t=1:Tl], z[i,t] <= l[t]
        [i=1:n, t=1:Tl], sum(z[:,t]) >= Nmin*l[t]
    end)

    # ligação splits/folhas
    ϵ = zeros(p)
    for j=1:p
        x = sort(X[:, j])
        e = x[2:end] - x[1:end-1]
        ϵ[j] = minimum(e[e .> 0]) 
    end
    M1 = (1 + maximum(ϵ))
    M2 = 1
    ϵ = minimum(ϵ)
    for i=1:n, t in leaf_nodes
        t_idx = indexin([t], leaf_nodes)[1]
        for m in anc_left(t)
            @constraint(model, a[:, m]'*X[i, :] + ϵ <= b[m] + M1*(1-z[i, t_idx]))
        end
        for m in anc_right(t)
            @constraint(model, a[:, m]'*X[i, :] >= b[m] - M2*(1-z[i, t_idx]))
        end
    end

    # objetivo
    @variables(model, begin
        L[1:Tl] >= 0
        c[1:K, 1:Tl], Bin
    end)
    @expressions(model, begin
        N1[t=1:Tl],  sum(z[:, t])
        N2[k=1:K, t=1:Tl], sum(z[[i for i=1:n if y[i]==k], t])
    end)
    @constraints(model, begin
        [t=1:Tl, k=1:K], L[t] >= N1[t] - N2[k, t] - n*(1-c[k, t])
        [t=1:Tl, k=1:K], L[t] <= N1[t] - N2[k, t] + n*c[k, t]
        [t=1:Tl], sum(c[:, t]) == l[t]
    end)
    L_hat = maximum([count(==(i), y) for i in unique(y)])
    @objective(model, Min, (1/L_hat)*sum(L) + α*sum(d))

    # chamada do solver
    set_attribute(model, "TimeLimit", timeLimit)
    optimize!(model)

    errs = sum(value.(L))
    err_perc = 100 * errs / size(y)[1]
    println("Total de erros: $errs ($err_perc %)")
    
    return a,b,c,d, β, β₀, L, N1, N2
end

# D=1, Nmin=1, α=0
a, b, c, d, β, β₀, L, N1, N2 = OCT(X, y, 1, 1, 0.0);

# D=2, Nmin=1, α=0
a, b, c, d, β, β₀, L, N1, N2 = OCT(X, y, 2, 1, 0.0);

# D=3, Nmin=1, α=0
a, b, c, d, β, β₀, L, N1, N2 = OCT(X, y, 3, 1, 0.0);

# D=3, Nmin=1, α=1
a, b, c, d, β, β₀, L, N1, N2 = OCT(X, y, 3, 1, .2);

# testes utilizando implementação DecisionTree
using DecisionTree

# D=1
model = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
fit!(model, X, y)
100 * sum(predict(model, X) .!= y) / size(y)[1]

# D=2
model = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1)
fit!(model, X, y)
100 * sum(predict(model, X) .!= y) / size(y)[1]

# D=3
model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1)
fit!(model, X, y)
100 * sum(predict(model, X) .!= y) / size(y)[1]