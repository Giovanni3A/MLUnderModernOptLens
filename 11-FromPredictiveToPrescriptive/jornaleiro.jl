"""
Lista 12 - From Predictive to Prescriptive Analysis
"""

# preparando o ambiente
include("exercicio.jl")
using Plots
using ProgressBars

const GUROBI_ENV = Gurobi.Env()

"""
Função geradora de dados a partir da lógica passada para o processo
ARMA(2,2) e uma lógica diferente para geração de valores de demanda relacionados.
"""
function generate_data(n)
    X = [X1'; X2']
    Y = []
    for i=1:n
        # calculo do x
        U = rand(MultivariateNormal(μ, ΣU))
        U1 = rand(MultivariateNormal(μ, ΣU))
        U2 = rand(MultivariateNormal(μ, ΣU))
        x = Φ1*X[end, :] + Φ2*X[end-1, :] + U + Θ1*U1 + Θ2*U2
        X = [X; x']
        # calculo do y e demanda
        # δ = rand(Normal(0, 0.1), dx)
        # ϵ = rand(Normal(0, 0.1))
        # y = A*(x + δ/4) + B*x*ϵ
        # y = maximum([0; y[1]])
        # d = 50 + 100*y[1]
        d = round.(130 .+ [1.5 -3 1.5]*x)
        Y = [Y; d]
    end
    # remover 2 valores iniciais usados para o autoregressivo
    X = X[3:end, :]
    
    return X, Y
end

# gerar cenários a partir dos dados passados
X, Y = generate_data(S)
histogram(Y, title="Distribuição da demanda nos 1000 cenários gerados", label="")
savefig("y.png")

# a) resolva o modelo de 2 estágios
# vamos solucionar o problema utilizando a técnica SAA,
# de aproximação do custo esperado pela média em Y
saa_model = Model(Gurobi.Optimizer)
@variables(saa_model, begin
    x_saa ≥ 0
    y_saa[1:S] ≥ 0
    z_saa[1:S] ≥ 0
end)
@constraints(saa_model, begin
    x_saa ≤ u
    y_saa .≤ Y
    y_saa + z_saa .≤ x_saa
end)
@objective(saa_model, Min, c*x_saa - 1/S*sum(q*y_saa + r*z_saa))
optimize!(saa_model)
value(x_saa)
objective_value(saa_model)

# b) gere 100 novos X e um novo Y
X_new, Y_new = generate_data(100)

# c) encontre o custo levando em conta o novo cenário e o x do saa_model
saa_model2 = Model(Gurobi.Optimizer)
@variables(saa_model2, begin
    y_saa2[1:100] ≥ 0
    z_saa2[1:100] ≥ 0
end)
@constraints(saa_model2, begin
    y_saa2 .≤ Y_new
    y_saa2 + z_saa2 .≤ value(x_saa)
end)
@objective(saa_model2, Min, c*value(x_saa) - 1/100*sum(q*y_saa2 + r*z_saa2))
optimize!(saa_model2)
objective_value(saa_model2)

# d) resolva o problema com o KNN para k = 5
k = 5
# calcular árvore
kdtree = KDTree(X');

"""
Função de solução do problema do jornaleiro para uma observação x dada
"""
function newsvendor_knn(x, Y, kdtree, k)

    neighbors = knn(kdtree, x, k, true)[1]

    knn_model = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GUROBI_ENV)
        )
    )
    MOI.set(knn_model, MOI.Silent(), true)
    @variables(knn_model, begin
        x_knn ≥ 0
        y_knn[1:k] ≥ 0
        z_knn[1:k] ≥ 0
    end)
    @constraints(knn_model, begin
        x_knn ≤ u
        y_knn .≤ Y[neighbors]
        y_knn + z_knn .≤ x_knn
    end)
    @objective(knn_model, Min, c*x_knn - 1/k*sum(q*y_knn + r*z_knn))
    optimize!(knn_model)
    return x_knn, knn_model
end

costs = []
for i=1:100
    x_knn, knn_model = newsvendor_knn(X_new[i,:], Y, kdtree, k)
    push!(costs, objective_value(knn_model))
end
mean(costs)

# e) varie k entre 3 e 15, resolva o problema com o knn para cada k
# faça o gráfico da evolução dos custos e dos k Y mais próximos conforme aumentamos k
costs = []
for k in ProgressBar(3:15)
    kcosts = []
    for i=1:100
        x_knn, knn_model = newsvendor_knn(X_new[i,:], Y, kdtree, k)
        push!(kcosts, objective_value(knn_model))
    end
    push!(costs, kcosts)
end


boxplot([3], label="", costs[1], color="grey")
boxplot!([4], label="", costs[2], color="grey")
boxplot!([5], label="", costs[3], color="grey")
boxplot!([6], label="", costs[4], color="grey")
boxplot!([7], label="", costs[5], color="grey")
boxplot!([8], label="", costs[6], color="grey")
boxplot!([9], label="", costs[7], color="grey")
boxplot!([10],label="",  costs[8], color="grey")
boxplot!([11],label="",  costs[9], color="grey")
boxplot!([12],label="",  costs[10], color="grey")
boxplot!([13],label="",  costs[11], color="grey")
boxplot!([14],label="",  costs[12], color="grey")
boxplot!([15],label="",  costs[13], color="grey")
title!("Custo das 100 novas amostras para valores de k")
savefig("e1.png")

# f) calcule p para uma nova amostra (200 cenários) e para k=13
X_new, Y_new = generate_data(200)

# cenário com informação perfeita
# compra-se exatamente a quantidade demandada (ou 150, caso ultrapasse)
decision1 = [minimum([y, 150]) for y in Y_new]
cost1 = mean([c*x - q*x for x in decision1])

# cenário SAA
saa_model = Model(Gurobi.Optimizer)
@variables(saa_model, begin
    x_saa ≥ 0
    y_saa[1:200] ≥ 0
    z_saa[1:200] ≥ 0
end)
@constraints(saa_model, begin
    x_saa ≤ u
    y_saa .≤ Y_new
    y_saa + z_saa .≤ x_saa
end)
@objective(saa_model, Min, c*x_saa - 1/200*sum(q*y_saa + r*z_saa))
optimize!(saa_model)
value(x_saa)
cost2 = objective_value(saa_model)

# cenário KNN
k = 13
# calcular árvore 
kdtree = KDTree(X_new');
cost3 = 0
for i=1:200
    x = X_new[i, :]
    x_knn, knn_model = newsvendor_knn(x, Y_new, kdtree, k)
    cost3 += objective_value(knn_model)
end
cost3 /= 200

P = (cost2 - cost3) / (cost2 - cost1)
