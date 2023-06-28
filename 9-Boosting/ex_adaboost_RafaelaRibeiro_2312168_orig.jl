
# Exercício de Boosting - Giovanni Amorim
# Aluna: Rafaela Ribeiro - 2312168

# Adicionando os pacotes

using DecisionTree
#using IterTools: product
using DecisionTree: build_stump, apply_tree, _weighted_error
using Plots
using Random
using Statistics

Random.seed!(42)

# Carregando os dados

features, labels = load_data("adult")
features = features[1:1000,[1,3,5,11,12,13]]
labels = labels[1:1000]

N = length(labels)
for i in 1:N
    if labels[i] == " <=50K"
        labels[i] = 1
    else
        labels[i] = -1
    end
end
n = Int(ceil(N*0.7))
X_treino = features[1:n,:]
X_teste = features[n+1:N,:]
Y_treino = labels[1:n]
Y_teste = labels[n+1:N]

# Modelo de classificação binária como aprendiz fraco
# Usaremos a função build_stump do pacote DecisionTree.jl

# O algoritmo

function AdaBoost(X_treino, Y_treino, X_teste, Y_teste, M::Integer, n::Integer)
    
    weights = []
    w = zeros(n, M+1)
    w[:,1] .= 1/n
    α = []
    ϵ = Array{Float64}(undef, M, 1)
    models = []
    (X, Y) = (X_treino, Y_treino)

    for m in 1:M
        model = build_stump(
            Y, X, w[:,m];
        )

        labels_pred_treino = apply_tree(model, X)
        acc = mean(labels_pred_treino .== Y) # 0.7739

        labels_pred_teste = apply_tree(model, X_teste)
        acc = mean(labels_pred_teste .== Y_teste) # 0.7694

        ϵ[m] = _weighted_error(Y, labels_pred_treino, w[:,m])

        α_ = 0.5 * log((1 - ϵ[m]) / ϵ[m])
        w[:,m+1] = w[:,m] .* exp.(-α_ .* Y .* labels_pred_treino)
        w[:,m+1] = w[:,m+1] ./ sum(w[:,m+1])

        push!(models, deepcopy(model))
        push!(α, α_)
        push!(weights, deepcopy(w[:,m+1]))

    end
    return models, α, weights
end

# Previsão

function prev(α, models, X)
    y = zeros(size(X)[1])
    for (m,t) in enumerate(models)
        y = α[m] .* apply_tree(t,X) # peso do modelo * a previsão da árvore
    end

    return sign.(y)
end

# O loop

iter = [1, 10, 20, 50]
total_iter = length(iter)
erro_amostra = Array{Float64}(undef, total_iter, 1)
erro_fora_amostra = Array{Float64}(undef, total_iter, 1)
pesos = Array{Vector{Float64}}(undef, total_iter, 1)
for (j,i) in enumerate(iter)
    println(i)
    models, alpha, peso = AdaBoost(X_treino, Y_treino, X_teste, Y_teste, i, n)

    pred_treino = prev(alpha, models, X_treino)
    erro_amostra[j] = sum(pred_treino .!= Y_treino) / size(X_treino)[1]

    pred_teste = prev(alpha, models, X_teste)
    erro_fora_amostra[j] = sum(pred_teste .!= Y_teste) / size(X_teste)[1]
    pesos[j] = peso[i]
end

# Plot 1, 10, 20 e 50

plot(histogram(pesos[1], bins=0:0.0001:maximum(pesos[1])), legend = false, 
histogram(pesos[2], bins=0:0.0001:maximum(pesos[2])), 
histogram(pesos[3], bins=0:0.0001:maximum(pesos[3])), 
histogram(pesos[4], bins=0:0.0001:maximum(pesos[4])), 
layout = (4, 1))

plot(iter, [mean(erro_amostra[1]), mean(erro_amostra[2]),
mean(erro_amostra[3]), mean(erro_amostra[4])])

plot!(iter, [mean(erro_fora_amostra[1]), mean(erro_fora_amostra[2]),
mean(erro_fora_amostra[3]), mean(erro_fora_amostra[4])])

# Acurácia

models, alpha, peso = AdaBoost(X_treino, Y_treino, X_teste, Y_teste, 20, n)
pred_teste = prev(alpha, models, X_teste)
acc_adaboost = sum(pred_teste .== Y_teste) / size(X_teste, 1) # 0.47

max_depth = 1
model = DecisionTreeClassifier(max_depth=max_depth)
fit!(model, X_treino, Y_treino)
stump = predict(model, X_teste)
acc_stump = sum(stump .== Y_teste) / size(X_teste, 1) # 0.75
