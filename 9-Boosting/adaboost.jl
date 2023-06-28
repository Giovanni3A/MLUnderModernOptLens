"""
ADABoost
"""

# preparando o ambiente
using DecisionTree
using Random
using MLUtils
using ProgressBars
using Plots

"""
Função de reamostragem dos dados baseado nos pesos

Entradas
--------
pesos: Array{Float}
    Distribuição de pesos dos dados

Retorna
-------
D: Array{Int}
    Re-Distribuição dos indexes dos pesos
"""
function reamostragem(pesos)
    p = cumsum(pesos)
    D = []
    for i=1:size(pesos)[1]
        s = rand()
        j = argmax(p .>= s)
        D = [D; j]
    end

    return D
end

# leitura dos dados
features, labels = load_data("adult")
idx = shuffleobs(1:size(features)[1])[1:10000]
features = features[idx, :]
labels = labels[idx]
labels = 2*(Int.(labels .== " >50K") .- 0.5)
# split treino teste
train_obs, test_obs = splitobs(shuffleobs(1:size(features)[1]), at=0.7)
X_train = @view features[train_obs, :]
X_test = @view features[test_obs, :]
y_train = @view labels[train_obs]
y_test = @view labels[test_obs]

# tamanho da base para treino
N = size(X_train)[1]
# número de iterações de boosting
M = 50

# salvar modelos
models = []
# pesos das amostras por iteração
w = ones((N, M)) * 1/N
# erro dos modelos (geral)
ϵ = zeros(M)
# importância dos modelos
α = zeros(M)
# erros medidos
train_eval = []
test_eval = []

for m in ProgressBar(1:M)
    # reamostragem dos dados baseado nos pesos ds iteração
    D = reamostragem(w[:, m])
    X_m = @view X_train[D, :]
    y_m = @view y_train[D]

    # treinar aprendiz fraco com pesos da iteração
    dt_m = DecisionTreeClassifier(max_depth=1)
    fit!(dt_m, X_m, y_m)
    push!(models, dt_m)
    
    # calcular erro do aprendiz treinado
    fm = predict(dt_m, X_m)
    fm = fm ./ abs.(fm)
    ϵ[m] = sum(w[:, m] .* (fm .!= y_m))

    # calcular importancia do aprendiz fraco
    α[m] = 0.5*log((1-ϵ[m])/ϵ[m])

    # calcular novos pesos
    if m < M
        w[:, m+1] = w[:, m] .* exp.(α[m] * (fm .!= y_m))
        # normalizar os pesos
        w[:, m+1] = w[:, m+1] / sum(w[:, m+1])
    end

    # calcular erros
    H_train = sign.(sum([α[im]*predict(models[im], X_train) for im=1:m]))
    acc_train = mean(H_train .== y_train)
    train_eval = [train_eval; acc_train]
    
    H_test = sign.(sum([α[im]*predict(models[im], X_test) for im=1:m]))
    acc_test = mean(H_test .== y_test)
    test_eval = [test_eval; acc_test]
end

# erro dentro da amostra
err_train = 1 .- train_eval
err_train[[1, 10, 20, 50]]
# erro fora da amostra
err_test = 1 .- test_eval
err_test[[1, 10, 20, 50]]

plot(err_train, label="Treino", title="Erro de classificação por iteração")
plot!(err_test, label="Teste")
savefig("plot1.png")

# distribuição de pesos
bins = range(0, 1e-3, length=50)
histogram(w[:, 1], bins=bins, label="Iteração 1", title="Distribuição dos pesos", alpha=0.5)
histogram!(w[:, 10], bins=bins, label="Iteração 10", alpha=0.5)
histogram!(w[:, 20], bins=bins, label="Iteração 20", alpha=0.5)
histogram!(w[:, 50], bins=bins, label="Iteração 50", alpha=0.5)
savefig("plot2.png")
