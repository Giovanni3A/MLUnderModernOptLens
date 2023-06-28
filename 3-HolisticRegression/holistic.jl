"""
O exercício consiste na implementação e avaliação
do algoritmo de detecção de ϵ-multicolinearidade apresentado no capítulo
"Holistic Regression" do livro.

Como exercício de aprendizado, também foi feita a implementação do algoritmo
de regressão descrito no capítulo.
"""

using Random
using Distributions
using LinearAlgebra
using JuMP
using Gurobi, GLPK

Random.seed!(0)

# primeiro criaremos um conjunto de dados para testar nosso algoritmo
# selecionaremos apenas um subconjunto das features disponíveis para
# de fato construir nosso vetor y
n = 300
p0 = 4
p = p0 + 20
features = shuffle(1:p)[1:p0]
β = zeros(p)
β[features] = rand(Normal(0, 1), p0)

X = rand(Normal(0, 1), n, p)
# adicionar relações de dependência linear
X[:, 1] = 10*X[:, 2] - 5*X[:, 21] - 9*X[:, 8] + rand(Normal(0, .1), n)
X[:, 7] = 12*X[:, 23] - 7*X[:, 16] + rand(Normal(0, .1), n)
X[:, 6] = 5*X[:, 4] - 8*X[:, 11] + rand(Normal(0, .1), n)
y = X*β + rand(Normal(0, 1), n)
X = round.(X, digits=2)
y = round.(y, digits=3)

λ = eigen(X'X)
min_λ = median(λ.values)
V = λ.vectors[:, findall(x -> x <= min_λ, λ.values)]

# devemos implementar o algoritmo de extração dos vetores caracteristicos
# de menor suporte
"""
Implementação do problema de busca de vetores característicos
com suporte mínimo

Entradas
--------
V::Array{Float64}
    Os autovetores da matriz caracteristica com autovalores menores do que ϵ
    previamente extraídas
δ::Float64
    Constante positiva que garante a≠0
M::Float64
    Constante positiva que limita a amplitude em valores absoluto dos elementos
    de a

Saídas
------
combinations::Vector{Array{Int64}}
    Combinações de índices de variáveis com relação identificada
"""
function minimum_support_span(V::Array{Float64}, δ::Float64=1e-1, M::Float64=1e3, maxIter::Int64=3)

    # inicia problema com variáveis de decisão
    p, m = size(V)
    model = Model(GLPK.Optimizer)
    @variable(model, θ[1:m])
    @variable(model, z[1:p], Bin)
    a = V*θ

    # |sum(θᵢ)| ≥ δ
    @variable(model, t, Bin)
    @constraint(model, t + sum(θ) ≥ δ)
    @constraint(model, -(1-t) + sum(θ) ≤ -δ)

    # |αⱼ| ≤ M.zⱼ ∀ j
    @constraint(model, [j=1:p], a[j] <=  M*z[j])
    @constraint(model, [j=1:p], a[j] >= -M*z[j])

    @objective(model, Min, sum(z))
    set_silent(model)
    set_attribute(model, "tm_lim", 600)

    combinations = Array{Any}[]
    c = 0
    iter = 0
    while true
        iter += 1

        # busca solução
        optimize!(model)
        
        if termination_status(model) == TerminationStatusCode(1)
            # caso encontre solução, extrair suporte, guardar,
            # adicionar restrição
            supp_a = findall(!iszero, round.(value.(z)))
            println("Identificada relação multicolinear de variáveis: $supp_a")
            push!(combinations, supp_a)
            @constraint(model, sum(z[supp_a]) ≤ size(supp_a)[1] - 1)
        else
            # caso contrario retornar combinações encontradas
            return combinations
        end

        if iter >= maxIter
            return combinations
        end
    end
end

# identificar as combinações com colinearidade
colin_supp = minimum_support_span(V, 0.1, 1e3, 3)

# agora vamos contruir o modelo completo de regressão logísitca
# que usa como insumo as relações identificadas
"""
Implementação do modelo de regressão holísitica com as propriedades:
    1) Esparcidade
    2) Esparcidade em grupo
    3) Limite de multicolinearidade em pares
    4) Robustez
    5) Limite de multicolinearidade em grupos

Entradas
--------
X::Array{Float64}
    Matriz de rergessores
y::Array{Float64}
    Valores objetivo
colin_supp::Vector{Array{Int64}}
    Vetor com grupos de variáveis para aplicação de restrição de
    limite de multicolinearidade
T::Float64
    Peso do termo de regularização Lasso
M::Float64
    Constante positiva que limita a amplitude em valores absoluto dos parâmetros
k::Int64
    Quantidade máxima de parâmetros diferentes de zero
c::Float64
    Limite de correlação entre duas variáveis acima do qual é adicionada a restrição
    de limite de uso do par
G::Vector{Vector{Int64}}
    Vetor com grupos para aplicação de restrição de esparcidade em grupo
    

Saídas
------
θ::Vector{Float64}
    Parâmetros calculados
"""
function holistic_regression(
    X::Matrix{Float64},
    y::Vector{Float64},
    colin_supp::Vector{Array{Any}},
    G::Vector{Vector{Int64}},
    k::Int64=5,
    T::Float64=0.1,
    M::Float64=1e2,
    c::Float64=0.7,
    )

    # G = [[14, 15]]

    n, p = size(X)
    model = Model(Gurobi.Optimizer)
    @variable(model, θ[1:p])

    # construir termo de regularização (robustez)
    @variable(model, reg[1:p])
    @constraints(model, begin
        reg .≥  θ
        reg .≥ -θ
    end)

    # esparcidade
    @variable(model, z[1:p], Bin)
    @constraints(model, begin
        θ .≥ -M*z
        θ .≤  M*z
        sum(z) ≤ k
    end)

    # esparcidade em grupo
    for (i, j) in G
        @constraint(model, z[i] == z[j])
    end

    # multicolinearidade em pares (identificar por correlação)
    HC = findall(x -> x ≥ c, cor(X))
    for pair in HC
        (i, j) = Tuple(pair)
        # não adicionar restrições repetidas nem de i=j
        if i < j
            @constraint(model, z[i] + z[j] ≤ 1)
        end
    end

    # multicolinearidade multipla
    for rel in colin_supp
        @constraint(model, sum(z[rel]) ≤ size(rel)[1] - 1)
    end

    err_term = 0.5*sum((y .- X*θ).^2)
    reg_term = T*sum(reg)
    @objective(model, Min, err_term)

    optimize!(model)

    return value.(θ)

end

# vamos fazer uma rodada qualquer, criando um grupo de variáveis para
# usar a funcionalidade de esparcidade em grupo
G = [[14, 15]]
θ = holistic_regression(X, y, colin_supp, G)

[β θ (θ-β)./β]
findall(x->x!=0, β)
findall(x->x!=0, θ)
# podemos perceber que mesmo com o parâmetro k "errado",
# ou seja, diferente do número real de features usadas,
# o modelo foi capaz de aproximar muito bem os valores reais

# vamos testar com o k "real"
θ = holistic_regression(X, y, colin_supp, G, 4)
[β θ (θ-β)./β]

# o modelo teve um resultado bom, identificando corretamente
# as variáveis necessárias e os valores dos parâmetros
