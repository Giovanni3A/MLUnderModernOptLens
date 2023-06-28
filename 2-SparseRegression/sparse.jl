"""
O exercício envolve a implementação da solução do modelo de regressão esparsa.
Primeiro será implementada uma versão direta da restrição de norma zero dos regressores,
depois uma implementação com big-M.
"""

# instalando e importando pacotes necessários

# using Pkg
# Pkg.add("CSV")
# Pkg.add("DataFrames")
# Pkg.add("Plots")

using Random
using JuMP, Gurobi
using Distributions
using MLBase
using MLUtils
using Plots

include("data_read.jl")

Random.seed!(0)
const GUROBI_ENV = Gurobi.Env()

# definir bases de dados a serem utilizadas
files = [
    "winequality-red.csv",
    "ObesityDataSet_raw_and_data_sinthetic.csv",
    "Twitter.data"
]

"""
Solução direta primal da regressão esparsa

Entrada
-------
X: matrix de regressores
y: valores da variável objetivo
λ: termo de regularização da norma L₂
k: quantidade máxima de regressores a serem utilizados

Saída
-----
model: modelo JuMP
β: solução encontrada
"""
function sparse_regression_1(X, y, λ::Float64=0.1, k::Int64=1)
    # dimensões do problema
    p = size(X)[2]

    # montar o modelo
    model = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GUROBI_ENV)
        )
    )
    MOI.set(model, MOI.Silent(), true)
    @variable(model, β[1:p])
    @objective(model, Min, mean((y .- X*β).^2) + 0.5/λ * sum(β.^2))
    # restrição de esparcidade com variável auxiliar e expressão quadrática
    @variable(model, s[1:p], Bin)
    @constraint(model, [i=1:p], (1-s[i])*β[i] == 0)
    @constraint(model, sum(s) ≤ k)

    # buscar solução
    optimize!(model)

    # erro no status
    if !in([
        TerminationStatusCode(1), 
        TerminationStatusCode(4)
    ])(termination_status(model))
        throw(DomainError(termination_status(model)))
    end

    return (model, value.(β))

end

"""
Solução do problema primal com método big M para criar
restrições de esparcidade

Entrada
-------
X: matrix de regressores
y: valores da variável objetivo
M: limites máximo dos regressores em valor absoluto
λ: termo de regularização da norma L₂
k: quantidade máxima de regressores a serem utilizados

Saída
-----
model: modelo JuMP
β: solução encontrada
"""
function sparse_regression_2(X, y, M, λ::Float64=0.1, k::Int64=1)
    # dimensões do problema
    p = size(X)[2]

    # montar o modelo
    model = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GUROBI_ENV)
        )
    )
    MOI.set(model, MOI.Silent(), true)
    @variable(model, β[1:p])
    @objective(model, Min, mean((y .- X*β).^2) + 0.5/λ * sum(β.^2))
    # restrição de esparcidade com variável auxiliar e expressão quadrática
    @variable(model, s[1:p], Bin)
    @constraint(model, [i=1:p], -M[i]*s[i] ≤ β[i])
    @constraint(model, [i=1:p], β[i] ≤ M[i]*s[i])
    @constraint(model, sum(s) ≤ k)

    # buscar solução
    optimize!(model)

    # erro no status
    if !in([
        TerminationStatusCode(1), 
        TerminationStatusCode(4)
    ])(termination_status(model))
        throw(DomainError(termination_status(model)))
    end

    return (model, value.(β))

end

"""
Lógica de cálculo dos valores de big M para as variáveis explicativas

Entrada
-------
X: matrix de regressores
y: valores da variável objetivo
λ: termo de regularização da norma L₂
k: quantidade máxima de regressores a serem utilizados

Saída
-----
M: limites em valor absoluto dos regressores
"""
function estim_bigM(X, y, λ::Float64=0.1, k::Int64=1)

    p = size(X)[2]

    # estimar upper bound da função objetivo
    aux_model = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GUROBI_ENV)
        )
    )
    MOI.set(aux_model, MOI.Silent(), true)
    @variable(aux_model, β[1:p])
    @objective(aux_model, Min, mean((y .- X*β).^2) + 0.5/λ * sum(β.^2))
    optimize!(aux_model)
    # filtrar os k maiores valores
    β₀ = value.(β)
    top_values = sort(abs.(β₀))[end-k+1:end]
    top_idx = findall(x -> abs(x) in top_values, β₀)
    β₀[Not(top_idx)] .= 0
    # valor na solução é o upper bound
    UB = mean((y .- X*β₀).^2) + 0.5/λ * sum(β₀.^2)

    # montar modelo de estimativa
    estim_model = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GUROBI_ENV)
        )
    )
    MOI.set(estim_model, MOI.Silent(), true)
    @variable(estim_model, β[1:p])
    @constraint(estim_model, mean((y .- X*β).^2) + 0.5/λ * sum(β.^2) ≤ UB)

    # vetor com limites
    M = Vector{Any}(undef, p)

    # serão estimados valores por explicativa
    for i=1:p

        # mínimo
        @objective(estim_model, Min, β[i])
        optimize!(estim_model)
        u₋ = value(β[i])
        
        # máximo
        @objective(estim_model, Max, β[i])
        optimize!(estim_model)
        u₊ = value(β[i])

        # maior limite em valor absoluto
        M[i] = maximum(abs.([u₋ u₊]))

    end

    return M
    
end

# para a primeira parte, vamos executar ambas as metodologias para todas as
# bases, comparando os tempos de execução e soluções encontradas, dado um valor
# k fixo
k = 3
λ = 5.0

for file in files

    println("\n\n+-+-+-+-+-+-+-+-+-+- Dataset: $file -+-+-+-+-+-+-+-+-+-+\n")

    # leitura dos dados
    data = get_data(file)
    # normalizar dadados
    data = (data .- mean(data, dims=(1))) ./ std(data, dims=(1))
    X = data[:, 1:end-1]
    y = data[:, end]

    # execução do modelo com restrição quadrática
    @time model1, β₁ = sparse_regression_1(X, y, λ, k)
    t = round(solve_time(model1), digits=3)
    println("\nTempo de execução do modelo 1 = $t")
    # objective_value(model1)
    # β₁

    # estimativa de big M
    @time M = estim_bigM(X, y, λ, k)

    # execução do modelo com restrição big M
    @time model2, β₂ = sparse_regression_2(X, y, M, λ, k)
    t = round(solve_time(model2), digits=3)
    println("\nTempo de execução do modelo 2 = $t")
    # objective_value(model2)
    # β₂

    diff = round(sum(abs.(β₁ - β₂)), digits=7)
    println("Diferença entre soluções = $diff")

end

# última parte: implementar uma métrica de decisão do
# número de explicativas k
k_space = 1:10
train_size = 0.7

# vamos utilizar o cross validation para selecionar o melhor
# k no conjunto do Twitter
println("\n\nComparação entre modelos")
data = get_data("winequality-red.csv")
data = (data .- mean(data, dims=(1))) ./ std(data, dims=(1))
X = data[:, 1:end-1]
y = data[:, end]

n, p = size(X)

# train test split
idx_train, idx_test = splitobs(shuffleobs(1:n), at=train_size)
X_train = X[idx_train, :]
y_train = y[idx_train]
X_test = X[idx_test, :]
y_test = y[idx_test]

# criar 7-fold cross validation
folds = collect(Kfold(size(X_train)[1], 7))
k_space_err = Vector{Any}()
for k in k_space

    folds_err = Vector{Any}()
    # loop nos folds
    for idx_fold in folds

        # dados de treino e validação
        idx_validate = idx_train[(!in).(idx_train, Ref(idx_fold))]
        X_val = X[idx_validate, :]
        y_val = y[idx_validate]
        X_ = X_train[idx_fold, :]
        y_ = y_train[idx_fold]

        # estimar big M
        M = estim_bigM(X_, y_, λ, k)
        # execução do modelo com restrição big M
        model, θ = sparse_regression_2(X_, y_, M, λ, k)

        # registrar erro de validacao
        val_err = mean((y_val .- X_val*θ).^2)
        append!(folds_err, [val_err])
    end

    # avaliar erros do k
    k_err = mean(folds_err)
    test_err = round(k_err, digits=4)
    println("Validation error (k=$k) = $test_err")
    append!(k_space_err, k_err)

end

# selecionar melhor k
k_hat = k_space[argmin(k_space_err)]

# erro de teste do sparse model
M = estim_bigM(X_train, y_train, λ, k_hat)
primal_model, θ = sparse_regression_2(X_train, y_train, M, λ, k_hat)
test_error_primal = round(mean((y_test .- X_test*θ).^2), digits=4)
println("Erro médio do modelo sparse = $test_error_primal")

# comparar resultado (em teste) com lasso e ada lasso

# lasso
using Lasso
lasso_model = fit(LassoModel, X_train, y_train, λ=[0.5/λ], intercept=false)
test_error_lasso = round(mean((predict(lasso_model, X_test) - y_test) .^2), digits=4)
println("Erro médio do modelo lasso = $test_error_lasso")

# ada lasso (utilizar inv coefs do lasso como pesos)
w = 1 ./ abs.(coef(lasso_model) .+ 1e-3)
ada_lasso = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV)
    )
)
@variable(ada_lasso, β[1:p])
@variable(ada_lasso, ε[1:p])
@constraint(ada_lasso, [i=1:p], ε[i] ≥ β[i])
@constraint(ada_lasso, [i=1:p], ε[i] ≥ -β[i])
train_mse = sum((y_train .- X_train*β).^2)
@objective(ada_lasso, Min, train_mse + 0.5/λ * w'*ε)

# buscar solução
optimize!(ada_lasso)
test_error_ada_lasso = round(mean((X_test*value.(β) - y_test) .^2), digits=4)
value.(β)
println("Erro médio do modelo ada lasso = $test_error_ada_lasso")