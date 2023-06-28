"""
O exercício consiste na implementação e avaliação
do algoritmo de regressão estável apresentado no
capítulo 17.
"""

using Random
using CSV, DataFrames
using MLBase
using MLUtils
using JuMP
using Gurobi

Random.seed!(0)
const GUROBI_ENV = Gurobi.Env()

# leitura de dados
df = CSV.read("Aula3//winequality-red.csv", DataFrame)
data = Array(df)
X = data[:, 1:end-1]
y = data[:, end]
n, p = size(X)

# parâmetros gerais
train_size = 0.9
lambda_space = [0.0, 0.01, 0.1, 1, 10]
k_space = [0.3, 0.4, 0.5]

# implementação da regressão estável
"""
Implementação do modelo de regressão estável, com ou sem termo
de regularização Lasso

Entradas
--------
X::Array{Float64}
    Matriz de rergessores
y::Array{Float64}
    Valores objetivo
λ::Float64
    Peso aplicado à regularização Lasso
useLasso::Bool
    Chave de uso de regularização Lasso    
K::Int64
    Tamanho do conjunto de treino a ser selecionado    

Saídas
------
θ::Vector{Float64}
    Parâmetros calculados
"""
function stable_regression(
    X::Array{Float64},
    y::Array{Float64},
    K::Int64,
    useLasso::Bool,
    λ::Float64,
    )
    
    n, p = size(X)
    if K > n
        throw(ArgumentError("K cannot be bigger than N=$n."))
    end

    model = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GUROBI_ENV)
        )
    )
    @variables(model,begin
        θ[1:p]
        δ ≥ 0
        u[1:n] .≥ 0
    end)

    # restricao com erro quadrático
    if useLasso
        # termo de regularização
        @variable(model, L[1:p])
        @constraints(model, begin
            L .≥  θ
            L .≥ -θ
        end)
        regul = λ*sum(L)
        err = (y .- X*θ).^2 .+ regul
    else
        err = (y .- X*θ).^2
    end
    @constraint(model, [i=1:n], δ + u[i] ≥ err[i])

    # objetivo
    @objective(model, Min, K*δ + sum(u))
    MOI.set(model, MOI.Silent(), true)

    # encontrar solução e retornar parâmetros e erro de validação
    optimize!(model)

    uval = value.(u)
    max_train_u = sort(uval, rev=true)[K]
    val_samples = findall(x->x>=max_train_u, uval)
    val_err = mean((X[val_samples, :]*θ .- y[val_samples]).^2)

    return (value.(θ), value(val_err))
end

b1, e1 = stable_regression(X, y, 100, false, 0.0)
b2, e2 = stable_regression(X, y, 100, true, 1.0)

# implementação da regressão Lasso com validação cruzada
"""
Implementação do modelo de regressão "simples", com 
possibilidade de regularização Lasso

Entradas
--------
X::Array{Float64}
    Matriz de rergessores
y::Array{Float64}
    Valores objetivo
λ::Float64
    Peso aplicado à regularização Lasso
useLasso::Bool
    Chave de uso de regularização Lasso  

Saídas
------
θ::Vector{Float64}
    Parâmetros calculados
"""
function simple_regression(
    X::Array{Float64},
    y::Array{Float64},
    useLasso::Bool,
    λ::Float64)

    p = size(X)[2]
    model = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GUROBI_ENV)
        )
    )
    @variable(model, θ[1:p])

    # função objetivo
    err = (y .- X*θ).^2
    if useLasso
        # termo de regularização
        @variable(model, L[1:p])
        @constraints(model, begin
            L .≥  θ
            L .≥ -θ
        end)
        obj = sum(err) + λ*sum(L)
    else
        obj = sum(err)
    end
    @objective(model, Min, obj)
    MOI.set(model, MOI.Silent(), true)

    # encontrar solução e retornar parâmetros e erro
    optimize!(model)

    return value.(θ)
end

# implementar a avaliação por validação cruzada com os modelos simples
"""
Loop de avaliação do modelo Lasso em conjuntos de validação

Entradas
--------
X::Array{Float64}
    Matriz de rergessores
y::Array{Float64}
    Valores objetivo
λ::Float64
    Peso aplicado à regularização Lasso
val_size::Float64
    Tamanho (em fração) do conjunto a ser utilizado como validação
n_splits::Int64
    Quantidade de vezes a serem repetidas o processo de treino
    e validação (o erro final será a média dos processos)

Saídas
------
mean_val_error::Float64
    Erro médio de validação calculado
best_b::Array{Float64}
    Estimativa β calculada no split de menor erro
"""
function eval_reglin(
    X::Array{Float64}, y, useLasso, λ, val_size, n_splits)

    # definir conjuntos de treino e validação
    n_val = floor(size(X)[1] * val_size)
    validation_folds = RandomSub(size(X)[1], n_val, n_splits)

    # loop pelos subconjuntos, treinando o modelo
    # e calculando erro out-of-sample
    idx_total = 1:size(X)[1]
    validation_errors = Vector{Float64}[]
    best_b = nothing
    smaller_err = 1e9
    for idx_validate in validation_folds
        idx_train = idx_total[(!in).(idx_total, Ref(idx_validate))]
        X_train = X[idx_train, :]
        y_train = y[idx_train]
        X_val = X[idx_validate, :]
        y_val = y[idx_validate]

        β = simple_regression(X_train, y_train, useLasso, λ)
        oos_err = mean((X_val*β .- y_val).^2)
        if oos_err < smaller_err
            smaller_err = oos_err
            best_b = β
        end
        validation_errors = [validation_errors; [oos_err]]
    end

    return mean(validation_errors), best_b
end

# agora vamos criar o loop final de avaliação
# nele, precisamos, para cada valor de λ, k,
# sendo k o tamanho da base de validação:
# 1) computar o resultado das regressões simples
# 2) computar o resultado das regressões robustas
# ao fim do loop em λ:
# 1) selecionar o melhor λ por modelo
# 2) treinar o modelo utilizando esse valor
# 3) avaliar o modelo no conjunto de teste
results = Array{Any}[]
for i=1:1
    # separação em conjunto de treino/teste
    idx_train, idx_test = splitobs(shuffleobs(1:n), at=train_size)
    X_train = X[idx_train, :]
    y_train = y[idx_train]
    X_test = X[idx_test, :]
    y_test = y[idx_test]

    for val_size in k_space 
        errs_lasso  = Vector{Float64}[]
        errs_stable2 = Vector{Float64}[]
        k = floor(Int, (1-val_size)*size(X_train)[1])

        for λ in lambda_space
            
            # computar o erro de Lasso
            err_lasso, β_lasso = eval_reglin(X_train, y_train, true, λ, val_size, 1)
            errs_lasso = [errs_lasso; [err_lasso]]

            # computar o erro da regressao lasso robusta
            βₛ2, err_stable2 = stable_regression(X_train, y_train, k, true, λ)
            errs_stable2 = [errs_stable2; [err_stable2]]
        end

        # identificar melhor lambda
        λ_lasso  = lambda_space[findmin(errs_lasso)[2]]
        λ_stable2 = lambda_space[findmin(errs_stable2)[2]]
    
        # treinar melhor modelo no dado de treino completo
        β_reglin = simple_regression(X_train, y_train, false, 0.0)
        β_lasso = simple_regression(X_train, y_train, true, λ_lasso)
        β_stable1, err_ = stable_regression(X_train, y_train, k, false, 0.0)
        β_stable2, err_ = stable_regression(X_train, y_train, k, true, λ_stable2)

        # avaliar em teste
        test_err_reglin = mean((X_test*β_reglin .- y_test).^2)
        test_err_lasso = mean((X_test*β_lasso .- y_test).^2)
        test_err_stable1 = mean((X_test*β_stable1 .- y_test).^2)
        test_err_stable2 = mean((X_test*β_stable2 .- y_test).^2)

        println("\n(i=$i)")
        println("Resultados para split de validação $val_size")
        println("Erro em teste (RegLin) = $test_err_reglin")
        println("Erro em teste (Lasso, λ=$λ_lasso) = $test_err_lasso")
        println("Erro em teste (RegLin Robusta) = $test_err_stable1")
        println("Erro em teste (Lasso Robusta, λ=$λ_stable2) = $test_err_stable2")
        r = [
            i, val_size, 
            β_reglin, β_lasso, β_stable1, β_stable2,
            test_err_reglin, test_err_lasso, test_err_stable1, test_err_stable2
        ]
        push!(results, r)
    end
end

# montar dataframe com resultados de avaliação
results_df = DataFrame(
    i=[r[1] for r in results],    
    k=[r[2] for r in results],    
    b1=[r[3] for r in results],    
    b2=[r[4] for r in results],    
    b3=[r[5] for r in results],   
    b4=[r[6] for r in results],    
    err1=[r[7] for r in results],    
    err2=[r[8] for r in results],
    err3=[r[9] for r in results],
    err4=[r[10] for r in results],
)

# resultados
results_df

# regressão linear simples
mean(results_df.err1)
std(results_df.err1)

# lasso simples
mean(results_df.err2)
std(results_df.err2)

# regressão linear robusta
mean(results_df.err3)
std(results_df.err3)

# lasso robusta
mean(results_df.err4)
std(results_df.err4)

results_df

# por k
combine(groupby(results_df, :k), :err1 => mean, :err2 => mean, :err3 => mean, :err4 => mean)
combine(groupby(results_df, :k), :err1 => std, :err2 => std, :err3 => std, :err4 => std)
