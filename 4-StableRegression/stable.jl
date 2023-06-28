"""
O exercício consiste na implementação e avaliação
dos modelos de minimos quadrados e lasso com seus
respectivos pares robustos.
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


# loop geral
results = Array{Any}[]
for i=1:2
    println(i)

    # separação em conjunto de treino/teste
    idx_train, idx_test = splitobs(shuffleobs(1:n), at=train_size)
    X_train = X[idx_train, :]
    y_train = y[idx_train]
    X_test = X[idx_test, :]
    y_test = y[idx_test]

    # iteração pelas proporções de validação
    for val_size in k_space
        k = floor(Int, (1-val_size)*size(X_train)[1])

        # definir conjunto de treino e validação (para modelos simples)
        n_train = size(X_train)[1]
        n_val = floor(n_train * val_size)
        idx_val, idx_train2 = splitobs(shuffleobs(1:n_train), at=val_size)
        X_train2 = X_train[idx_train2, :]
        y_train2 = y_train[idx_train2]
        X_val = X_train[idx_val, :]
        y_val = y_train[idx_val]

        # minimos quadrados
        global β_ls = simple_regression(X_train2, y_train2, false, 0.0)
        avg_test_err = mean((X_test*β_ls .- y_test).^2)
        std_estim = std(X_test*β_ls .- y_test)
        push!(results, ["ls", val_size, avg_test_err, std_estim])

        # lasso - iterar entre lambdas
        min_error = 1e9
        λ_lasso = nothing
        global β_lasso = nothing
        for λᵢ in lambda_space
            βᵢ = simple_regression(X_train2, y_train2, true, λᵢ)
            val_err = mean((X_val*βᵢ .- y_val).^2)
            if val_err < min_error
                min_error = val_err
                λ_lasso = λᵢ
                β_lasso = βᵢ
            end
        end
        avg_test_err = mean((X_test*β_lasso .- y_test).^2)
        std_estim = std(X_test*β_lasso .- y_test)
        push!(results, ["lasso", val_size, avg_test_err, std_estim])

        # minimos quadrados (robusto)
        global β_ls_rob, _ = stable_regression(X_train, y_train, k, false, 0.0)
        avg_test_err = mean((X_test*β_ls_rob .- y_test).^2)
        std_estim = std(X_test*β_ls_rob .- y_test)
        push!(results, ["ls_rob", val_size, avg_test_err, std_estim])
        
        # lasso (robusto)
        min_error = 1e9
        λ_lasso_rob = nothing
        global β_lasso_rob = nothing
        for λᵢ in lambda_space
            βᵢ, val_err = stable_regression(X_train, y_train, k, true, λᵢ)
            if val_err < min_error
                min_error = val_err
                λ_lasso_rob = λᵢ
                β_lasso_rob = βᵢ
            end
        end
        avg_test_err = mean((X_test*β_lasso_rob .- y_test).^2)
        std_estim = std(X_test*β_lasso_rob .- y_test)
        push!(results, ["lasso_rob", val_size, avg_test_err, std_estim])
    end
end

results_df = DataFrame(
    model=[r[1] for r in results],
    split=[r[2] for r in results],
    err=[r[3] for r in results],
    std_est=[r[4] for r in results],
)

avg_result = combine(
    groupby(results_df, [:model, :split]),
    :err => mean,
    :std_est => mean
)


"""
12×3 DataFrame
 Row │ model      split    β_f      
     │ String     Float64  Float64
─────┼──────────────────────────────
   1 │ ls             0.3  0.590565
   2 │ lasso          0.3  0.594346
   3 │ ls_rob         0.3  0.589791
   4 │ lasso_rob      0.3  0.589833
   5 │ ls             0.4  0.589003
   6 │ lasso          0.4  0.595186
   7 │ ls_rob         0.4  0.589532
   8 │ lasso_rob      0.4  0.589506
   9 │ ls             0.5  0.591004
  10 │ lasso          0.5  0.596908
  11 │ ls_rob         0.5  0.590388
  12 │ lasso_rob      0.5  0.590389
"""