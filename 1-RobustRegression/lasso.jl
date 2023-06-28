# using Pkg
# Pkg.add("CSV")

# Pkg.add("DataFrames")
# Pkg.add("MLBase")
# Pkg.add("MLUtils")
# Pkg.add("Plots")

using Logging
using Random
using MLBase
using MLUtils
using JuMP, Gurobi
using Plots

include("data_read.jl")

Random.seed!(0)
const GUROBI_ENV = Gurobi.Env()
log_io = open("LassoExperiment.log", "w+")
logger = SimpleLogger(log_io)
global_logger(logger)

# general parameters
filename = "Twitter.data"
train_size = 0.7
k = 5
lambda_space = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

# execution code

# data reading
data = get_data(filename)
# normalize data
data = (data .- mean(data, dims=(1))) ./ std(data, dims=(1))
# get matrix X, y and shapes
X = data[:, 1:end-1]
y = data[:, end]
n, m = size(X)

# train test split
idx_train, idx_test = splitobs(shuffleobs(1:n), at=train_size)
X_train = X[idx_train, :]
y_train = y[idx_train]
X_test = X[idx_test, :]
y_test = y[idx_test]
folds = collect(Kfold(size(X_train)[1], k))

"""
Construct and execute the optimization model
for a given set of train data
"""
function run_lasso(X_, y_, λ_, p_, q_)
    # optimization model
    n_, m_ = size(X_)
    lasso = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GUROBI_ENV)
        )
    )

    # beta variable
    @variable(lasso, β[1:m_])

    # error factor depends on p arg
    if p_ == 1
        @variable(lasso, ε[1:n_])
        @constraints(lasso, begin
            ε .≥ (X_*β - y_)
            ε .≥ -(X_*β - y_)
        end)
    elseif p_ == 2
        ε = (X_*β - y_).^p_
    end

    # regularization factor depends on q arg
    if q_ == 1
        @variable(lasso, reg_pen[1:m_])
        @constraints(lasso, begin
            reg_pen .≥ β
            reg_pen .≥ -β
        end)
    elseif q_ == 2
        reg_pen = β.^q_
    end

    # create objective and optimize
    @objective(lasso, Min, sum(ε) + λ_*sum(reg_pen))
    MOI.set(lasso, MOI.Silent(), true)
    optimize!(lasso)
    
    # error status not 4
    if !in([
        TerminationStatusCode(1), 
        TerminationStatusCode(4)
    ])(termination_status(lasso))
        throw(DomainError(termination_status(lasso)))
    end

    return lasso
end


"""
Lambda selection via K-Fold Cross Validation
"""
function select_lambda(p_, q_)
    # iterate in lambda space
    lambdas_err = Vector{Any}()
    @info "Iterating in lambda space: $lambda_space"
    for λ in lambda_space
        @info "λ = $λ"

        # k-fold cross validation
        folds_err = Vector{Any}()

        # loop in folds
        for idx_fold in folds

            # get train except fold
            idx_validate = idx_train[(!in).(idx_train, Ref(idx_fold))]
            X_val = X[idx_validate, :]
            y_val = y[idx_validate]

            X_ = X_train[idx_fold, :]
            y_ = y_train[idx_fold]
            t1 = @elapsed (lasso = run_lasso(X_, y_, λ, p_, q_))
            @info "T_lasso(s) = $t1"

            # calcular e salvar erro em teste
            β = value.(object_dictionary(lasso)[:β])
            if p_ == 1
                err = mean(abs.(X_val*β - y_val))
            elseif p_ == 2
                err = mean((X_val*β - y_val).^2)
            end
            append!(folds_err, [err])
        end

        # evaluate folds error
        lambda_err = mean(folds_err)
        test_err = round(lambda_err, digits=4)
        @info "Test error = $test_err"
        println("Test error = $test_err")
        append!(lambdas_err, lambda_err)

    end

    # select best lambda
    λ_hat = lambda_space[argmin(lambdas_err)]

    return λ_hat
end

p_q_errs = Vector{Any}()
for p=1:2, q=1:2
    @info "Starting iteration | p = $p | q = $q"
    println("Starting iteration | p = $p | q = $q")
    # get best lambda
    t2 = @elapsed (λ_hat = select_lambda(p, q))
    @info "Selected λ = $λ_hat ($t2 s)"

    # use full train data
    lasso = run_lasso(X_train, y_train, λ_hat, p, q);

    # compute and save out of sample error
    β_hat = value.(object_dictionary(lasso)[:β])
    err = (X_test*β_hat - y_test)
    append!(p_q_errs, [err])
end

summary = [
    [
        mean(abs.(err)), # norm 1
        mean(err.^2), # norm 2
        maximum(abs.(err)), # max
        mean(err), # avg
        std(err) # std
    ]
for err in p_q_errs]
summary = mapreduce(permutedims, vcat, summary)
results = DataFrame(
    mae = summary[:, 1],
    mse = summary[:, 2],
    max_err = summary[:, 3],
    avg_err = summary[:, 4],
    std_err = summary[:, 5],
)
CSV.write("results.csv", results)

# plot error distribution
b_range = range(-3, 3, length=50)
for i=1:4
    p = div(i+1,2)
    q = 2-mod(i,2)
    # create and save histogram
    plot = histogram(p_q_errs[i], label="p=$p, q=$q", bins=b_range, color=:grey)
    savefig(plot, "err_dist_$(p)_$(q).png")
end

# fechar arquivo de log
close(log_io)