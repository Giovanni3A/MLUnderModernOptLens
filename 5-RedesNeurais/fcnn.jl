"""
Fully Connected Neural Networks
"""

# instalação de pacotes necessários
using Pkg
if ~in("Flux" , keys(Pkg.installed()))
    Pkg.add("Flux")
end

# preparação de ambiente
using Plots
using Random
using Distributions
using MLUtils
using MLBase
using Flux
using ProgressBars
using Base.Iterators

Random.seed!(0)

# simular conjunto de dados a partir de função não linear
n = 3000
p = 5
X = rand(Normal(0, 1), (n, p))
y = abs.(X[:, 1]) + sin.(X[:, 2]) - cos.(X[:, 3])
y += rand(Normal(0, 1), n)
X = [X y]

# separar em treino e validação
train_size = 0.7
idx_train, idx_val = splitobs(shuffleobs(1:n), at=train_size)
n_train = size(idx_train)[1]
X_train = @view X[idx_train, 1:end-1]
y_train = @view X[idx_train, end]
X_val = @view X[idx_val, 1:end-1]
y_val = @view X[idx_val, end]

# definir uma rede neural totalmente contectada com uma camada escondida
model = Chain(
    Dense(p => 256, tanh),
    Dense(256, 1)
)
# otimizador Adam
opt = Adam(0.01)
# loop de treino salvando mse de validação
val_loss = [mean((model(X_train') - y_train').^2)]
trn_loss = [mean((model(X_val') - y_val').^2)]
for epoch in ProgressBar(1:500)
    # calcular erro e gradiente em treino
    gs = gradient(Flux.params(model)) do
        y_hat = model(X_train')
        l = sum((y_hat - y_train').^2)
    end
    
    # atualizar pesos da rede
    Flux.update!(opt, Flux.params(model), gs)

    # avaliar nos conjuntos de treino e validação
    mse_trn = mean((model(X_train') - y_train').^2)
    mse_val = mean((model(X_val') - y_val').^2)
    push!(trn_loss, mse_trn)
    push!(val_loss, mse_val)
end
# plot dos erros por época
plot(trn_loss, label="Train", title="MSE / Epoch")
plot!(val_loss, label="Validation")
savefig("plot1")

# mesma avaliação porém para diferentes tamanhos de modelo
hidden_size_list = 4:4:64
val_loss = []
trn_loss = []
for hidden_size in ProgressBar(hidden_size_list)
    # definir o modelo de hidden_size variável
    model = Chain(
        Dense(p => hidden_size, tanh),
        Dense(hidden_size, 1)
    )
    # otimizador Adam
    opt = Adam(0.01)
    # loop de treino
    for epoch=1:200
        # calcular erro e gradiente em treino
        gs = gradient(Flux.params(model)) do
            y_hat = model(X_train')
            l = sum((y_hat - y_train').^2)
        end
        
        # atualizar pesos da rede
        Flux.update!(opt, Flux.params(model), gs)
    end
    
    # salvar erro final de treino e validação
    mse_trn = mean((model(X_train') - y_train').^2)
    mse_val = mean((model(X_val') - y_val').^2)
    push!(trn_loss, mse_trn)
    push!(val_loss, mse_val)
end
# plot dos erros por tamanho de modelo
plot(hidden_size_list, trn_loss, label="Train", title="Val MSE / Hidden Size")
plot!(hidden_size_list, val_loss, label="Validation")
hidden_size_list[argmin(val_loss)]
savefig("plot2")

# utilizar melhor modelo em validação considerando termo de regularização
model = Chain(
    Dense(p => 28, tanh),
    Dense(28, 1)
)
# otimizador Adam
opt = Adam(0.01)
val_loss = [mean((model(X_train') - y_train').^2)]
trn_loss = [mean((model(X_val') - y_val').^2)]
# loop de treino
for epoch in ProgressBar(1:200)
    # calcular erro e gradiente em treino
    gs = gradient(Flux.params(model)) do
        
        # termo de regularização
        W1, b1, W2, b2 = Flux.params(model)
        regul_term = sum(W1.^2) + sum(b1.^2) + sum(W2.^2) + sum(b2.^2)

        y_hat = model(X_train')
        l = sum((y_hat - y_train').^2) + regul_term
    end
    
    # atualizar pesos da rede
    Flux.update!(opt, Flux.params(model), gs)

    # avaliar nos conjuntos de treino e validação
    mse_trn = mean((model(X_train') - y_train').^2)
    mse_val = mean((model(X_val') - y_val').^2)
    push!(trn_loss, mse_trn)
    push!(val_loss, mse_val)
end

plot(trn_loss, label="Train", title="MSE / Epoch")
plot!(val_loss, label="Validation")
savefig("plot3")