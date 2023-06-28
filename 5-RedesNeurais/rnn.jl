"""
Recurrent Neural Networks
"""

# instalação de pacotes necessários
using Pkg
for lib in []
    if ~in(lib , keys(Pkg.installed()))
        Pkg.add(lib)
    end
end

# preparação de ambiente
using Random, Distributions
using MLUtils
using MLBase
using ProgressBars
using Flux
using Plots

Random.seed!(0)

# simule uma série temporal
n = 305
x1 = rand(Normal(0, 1), n) # ruido branco
β = rand(5)
x2 = [β'*x1[i-5:i-1] for i=6:n] # auto regressivo (5)
x3 = cumsum(x2) # integração
x4 = x3 + 10 * [sin(i/5) for i=1:n-5] # sazonalidade senoidal
s = (x4 .- mean(x4)) / std(x4) # normalizada
plot(s, label="Série Temporal criada")
savefig("plot4")

# separar dados em treino e teste
s_train = s[1:Int(0.8*size(s)[1])]
s_test  = s[size(s_train)[1]+1:end]

# definir e treinar uma rede recorrente
# foi definida uma rede recorrente com uma
# camada densa antes da previsão
recurrent = Flux.Recur(
    Flux.RNNCell(1 => 8, tanh),
)
model = Chain(recurrent, Dense(8 => 1))
# otimizador Adam
opt = Adam(0.005)

"""
Função de previsão recorrente de conjunto de dados
"""
function predict(x)
    Flux.reset!(model)
    return [model([xi])[1] for xi in x] 
end

# loop de treino salvando mse de treino
init_err = sum((
    predict(s_train[1:end-1]) - s_train[2:end]
).^2)
loss = [init_err]
for epoch in ProgressBar(1:50)
    # calcular erro e gradiente em treino
    gs = gradient(Flux.params(model)) do
        l = sum((predict(s_train[1:end-1]) - s_train[2:end]).^2)
    end
    
    # atualizar pesos da rede
    Flux.update!(opt, Flux.params(model), gs)

    # avaliar novamente
    err = sum((predict(s_train[1:end-1]) - s_train[2:end]).^2)
    push!(loss, err)
end

# plot dos erros por época
plot(loss, label="Train", title="MSE / Epoch")
savefig("plot5")
loss

plot(s_train[2:end], label="Series", title="Performance do modelo nos dados de treino")
plot!(predict(s_train[1:end-1]), label="Prediction")


plot(s[2:end], label="Series", title="Performance do modelo na série completa")
plot!(predict(s[1:end-1]), label="Prediction")
savefig("plot6")
