"""
Convolutional Neural Networks
"""

# instalação de pacotes necessários
using Pkg
for lib in ["BSON", "MLDatasets", "ImageView"]
    if ~in(lib , keys(Pkg.installed()))
        Pkg.add(lib)
    end
end

# preparação de ambiente
using MLUtils
using MLBase
using ProgressBars
using Flux
using BSON
using MLDatasets: MNIST
using Plots
using ImageView

Random.seed!(0)

# inicializar modelo a partir de arquivo
BSON.@load "mnist_cnn.bson" model

# leitura e normalização do dataset
dataset = MNIST(split=:test)
X = dataset.features
y = dataset.targets
n = dataset.metadata["n_observations"]
normalized = @. 2.0f0 * X - 1.0f0
X = reshape(normalized, size(X, 1), size(X, 2), 1, :)
y_space = 0:9

# definir amostra do conjunto de teste e avaliar modelo
idx_test = shuffleobs(1:n)[1:1000]
X_test = X[:, :, :, idx_test]
y_test = y[idx_test]

# avaliar algumas predições
for i=1:10
    model_output = model(X_test[:, :, :, i:i])
    println("i=$i | y_true = $(y_test[i]) | y_hat = $(y_space[argmax(model_output)])")
end
# verificar visualmente alguns dos dígitos
i = 10
imshow(X_test[:, :, :, i])

# calcular a acurácia do modelo para o conjunto definido
model_outp = model(X_test)
y_hat = [y_space[c[1]] for c in argmax(model_outp, dims=(1))]
acc = mean(y_hat[1, :] .== y_test)
println("Acurácia calculada = $(round(100*acc, digits=3))%")

# aplicar função de ataque adversário
"""
Given a flux model `cnn` and an example `x`, calculates the gradient for each
input pixel. A new image, `perturbed_image` is created with a noise controlled by
`epsilon`. Values pass through `clamp` to ensure it stays on the expected domain
of the trained `cnn`
"""
function fgsm_attack(cnn, x, attack_label::Int = 1, epsilon = 5e-2)
    grad = Flux.jacobian(x -> cnn(x), x)[1] # (labels)
    data_grad = reshape(grad, :, 28, 28)
    perturbed_image = clamp.(
    x + epsilon .* sign.(data_grad[attack_label, :, :]),
        -1., 1.
    )
    return perturbed_image
end

for ϵ in [0, 1e-2, 1e-1, 1, 10]
    right_ans = 0
    for i=1:100
        perturbed_x = fgsm_attack(model, X_test[:, :, :, i:i], y_test[i]+1, ϵ)
        if y_test[i] == y_space[argmax(model(perturbed_x))]
            right_ans += 1
        end
    end
    acc = 100 * round(right_ans / 100, digits=5)
    println("ϵ = $ϵ | acc = $acc")
end

# uma forma de reduzir o efeito desse ataque é adicionando
# samples de dados perturbados em um conjunto de refinamento
# do modelo após o treino