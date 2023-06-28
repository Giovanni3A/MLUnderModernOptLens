"""
Implementação e avaliação do modelo SARIMA
como MIP
"""

# preparação do ambiente
using CSV, DataFrames
using JuMP, Gurobi
using Plots, ProgressBars
using Distributions

const GUROBI_ENV = Gurobi.Env()

# leitura dos dados
df = CSV.read("AirPassengers.csv", DataFrame)
y = Array(df[2:end, end])
T = size(y)[1]

"""
Implementação do modelo SARIMA
como MIP para série temporal univariada utilizando diferenciação 1.

Entradas
--------
y: Vector{Float}
    Série temporal
p: Int
    Lags a serem considerados na auto-regressão
K: Int
    Parâmetro de regularização por norma zero
M: Float
    Big-M
"""
function sarima(y, p, K, M=10.0)

    T = size(y)[1]

    # calcular série com diferenciação
    Δy = [0; y[2:end] .- y[1:end-1]]

    # construção do modelo
    model = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GUROBI_ENV)
        )
    )
    MOI.set(model, MOI.Silent(), true)
    @variables(model, begin
        α
        β
        γ
        ϕ[1:p]
        ϵ[1:T]
    end)
    # desconsiderar resíduos para p períodos iniciais
    @constraint(model, [t=1:p], ϵ[t] == 0)
    # para os outros períodos, montar cálculo do ϵ
    @constraint(model, [t=p+1:T], Δy[t] == α + β*t + γ*y[t-1] + ϕ'*Δy[t-p:t-1] + ϵ[t])
    # regularização
    @variable(model, z[1:p+3], Bin)
    @constraints(model, begin
        α ≤ M*z[1]
        β ≤ M*z[2]
        γ ≤ M*z[3]
        ϕ .≤ M*z[4:end]
        sum(z) ≤ K
    end)
    @objective(model, Min, sum(ϵ.^2))

    # solucionar
    optimize!(model)

    # gerar previsão
    Δpred = Float64.([Δy[1:p]...])
    for t=p+1:T
        outp = value.(α + β*t + γ*y[t-1] + ϕ'*Δy[t-p:t-1])
        push!(Δpred, outp)
    end
    pred = y[1] .+ cumsum(Δpred)


    return value(α), value(β), value(γ), value.(ϕ), value.(ϵ), pred, Δpred
end

# tuning do hiper-parâmetro K
p = 25
aic = []
mse = []
for k in ProgressBar(1:p+3)
    α, β, γ, ϕ, ϵ, pred, Δpred = sarima(y, 25, k);
    push!(aic, 2*k + (size(pred)[1]-p)*log(var(ϵ[p+1:end])))
    push!(mse, mean((pred - y)[p+1:end].^2))
end

fig1 = plot(aic, title="Validação do parâmetro K (p=25)", label="")
xlabel!("K")
ylabel!("AICC")
savefig("fig1.png")

k_star = argmin(aic)
preds = []
Δpreds = []
mse = []
p_list = [3,6,13,25]
# tuning do hiper-parâmetro p
for p in ProgressBar(p_list)
    α, β, γ, ϕ, ϵ, pred, Δpred = sarima(y, p, k_star);
    push!(Δpreds, Δpred)
    push!(preds, pred)
    push!(mse, mean((pred - y)[p+1:end].^2))
end

fig2 = plot(
    p_list, mse, 
    markersize = 5, markershape = :hexagon,
    title="Validação do parâmetro p (K=$k_star)", label="")
xlabel!("p")
ylabel!("MSE")
savefig("fig2.png")

figs3 = []
i = 1
for p in p_list
    fig3_p = plot(1:T, Δy, title="K=$k_star | p=$p", label="Δy")
    plot!(1:T, Δpreds[i], label="Previsão de Δy")
    push!(figs3, fig3_p)
    i += 1
end
fig3 = plot(figs3..., size=(800, 800))
display(fig3)
savefig("fig3.png")

figs4 = []
i = 1
for p in p_list
    fig4_p = plot(1:T, y, title="K=$k_star | p=$p", label="y")
    plot!(1:T, preds[i], label="Previsão de y")
    push!(figs4, fig4_p)
    i += 1
end
fig4 = plot(figs4..., size=(800, 800))
display(fig4)
savefig("fig4.png")
