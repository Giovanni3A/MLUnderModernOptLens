"""
Lista 12 - Prescriptive Analysis
"""

# preparando o ambiente
using Random, Distributions
using JuMP, Gurobi
using Plots
using ProgressBars

Random.seed!(1)
const GUROBI_ENV = Gurobi.Env()

# gerar conjunto de dados
n = 100
temp = round.(rand(Normal(28, 5), n), digits=1)
week = ones(n)
week[rand(n) .>= 5/7] .= 0

demn = rand(Poisson(150), n)
demn .-= (temp.>35) .* rand(Poisson(50), n)
demn .-= week .* rand(Poisson(30), n)

# resolva o problema do jornaleiro com SAA
c = 10
r = 5
q = 25
u = 150
saa_model = Model(Gurobi.Optimizer)
@variables(saa_model, begin
    x_saa ≥ 0
    y_saa[1:n] ≥ 0
    z_saa[1:n] ≥ 0
end)
@constraints(saa_model, begin
    x_saa ≤ u
    y_saa .≤ demn
    y_saa + z_saa .≤ x_saa
end)
@objective(saa_model, Max, -c*x_saa + 1/n*sum(q*y_saa + r*z_saa))
optimize!(saa_model)
value(x_saa)
objective_value(saa_model)

# estime um modelo que preveja demanda em função de temperatura e dia da semana

# modelo de semana
idx1 = week.==1
pred1 = Model(Gurobi.Optimizer)
@variable(pred1, β1[1:2])
@objective(pred1, Min, sum((demn[idx1] .- β1[2]*temp[idx1] .- β1[1]).^2))
optimize!(pred1)
value.(β1)

# modelo de fim de semana
idx2 = week.==0
pred2 = Model(Gurobi.Optimizer)
@variable(pred2, β2[1:2])
@objective(pred2, Min, sum((demn[idx2] .- β2[2]*temp[idx2] .- β2[1]).^2))
optimize!(pred2)
value.(β2)

β = [value.(β1)'; value.(β2)']

function predict_demand(temp, week)
    if week == 1
        return β[1,1] + β[1,2]*temp
    else
        return β[2,1] + β[2,2]*temp
    end
end

# resolver o problema utilizando ER-SAA para 2 amostras

"""
Cálculo do custo a partir de solução e demanda dadas
"""
function cost_x(x, d)
    venda = minimum([x, d])
    sobra = x-venda

    return -c*x + q*venda + r*sobra
end

"""
Implementação do modelo ER SAA
"""
function er_saa_jornaleiro(t, w)
    
    # calcular os residuos
    ϵ1 = value.(demn[idx1] .- β1[2]*temp[idx1] .- β1[1])
    ϵ2 = value.(demn[idx2] .- β2[2]*temp[idx2] .- β2[1])

    # calcular a previsão
    f = predict_demand(t, w)

    er_demn = round.(f .+ [ϵ1; ϵ2])    
    er_model = Model(Gurobi.Optimizer)
    @variables(er_model, begin
        x_er ≥ 0
        y_er[1:n] ≥ 0
        z_er[1:n] ≥ 0
    end)
    @constraints(er_model, begin
        x_er ≤ u
        y_er .≤ er_demn
        y_er + z_er .≤ x_er
    end)
    @objective(er_model, Max, -c*x_er + 1/n*sum(q*y_er + r*z_er))
    optimize!(er_model)
    return value(x_er), objective_value(er_model)
end

i = 1
temp[i], week[i], demn[i]
x_i, cost_i = er_saa_jornaleiro(temp[i], week[i])
cost_x(x_i, demn[i])
cost_x(value(x_saa), demn[i])
value.(x_saa)

i = 2
temp[i], week[i], demn[i]
x_i, cost_i = er_saa_jornaleiro(temp[i], week[i])
cost_x(x_i, demn[i])
cost_x(value(x_saa), demn[i])
value.(x_saa)
