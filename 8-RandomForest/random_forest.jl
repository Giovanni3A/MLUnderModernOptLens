"""
Random Forest - Validação cruzada nos hiper-parâmetros
"""

# preparando o ambiente
using DecisionTree
using DataFrames

# leitura dos dados
features, labels = load_data("adult")
features = @view features[1:5000, :]
labels = @view labels[1:5000]

# lista de hiper-parâmetros a serem avaliados
n_subfeatures_list = [2, 6, 12, 14]
n_trees_list = [10, 15, 20]
max_depth_list = [5, 10, 20]
min_samples_leaf_list = [5, 10, 15]

# hiper-parâmetros fixos
partial_sampling = 0.7
min_samples_split = 2
min_purity_increase = 0.0

# validação cruzada
n_folds = 3
results = Array{Any}[]
for n_subfeatures in n_subfeatures_list,
    n_trees in n_trees_list,
    max_depth in max_depth_list,
    min_samples_leaf in min_samples_leaf_list

    # seed aleatória para replicação da divisão da base
    seed = 370

    # verbose
    println("\n n_subfeatures=$n_subfeatures | n_trees=$n_trees | max_depth=$max_depth | min_samples_leaf=$min_samples_leaf")

    # 3-fold CV
    accuracy = nfoldCV_forest(
        labels, 
        features,
        n_folds,
        n_subfeatures[1],
        n_trees[1],
        partial_sampling,
        max_depth[1],
        min_samples_leaf[1],
        min_samples_split,
        min_purity_increase;
        verbose=false,
        rng=seed
    )
    avg_acc = sum(accuracy) / size(accuracy)[1]

    # salvar resultados
    r = [n_subfeatures, n_trees, max_depth, min_samples_leaf, avg_acc]
    push!(results, r)
end

# analisando melhor resultado
results_df = DataFrame(
    n_subfeatures=[r[1] for r in results],
    n_trees=[r[2] for r in results],
    max_depth=[r[3] for r in results],
    min_samples_leaf=[r[4] for r in results],
    avg_acc=[r[5] for r in results],
)
sort!(results_df, [:avg_acc], rev=true)
results_df[1:5, :]
results_df[end-4:end, :]
n_subfeatures_star, 
n_trees_star, 
max_depth_star, 
min_samples_leaf_star = results_df[1, [:n_subfeatures, :n_trees, :max_depth, :min_samples_leaf]]

# comparar melhor configuração com árvore única
# obs: unicos hiper-parametros tunados aplicaveis são
# max_depth, min_samples_leaf
accuracy = nfoldCV_tree(
    labels, features,
    n_folds,
    1.0,
    Int(max_depth_star),
    Int(min_samples_leaf_star),
    min_samples_split,
    min_purity_increase;
    verbose=true,
    rng=370
)
sum(accuracy) / size(accuracy)[1]