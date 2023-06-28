using CSV, DataFrames

# path to data folder
data_folder = "dados"

function get_data(filename)
    
    # path to data file
    data_path = joinpath(data_folder, filename)

    # red wine quality
    # all columns are numerical
    if filename == "winequality-red.csv"
        df = CSV.read(data_path, DataFrame)
        data = Array(df)

    elseif filename == "Twitter.data"
        df = CSV.read(data_path, DataFrame, header=false, limit=10000)
        data = Array(df)[:, end-20:end]

    elseif filename == "ObesityDataSet_raw_and_data_sinthetic.csv"
        df = CSV.read(data_path, DataFrame)
        # get objective value
        y = df[:, end]
        y_dict = Dict([
            ("Normal_Weight", 1),
            ("Overweight_Level_I", 2),
            ("Overweight_Level_II", 3),
            ("Obesity_Type_I", 4),
            ("Insufficient_Weight", 0),
            ("Obesity_Type_II", 5),
            ("Obesity_Type_III", 6)
        ])
        y = [y_dict[x] for x ∈ y]
        df = df[:, 1:end-1]
        # create numerical array
        numerical_df = df[:, eltype.(eachcol(df)) .== Float64]
        data = Array(numerical_df)
        # one hot enconde non numerical columns
        non_numerical_df = df[:, Not(names(numerical_df))]
        for col in names(non_numerical_df)
            combine(groupby(df, "Gender"), nrow)
            v = df[:, col]
            data = [data (unique(v) .== permutedims(v))'[:, 1:end-1]]
        end
        # add final column
        data = [data y]

    # all columns are numerical
    elseif filename == "Supercondut_train.csv"
        df = CSV.read(data_path, DataFrame)
        data = Array(df)[1:1000, :]
    end

    return data
end
