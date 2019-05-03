using CSV, DataFrames, DelimitedFiles
import PyPlot; const plt = PyPlot

include("data.jl")

df = CSV.File("./data/SpendNetwork/ocds_contracts_clean.csv") |> DataFrame
G, G_number, G_money = get_matrices(df)
n_buyers, n_suppliers = size(G)

include("plot.jl")

make_link_plot(G; is_title=false)
plt.gcf()

### PCA baseline

using LinearAlgebra, MultivariateStats

include("utility.jl")

function compute_embedding(G; opt1=1, opt2=1, maxoutdim=100)
    n_buyers, n_suppliers = size(G)

    if opt1 == 1
        A = complete_adj(G)
        println("A = [[G 0], [0 G']] constructed")
    elseif opt1 == 2
        A = G' * G
        println("A = G' * G computed")
    elseif opt1 == 3
        A = G * G'
        println("A = G * G' computed")
    else
        error("invalid opt1=$opt1")
    end

    D = diagm(0 => vec(sum(A; dims=1)))
    println("D computed")

    if opt2 == 1
        L = D - A
        println("L = D - A computed")
    elseif opt2 == 2
        L = inv(D) * A
        println("L = inv(D) * A computed")
    else
        error("invalid opt2=$opt2")
    end

    E = eigen(L)
    println("Eigen decomposition completed")

    return E.vectors, E.values
end

is_generate_all = false
if is_generate_all
    prinvecs = nothing
    prinvars = nothing
    for opt1=1:3, opt2=1:2
        global prinvecs, prinvars
        try
            prinvecs, prinvars = compute_embedding(G_money; opt1=opt1, opt2=opt2)
            open("./models/pca-money-$opt1-$opt2.csv", "w") do io
                writedlm(io, prinvecs)
            end
        catch
            println("money-$opt1-$opt2 failed")
        end
    end
end

is_vis_graph_embedding = true
if is_vis_graph_embedding
    prinvecs, prinvars = compute_embedding(G_number; opt1=1, opt2=1)
    Y = prinvecs[:,1:10]

    plt.figure(figsize=(7,7))
    i_plot = 1
    row_max = 5
    col_max = 5
    for i=1:row_max, j=1:col_max
        global i_plot
        plt.subplot(row_max, col_max, i_plot)
        if i != j
            plt.scatter(Y[:,i], Y[:,j], c="red", alpha=0.2)
        else
            plt.axis("off")
        end
        i_plot += 1
    end

    plt.gcf()
end
