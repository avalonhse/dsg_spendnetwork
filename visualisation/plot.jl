using Random

function make_link_plot(G; thinning=0.4, is_output=false, is_title=false,
                        marked_buyers=nothing, marked_supplier=nothing)
    n_buyers, n_suppliers = size(G)

    p = plt.figure(figsize=(7,11))

    rng = MersenneTwister(1234);

    for i_buyer in 1:n_buyers, j_supplier in 1:n_suppliers
        if G[i_buyer, j_supplier] > 0 && rand(rng) < thinning

            plt.plot([1.0, 2.0], [i_buyer - n_buyers / 2, j_supplier - n_suppliers / 2], "-",
                     c="black", alpha=0.2, linewidth=0.2)
        end
    end

    buyer_pos_x = ones(n_buyers)
    buyer_pos_y = [i for i = 1:n_buyers] .- n_buyers / 2
    supplier_pos_x = 2 * ones(n_suppliers)
    supplier_pos_y = [i for i = 1:n_suppliers] .- n_suppliers / 2

    c_buyer = ["red" for _ = 1:n_buyers]
    if marked_buyers != nothing
        for i in marked_buyers
            c_buyer[i] = "green"
        end
    end
    s_buyer = 10sum(G; dims=2)
    if marked_buyers != nothing
        s_buyer_bak = copy(s_buyer)
        s_buyer = s_buyer ./ s_buyer * 0.02
        for i in marked_buyers
            s_buyer[i] = s_buyer_bak[i]
        end
    end
    p_scatter_buyer = plt.scatter(buyer_pos_x, buyer_pos_y,
                                  alpha=0.2, s=s_buyer, marker=".", c=c_buyer, label="buyer")
    c_supplier = ["orange" for _ = 1:n_suppliers]
    s_supplier = 10sum(G; dims=1)
    if marked_supplier != nothing
        s_supplier_bak = copy(s_supplier)
        s_supplier = s_supplier ./ s_supplier * 0.02
        for i in marked_supplier
            c_supplier[i] = "blue"
            s_supplier[i] = s_supplier_bak[i]
        end
    end
    p_scatter_supplier = plt.scatter(supplier_pos_x, supplier_pos_y,
                                     alpha=0.2, s=s_supplier, marker=".", c=c_supplier, label="supplier")
    plt.legend([p_scatter_buyer, p_scatter_supplier],
                ["Buyer", "Supplier"])
    plt.axis("off")

    if is_title
        plt.title("Links from `ocds_contracts_clean.csv` ($(100thinning)% links)")
    end

    if is_output
        for fmt in ["pdf", "png"]
            plt.savefig("buyer-supplier-popular.$fmt")
        end
    end

    return p
end
