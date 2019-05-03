function get_matrices(df)
    buyers = lowercase.(unique(df.buyer))
    n_buyers = length(buyers)
    buyer_to_id = Dict(buyers[i] => i for i = 1:length(buyers))
    suppliers = lowercase.(unique(df.supplier))
    n_suppliers = length(suppliers)
    supplier_to_id = Dict(suppliers[i] => i for i = 1:length(suppliers))

    @info "df" string(names(df)) size(df) n_buyers n_suppliers

    G = zeros(n_buyers, n_suppliers)
    G_number = zeros(n_buyers, n_suppliers)
    G_money = zeros(n_buyers, n_suppliers)

    for i = 1:size(df, 1)
        buyer_id = buyer_to_id[lowercase(df[i,:].buyer)]
        supplier_id = supplier_to_id[lowercase(df[i,:].supplier)]
        G[buyer_id, supplier_id] = 1
        G_number[buyer_id, supplier_id] += 1
        G_money[buyer_id, supplier_id] += df[i,:].clean_aw_val
    end

    return G, G_number, G_money, buyer_to_id, supplier_to_id
end
