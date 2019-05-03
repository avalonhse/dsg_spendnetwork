function complete_adj(G)
    n_buyers, n_suppliers = size(G)
    N = n_buyers + n_suppliers
    A = zeros(N, N)
    A[1:n_buyers,1:n_suppliers] .= G
    A[n_buyers+1:N,n_suppliers+1:N] .= G'
    return A
end
