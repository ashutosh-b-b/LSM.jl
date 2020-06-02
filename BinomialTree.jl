function BinomialTreeAM1D(S0 , N , r , beta)
    V = zeros(N+1)
    dT = T/N
    u = exp(beta*sqrt(dT))
    d = 1/u
    S_T = [S0*(u^j)* (d^(N-j)) for j in 0:N]
    a = exp(r*dT)
    p = (a - d)/(u - d)
    q = 1.0 - p
    V = [max(K - x , 0) for x in S_T]
    for i in N-1:-1:0
      V[1:end-1] = exp(-r*dT).*(p*V[2:end] + q*V[1:end-1])
      S_T = S_T*u
      V = [max(K - S_T[i] , V[i]) for i in 1:size(S_T)[1]]
    end
    return V[1]
end
