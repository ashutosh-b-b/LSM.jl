using LinearAlgebra , Polynomials , StochasticDiffEq
N = 4          # number of time steps
r = 0.06       # interest rate
K = 1.1        # strike
T = 3          # Maturity

dt = T/(N-1)          # time interval
df = exp(-r * dt)
M = 10
S0 = fill(100.0 , M)
r = 0.04
sigma = 0.2
f(u , p , t) = r
g(u , p , t) = sigma
prob = SDEProblem(f, g, S0 , (0.0 , 1.0))
sol = solve(prob , EM() , dt = 1/N-1)
for u in sol.u
  S0 = hcat(S0 , u)
end
S0 = S0[: ,2:end]
H = S0
for i in CartesianIndices(H)
  H[i] = max(K - H[i] , 0.00)
end
V = zeros(size(H))
V[:,end] = H[:,end]
for t in N-1:-1:1
  paths = H[:, t] .> 0
  poly = polyfit(S0[paths , t] , V[paths , t + 1], 2)
  C = polyval(poly , S0[paths , t])
  exc = fill(false , length(paths))
  exercise[paths] = H[paths , t] .> C
  V[exercise,t] = H[exercise,t]
  V[exercise,t+1:] = 0
  discount_path = (V[:,t] == 0)
  V[discount_path,t] = V[discount_path,t+1] * df
end
price = mean(V[: , 2])*df
