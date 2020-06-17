using LinearAlgebra , Polynomials , StochasticDiffEq
using Polynomials.PolyCompat , Distributions

N = 4          # number of time steps
K = 110.00        # strike
T = 3.00          # Maturity
M = 10
S0 = fill(100.0 , M)
r = 0.04         #interest_rate
sigma = 0.2      #volatility

function LSM_pricing(r , sigma, K , S0 , T , N , M)
  dt = T/(N-1)          # time interval
  df = exp(-r*dt)
  f(u , p , t) = r.*u
  g(u , p , t) = sigma.*u
  prob = SDEProblem(f, g, S0 , (0.0 , T))
  sol = solve(prob , EM() , dt = dt)
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
    poly = Polynomials.fit(S0[paths , t] , V[paths , t + 1].* df, 2)
    C = poly.(S0[paths , t])

    exercise = fill(false , length(paths))
    exercise[paths] = H[paths , t] .> C
    V[exercise,t] = H[exercise,t]
    for t_ in t+1:1:N-1
      V[exercise , t_] .= 0
    end
    discount_path = (V[:,t] .== 0)

    V[discount_path,t] = V[discount_path,t+1] * df
  end
  price = mean(V[: , 2])*df
  return price
end

LSM_pricing(r , sigma , K , S0 , T , N , M)
