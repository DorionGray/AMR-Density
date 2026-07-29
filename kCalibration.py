import numpy as np
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt


v = 0.0014
epson = 0.0667  # Probability of de novo resistance development
gamma =  0.167 # Rate of natural recovery
gamma_a = 0.379  # Rate of recovery with antibiotics (sensitive only)
mu = 0.035  # Mortality rate due to disease
alpha = 0.5  # Fitness cost of resistance
p = 0.02
N0      = 1996.4 * 18.6     # birds at start of cycle
A       = 1996.4        # m^2 pavilion area
T_CYCLE = 42           # days in a production cycle
day_max = 35           # last permissible day of antibiotic treatment

times = np.linspace(0.001, 100, 1000)
t = np.linspace(1, 42, 1000)
xplot = np.array([])
yplot = np.array([])
goal = 0.0444 - 0.018
for k in times:
    density = 18.6
    S0 = density * A - (density * A * 0.058)
    IS0 = density * A * 0.05452
    IR0 = 0.00348 * density * A
    R0 = D0 = 0
    N0 = S0 + IS0


    def SIR_MODEL(y, t, k, v, gamma, gamma_a, mu, alpha, epson):
        S, IS, IR, R, D, N = y
        lambda_S = k * v * IS / A
        lambda_R = k * v * IR * (1 - alpha) / A
        treatment = (D / N0 >= p and day_max >= t)
        dS = -(lambda_S + lambda_R) * S

        if treatment:
            dIS = lambda_S * S - (gamma_a + mu + epson) * IS
            dIR = epson * IS + lambda_R * S - (gamma + mu) * IR
            dR = gamma * IR + gamma_a * IS
        else:
            dIS = lambda_S * S - (gamma + mu) * IS
            dIR = lambda_R * S - (gamma + mu) * IR
            dR = gamma * (IS + IR)
        dD = mu * (IS + IR)
        dN = -mu * (IS + IR)
        return [dS, dIS, dIR, dR, dD, dN]

    result = odeint(SIR_MODEL, y0 = [S0, IS0, IR0, R0, D0, N0],t=t, args=(k,  v, gamma, gamma_a, mu, alpha, epson))
    S, IS, IR, R, D, N = result.T

    squared = ((D[-1]/N0) - goal) ** 2
    xplot = np.append(xplot, k)
    yplot = np.append(yplot, squared)

i = np.argmin(yplot)
print(xplot[i])
plt.plot(xplot, yplot)
plt.xlabel('Candidate values for K')
plt.ylabel('Module of distance to the goal')
plt.show()