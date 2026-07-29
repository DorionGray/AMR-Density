import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt
import oapackage
from kneed import KneeLocator


kv = 0.2
epson = 0.125 # The probability of de novo resistance development
gamma = 0.143 # The rate of normal recovery
gamma_a = 0.715 # The rate of recovery with antibiotics (only works in sensitive strains)
mu = 0.00886  # Rate of mortality due to disease
alpha = 0.36 # Fitness cost
average_expenditure = 2.42 # Cost of maintenance animals per day per animal
antibiotic_cost = 0.075 # cost of antibiotic per day per chicken
W = 18.14 # Average weight of a healthy weaner
P = 6.08 #Price per kg of final chicken
current = 2.4
Wloss = 0.189
sigma = Wloss * gamma * W
p = 0.008
day_max = 31
IS0_relative = 0.162
IR0_relative = 0.018
density_min = 0.5
density_max = 3
A = 874.12 #Average size of farm in square meters

densities = np.linspace(density_min, density_max, 1000)
t = np.linspace(0,35,1000)

xplot = np.array([])
yplot = np.array([])

y_revenue = np.array([])
yrev_norm = np.array([])
yamr_norm = np.array([])
y_final = np.array([])
for density in densities:
    # Initial State Values
    S0 = density * A -  (density * A * (IS0_relative + IR0_relative))
    IS0 = density * A * IS0_relative
    IR0 = IR0_relative * density * A
    R0 = 0
    D0 = 0
    N0 = S0 + IS0 + IR0 + R0 + D0


    def SIR_MODEL(y,t, kv, gamma,gamma_a, mu, alpha, epson):
        S, IS, IR, R, D,N = y
        lambda_S = kv * IS / A
        lambda_R = kv * IR * (1 - alpha) / A
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

    result = odeint(SIR_MODEL, y0=[S0, IS0, IR0, R0, D0, N0], t= t, args=(kv, gamma, gamma_a, mu, alpha, epson), tcrit=[day_max], mxstep=5000)

    S, IS, IR, R, D, N = result.T
    antibiotic_active = (D/N0 > p) & (day_max >= t)
    gross_revenue = N[-1] * P * W
    maintenance = trapezoid(N, t) * average_expenditure
    antibiotic = trapezoid(N * antibiotic_active, t) * antibiotic_cost
    rec_rate = np.where(antibiotic_active == False, gamma / (gamma + mu), (gamma_a * (IS / (IS + IR)) + gamma * (IR / (IR + IS))) / (gamma_a * (IS / (IS + IR)) + gamma * (IR / (IR + IS)) + mu))
    weight_loss = trapezoid((IR + IS) * rec_rate * sigma, t)
    Revenue = gross_revenue - weight_loss * P - maintenance - antibiotic
    xplot = np.append(xplot, density)
    FR = (trapezoid(IR, t)/ trapezoid(N, t)) * 100
    yplot = np.append(yplot, FR)
    y_revenue = np.append(y_revenue, Revenue)

yamr_norm = (yplot - yplot.min()) / (yplot.max() - yplot.min())
yrev_norm = (y_revenue - y_revenue.min()) / (y_revenue.max() - y_revenue.min())
y_final = yamr_norm - yrev_norm
index_of = np.where(y_final == y_final.min())
datapoints = np.array([yplot, y_revenue])
pareto_data = np.array([-yplot, y_revenue])
pareto = oapackage.ParetoDoubleLong()

for ii in range(0, pareto_data.shape[1]):
    w = oapackage.doubleVector((pareto_data[0, ii], pareto_data[1, ii]))
    pareto.addvalue(w, ii)

pareto.show(verbose=1)
lst = pareto.allindices()
optimal_datapoints = datapoints[:, lst]

sorted_idx = np.argsort(optimal_datapoints[0, :])
pareto_x = optimal_datapoints[0, sorted_idx]
pareto_y = optimal_datapoints[1, sorted_idx]
ki = KneeLocator(pareto_x, pareto_y)
print(f"Knee point — FR: {ki.knee:.2f}%, Revenue: {ki.knee_y:.2f}€")
knee_density_idx = np.where(y_revenue == ki.knee_y)[0]
print("Knee point density: ", xplot[knee_density_idx])

print(D[-1])
print(trapezoid(IR, t))
print(trapezoid(IS, t))
print(IR[-1])
plt.subplot(2, 2, 1)
plt.plot(xplot, yplot)
plt.title('AMR')
plt.axvline(2.5, color='r')
plt.xlabel('Density')
plt.ylabel('Resistant animals (%)')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(xplot, y_revenue)
plt.title('Revenue')
plt.axvline(2.5, color='r')
plt.xlabel('Density')
plt.ylabel('Revenue (€)')
plt.grid(True)
plt.subplot(2,2,3)

plt.plot(datapoints[0, :], datapoints[1, :], ".b", label="Non Pareto")
plt.plot(optimal_datapoints[0, :], optimal_datapoints[1, :], ".r", label="Pareto")
#plt.plot(ki.knee, ki.knee_y, ".g", label="Knee")
index = np.where(np.isclose(xplot, current))
plt.plot(datapoints[0, index], datapoints[1, index], ".", color='#5aa0d6', label="Current", markersize=10)
plt.title('Pareto plot')
plt.xlabel('FR')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)

plt.subplot(2,2,4)
plt.plot(t, S, label="Susceptible")
plt.plot(t, IS, label="Infected w/ susceptible strain")
plt.plot(t, IR, label="IR w/ resistant strain")
plt.plot(t, R, label="Recovered")
plt.plot(t, D, label="Diseased")
plt.legend()
plt.grid(True)

plt.show()

