import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt
import oapackage
from kneed import KneeLocator
k = 12.413
v = 0.0014
epson = 0.0667 # The probability of de novo resistance development
gamma = 0.167 # The rate of normal recovery
gamma_a = 0.379 # The rate of recovery with antibiotics (only works in sensitive strains)
mu = 0.035 # Rate of mortality due to disease

alpha = 0.5 # Fitness cost
average_expenditure = 0.0153 # Cost of maintenance animals per day per animal
antibiotic_cost = 0.0019 # cost of antibiotic per day per chicken
W = 2.406 # Average weight of a healthy chicken
P = 0.83 #Price per kg of final chicken
Wloss = 0.1389
sigma = gamma * Wloss * W
p = 0.02
day_max = 35
IS0_relative = 0.05452
IR0_relative = 0.00348
density_min = 5
density_max = 25
current = 21
A = 1996.4 #Average size of farm in square meters

if __name__ == "__main__":
    densities = np.linspace(density_min, density_max, 1000)
    t = np.linspace(0,42,1000)

    xplot = np.array([])
    yplot = np.array([])
    yplot2 = np.array([])
    y_revenue = np.array([])
    for density in densities:
        # Initial State Values
        S0 = density * A -  (density * A * (IS0_relative + IR0_relative))
        IS0 = density * A * IS0_relative
        IR0 = IR0_relative * density * A
        R0 = 0
        D0 = 0
        N0 = S0 + IS0 + IR0 + R0 + D0


        def SIR_MODEL(y,t, k, v, gamma,gamma_a, mu, alpha, epson):
            S, IS, IR, R, D,N = y
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

        result = odeint(SIR_MODEL, y0=[S0, IS0, IR0, R0, D0, N0], t= t, args=(k, v, gamma, gamma_a, mu, alpha, epson))

        S, IS, IR, R, D, N = result.T
        antibiotic_active = (D/N0 > p) & (day_max >= t)
        W_N = W - density * 0.0267
        gross_revenue = N[-1] * P * W_N
        rec_rate = np.where(antibiotic_active == False, gamma/(gamma + mu), (gamma_a * (IS/(IS + IR)) + gamma * (IR/(IR + IS)))/ (gamma_a * (IS/(IS + IR)) + gamma * (IR/(IR + IS)) + mu))
        weight_loss_cost = trapezoid((IR + IS) * rec_rate, t) * sigma
        maintenance = trapezoid(N, t) * average_expenditure
        antibiotic = trapezoid(N * antibiotic_active, t) * antibiotic_cost
        Revenue = gross_revenue - weight_loss_cost * P - maintenance - antibiotic
        xplot = np.append(xplot, density)
        FR = (trapezoid(IR, t)/ trapezoid(N, t)) * 100
        yplot = np.append(yplot, (trapezoid(IR, t)/ trapezoid(IR + IS, t) * 100))
        yplot2 = np.append(yplot2, FR)
        y_revenue = np.append(y_revenue, Revenue)

    datapoints = np.array([yplot2, y_revenue])
    pareto_data = np.array([-yplot2, y_revenue])
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
    kneedle_idx = KneeLocator(pareto_x, pareto_y)
    print(f"Knee point — FR: {kneedle_idx.knee:.2f}%, Revenue: {kneedle_idx.knee_y:.2f}€")
    print("Knee point density: ", xplot[np.where(yplot2 == kneedle_idx.knee)[0][0]])
    print(D[-1])
    print(trapezoid(IR, t))
    print(trapezoid(IS, t))
    print(IR[-1])

    plt.subplot(2, 2, 1)
    plt.plot(xplot, yplot2)
    plt.title('AMR')

    plt.xlabel('Density')
    plt.ylabel('Resistant animals (%)')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(xplot, y_revenue)

    plt.title('Revenue')
    plt.xlabel('Density')
    plt.ylabel('Revenue (€)')
    plt.grid(True)

    plt.subplot(2,2,3)

    plt.plot(datapoints[0, :], datapoints[1, :], ".b", label="Non Pareto")
    plt.plot(optimal_datapoints[0, :], optimal_datapoints[1, :], ".r", label="Pareto")
    plt.plot(kneedle_idx.knee, kneedle_idx.knee_y, ".g", label="Knee")
    index = np.where(np.isclose(xplot, current))
    plt.plot(datapoints[0, index], datapoints[1, index], ".", color='#5aa0d6', label="Current", markersize=10)
    plt.title('Pareto plot')
    plt.xlabel('FR')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid(True)

    plt.subplot(2,2,4)
    plt.plot(t, S, label="S")
    plt.plot(t, IS, label="IS")
    plt.plot(t, IR, label="IR")
    plt.plot(t, R, label="R")
    plt.plot(t, D, label="D")
    plt.grid(True)
    plt.show()

