from SIR import Wloss

import numpy as np
import oapackage
from kneed import KneeLocator
from scipy.integrate import odeint, trapezoid

from django.shortcuts import render

from .forms import SIRForm


def _solve_sir(density, t, k, v, gamma, gamma_a, mu, alpha, epson, A, p, day_max,
                IS0_relative, IR0_relative):
    S0  = density * A * (1 - IS0_relative - IR0_relative)
    IS0 = density * A * IS0_relative
    IR0 = density * A * IR0_relative
    N0  = S0 + IS0 + IR0

    def SIR_MODEL(y, t_val, k, v, gamma, gamma_a, mu, alpha, epson):
        S, IS, IR, R, D, N = y
        lam_S = k * v * IS / A
        lam_R = k * v * IR * (1 - alpha) / A
        treat = (D / N0 >= p) and (t_val <= day_max)
        dS = -(lam_S + lam_R) * S
        if treat:
            dIS = lam_S * S - (gamma_a + mu + epson) * IS
            dIR = epson * IS + lam_R * S - (gamma + mu) * IR
            dR  = gamma * IR + gamma_a * IS
        else:
            dIS = lam_S * S - (gamma + mu) * IS
            dIR = lam_R * S - (gamma + mu) * IR
            dR  = gamma * (IS + IR)
        dD = mu * (IS + IR)
        dN = -mu * (IS + IR)
        return [dS, dIS, dIR, dR, dD, dN]

    res = odeint(SIR_MODEL, y0=[S0, IS0, IR0, 0, 0, N0], t=t,
                 args=(k, v, gamma, gamma_a, mu, alpha, epson))
    return res.T


def _fr_and_revenue(S, IS, IR, R, D, N, t, gamma, gamma_a, mu, P, W, avg_exp, ab_cost, sigma, p, day_max):
    antibiotic_active = (D / N[0] > p) & (t <= day_max)
    gross_revenue = N[-1] * P * W
    rec_rate = np.where(antibiotic_active == False, gamma/(gamma + mu), (gamma_a * (IS/(IS + IR)) + gamma * (IR/(IR + IS)))/ (gamma_a * (IS/(IS + IR)) + gamma * (IR/(IR + IS)) + mu))
    weight_loss = trapezoid((IR + IS) * rec_rate, t) * sigma
    maintenance = trapezoid(N, t) * avg_exp
    antibiotic  = trapezoid(N * antibiotic_active, t) * ab_cost
    revenue     = gross_revenue - weight_loss * P - maintenance - antibiotic

    denom = trapezoid(N, t)
    fr    = (trapezoid(IR, t) / denom * 100) if denom > 0 else 0.0
    return fr, revenue


def _run_simulation(cd):
    k           = cd['k']
    v           = cd['v']
    epson       = cd['epson']
    gamma       = cd['gamma']
    gamma_a     = cd['gamma_a']
    mu          = cd['mu']
    alpha       = cd['alpha']
    p           = cd['p']
    day_max     = cd['day_max']
    IS0_relative = cd['IS0_relative']
    IR0_relative = cd['IR0_relative']
    A           = cd['A']
    W           = cd['W']
    P           = cd['P']
    Wloss      = cd['Wloss']
    sigma = Wloss * W * gamma
    avg_exp     = cd['average_expenditure']
    ab_cost     = cd['antibiotic_cost']
    current_density = cd['current_density']

    densities = np.linspace(cd['density_min'], cd['density_max'], 100)
    t = np.linspace(0, 42, 1000)

    xplot, yplot, y_revenue = [], [], []

    for density in densities:
        S, IS, IR, R, D, N = _solve_sir(density, t, k, v, gamma, gamma_a, mu, alpha, epson,
                                         A, p, day_max, IS0_relative, IR0_relative)
        fr, revenue = _fr_and_revenue(S, IS, IR, R, D, N, t, gamma, gamma_a, mu,
                                       P, W, avg_exp, ab_cost, sigma, p, day_max)

        xplot.append(density)
        yplot.append(fr)
        y_revenue.append(revenue)

    xplot     = np.array(xplot)
    yplot     = np.array(yplot)
    y_revenue = np.array(y_revenue)

    # Dynamics + FR/Revenue at the user's current stocking density
    S_cur, IS_cur, IR_cur, R_cur, D_cur, N_cur = _solve_sir(
        current_density, t, k, v, gamma, gamma_a, mu, alpha, epson,
        A, p, day_max, IS0_relative, IR0_relative)
    fr_current, revenue_current = _fr_and_revenue(
        S_cur, IS_cur, IR_cur, R_cur, D_cur, N_cur, t, gamma, gamma_a, mu,
        P, W, avg_exp, ab_cost, sigma, p, day_max)

    # Pareto analysis
    datapoints  = np.array([yplot, y_revenue])
    pareto_data = np.array([-yplot, y_revenue])
    pareto = oapackage.ParetoDoubleLong()
    for ii in range(pareto_data.shape[1]):
        w = oapackage.doubleVector((float(pareto_data[0, ii]), float(pareto_data[1, ii])))
        pareto.addvalue(w, ii)
    lst = pareto.allindices()
    optimal_datapoints = datapoints[:, lst]

    sorted_idx = np.argsort(optimal_datapoints[0, :])
    pareto_x   = optimal_datapoints[0, sorted_idx]
    pareto_y   = optimal_datapoints[1, sorted_idx]

    knee_fr = knee_revenue = knee_density = None
    if len(pareto_x) > 1:
        knee_idx = KneeLocator(pareto_x, pareto_y)
        if knee_idx is not None:
            knee_fr      = float(knee_idx.knee)
            knee_revenue = float(knee_idx.knee_y)
            match = np.where(np.isclose(y_revenue, knee_revenue))[0]
            if len(match) > 0:
                knee_density = float(xplot[match[0]])

    return dict(
        xplot=xplot, yplot=yplot, y_revenue=y_revenue,
        datapoints=datapoints, optimal_datapoints=optimal_datapoints,
        pareto_x=pareto_x, pareto_y=pareto_y,
        t=t, S_cur=S_cur, IS_cur=IS_cur, IR_cur=IR_cur, R_cur=R_cur, D_cur=D_cur,
        current_density=current_density, fr_current=fr_current, revenue_current=revenue_current,
        knee_density=knee_density, knee_fr=knee_fr, knee_revenue=knee_revenue,
    )


def _series(xs, ys):
    return [{'x': round(float(x), 4), 'y': round(float(y), 4)} for x, y in zip(xs, ys)]


def _build_chart_data(r):
    knee_point = []
    if r['knee_fr'] is not None:
        knee_point = [{'x': round(r['knee_fr'], 4), 'y': round(r['knee_revenue'], 4)}]

    current_density = round(float(r['current_density']), 4)
    current_fr       = round(float(r['fr_current']), 4)
    current_revenue  = round(float(r['revenue_current']), 4)

    return {
        'amr': {
            'sweep':   _series(r['xplot'], r['yplot']),
            'current': [{'x': current_density, 'y': current_fr}],
        },
        'revenue': {
            'sweep':   _series(r['xplot'], r['y_revenue']),
            'current': [{'x': current_density, 'y': current_revenue}],
        },
        'pareto': {
            'all':     _series(r['datapoints'][0, :], r['datapoints'][1, :]),
            'optimal': _series(r['optimal_datapoints'][0, :], r['optimal_datapoints'][1, :]),
            'knee':    knee_point,
            'current': [{'x': current_fr, 'y': current_revenue}],
        },
        'dynamics': {
            'density': round(float(r['current_density']), 2),
            'S':  _series(r['t'], r['S_cur']),
            'IS': _series(r['t'], r['IS_cur']),
            'IR': _series(r['t'], r['IR_cur']),
            'R':  _series(r['t'], r['R_cur']),
            'D':  _series(r['t'], r['D_cur']),
        },
    }


def simulate(request):
    form = SIRForm(request.POST or None)
    chart_data = None
    results = None

    if request.method == 'POST' and form.is_valid():
        r = _run_simulation(form.cleaned_data)
        chart_data = _build_chart_data(r)
        results = {
            'knee_density':  round(r['knee_density'],   3) if r['knee_density']  is not None else None,
            'knee_fr':       round(r['knee_fr'],        2) if r['knee_fr']       is not None else None,
            'knee_revenue':  round(r['knee_revenue'],   2) if r['knee_revenue']  is not None else None,
        }

    return render(request, 'simulator/index.html', {
        'form': form,
        'chart_data': chart_data,
        'results': results,
    })
