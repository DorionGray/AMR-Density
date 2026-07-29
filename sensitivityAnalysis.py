
import numpy as np
from scipy.stats import qmc, rankdata, t as student_t, truncnorm
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import oapackage
from kneed import KneeLocator


PARAM_RANGES = {
    "k":       (7.308,    17.518, 12.413),
    "v":       (0.0001,  0.0028, 0.0014),
    "gamma":   (0.0837, 0.2503, 0.0833),
    "gamma_T": (0.325,   0.433, 0.379),
    "mu":      (0.0204,  0.0496, 0.035),
    "epson":   (0.0589,  0.0745, 0.0667),
    "alpha":   (0.34,    0.66, 0.5),
}

A            = 1996.4
DENSITY_SA   = 21
DAYS         = 42
p            = 0.02
day_max      = 35
W            = 2.406
PRICE        = 0.83
WLOSS        = 0.1389
C_MAINT      = 0.0153
C_ANTIBIOTIC = 0.0019

DENSITY_MIN      = 5.0
DENSITY_MAX      = 25.0
N_SIM            = 1000
N_KNEE           = 200   # LHS samples for the knee sweep (each needs ~50 ODE runs)
N_KNEE_DENSITIES = 50    # density grid for each knee sweep


def run_model(params, density):
    k, v         = params["k"], params["v"]
    gamma, gamma_T = params["gamma"], params["gamma_T"]
    mu, epson, alpha = params["mu"], params["epson"], params["alpha"]

    N0 = density * A
    S0, IS0, IR0, R0, D0 = 0.942 * N0, 0.058 * N0, 0.0, 0.0, 0.0

    def SIR_MODEL(y, t, k, v, gamma, gamma_a, mu, alpha, epson):
        S, IS, IR, R, D, N = y
        lambda_S  = k * v * IS / A
        lambda_R  = k * v * IR * (1 - alpha) / A
        treatment = (D / N0 >= p and day_max >= t)
        dS = -(lambda_S + lambda_R) * S
        if treatment:
            dIS = lambda_S * S - (gamma_a + mu + epson) * IS
            dIR = epson * IS + lambda_R * S - (gamma + mu) * IR
            dR  = gamma * IR + gamma_a * IS
        else:
            dIS = lambda_S * S - (gamma + mu) * IS
            dIR = lambda_R * S - (gamma + mu) * IR
            dR  = gamma * (IS + IR)
        dD = mu * (IS + IR)
        dN = -mu * (IS + IR)
        return [dS, dIS, dIR, dR, dD, dN]

    t   = np.linspace(0, DAYS, DAYS * 4 + 1)
    sol = odeint(SIR_MODEL, [S0, IS0, IR0, R0, D0, N0], t,
                 args=(k, v, gamma, gamma_T, mu, alpha, epson))
    S, IS, IR, R, D, N = sol.T

    cum_IR = np.trapezoid(IR, t)
    cum_N = np.trapezoid(N, t)
    cum_IS = np.trapezoid(IS, t)
    F_R    = cum_IR / cum_N if (cum_N) > 0 else 0.0
    P_R = cum_IR / (cum_IR + cum_IS) if (cum_IR + cum_IS) > 0 else 0.0
    antibiotic_active = (D / N0 > p) & (day_max >= t)
    W_N         = W - density * 0.0267
    gross       = N[-1] * PRICE * W_N
    MC          = np.trapezoid(N, t) * C_MAINT
    AC          = np.trapezoid(N * antibiotic_active, t) * C_ANTIBIOTIC
    rec_rate    = np.where(
        ~antibiotic_active,
        gamma / (gamma + mu),
        (gamma_T * (IS / (IS + IR)) + gamma * (IR / (IS + IR)))
        / (gamma_T * (IS / (IS + IR)) + gamma * (IR / (IS + IR)) + mu),
    )
    sigma       = WLOSS * gamma * W
    weight_loss = np.trapezoid((IR + IS) * rec_rate * sigma, t)
    revenue     = gross - weight_loss * PRICE - MC - AC

    return F_R, revenue, P_R, D[-1], cum_IR, cum_IS, IR[-1]


def find_knee(params, densities):
    frs, revs = [], []
    for d in densities:
        fr, rev, *_ = run_model(params, d)
        frs.append(fr * 100)
        revs.append(rev)
    frs  = np.array(frs)
    revs = np.array(revs)

    pareto_data = np.array([-frs, revs])
    pareto = oapackage.ParetoDoubleLong()
    for ii in range(pareto_data.shape[1]):
        w = oapackage.doubleVector((float(pareto_data[0, ii]), float(pareto_data[1, ii])))
        pareto.addvalue(w, ii)
    lst = list(pareto.allindices())
    if len(lst) < 2:
        return None, None, None

    opt  = np.array([frs[lst], revs[lst]])
    sidx = np.argsort(opt[0, :])
    px, py = opt[0, sidx], opt[1, sidx]

    try:
        ki = KneeLocator(frs, revs)
        if ki is None:
            return None, None, None
        knee_fr  = float(ki.knee)
        knee_rev = float(ki.knee_y)
        match    = np.where(np.isclose(revs, knee_rev))[0]
        knee_d   = float(densities[match[0]]) if len(match) > 0 else None
        return knee_d, knee_fr, knee_rev
    except Exception:
        return None, None, None


def draw_lhs(n_samples, seed=0):
    names = list(PARAM_RANGES.keys())
    lows  = np.array([PARAM_RANGES[p][0] for p in names])
    highs = np.array([PARAM_RANGES[p][1] for p in names])
    unit  = qmc.LatinHypercube(d=len(names), seed=seed).random(n_samples)
    sample = np.empty_like(unit)
    for j, name in enumerate(names):
        mean = PARAM_RANGES[name][2]
        std = (highs[j]-lows[j])/ 6
        a = (lows[j] - mean) / std
        b = (highs[j] - mean) / std
        sample[:, j] = truncnorm.ppf(unit[:, j], a, b, loc=mean, scale=std)
    return names, sample


def lhs_scatter_plot(FR, REV, path="lhs_scatter.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(FR, REV, s=12, alpha=0.5, color="#5aa0d6", edgecolor="none")
    ax.set_xlabel("F_R")
    ax.set_ylabel("Revenue")
    ax.set_title("LHS samples: F_R vs revenue", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved -> {path}")


def evaluate(names, sample, density):
    FR       = np.empty(len(sample))
    REV      = np.empty(len(sample))
    PR       = np.empty(len(sample))
    MORT     = np.empty(len(sample))
    CUM_IR   = np.empty(len(sample))
    CUM_IS   = np.empty(len(sample))
    IR_FINAL = np.empty(len(sample))
    for i, row in enumerate(sample):
        (FR[i], REV[i], PR[i], MORT[i],
         CUM_IR[i], CUM_IS[i], IR_FINAL[i]) = run_model(dict(zip(names, row)), density)
    return FR, REV, PR, MORT, CUM_IR, CUM_IS, IR_FINAL


def uncertainty_report(**series):
    for label, y in series.items():
        lo, med, hi = np.percentile(y, [2.5, 50, 97.5])
        print(f"{label:>14}:  median={med:,.4g}   "
              f"95% band=[{lo:,.4g}, {hi:,.4g}]   "
              f"mean={np.mean(y):,.4g}   sd={np.std(y):,.4g}")


def uncertainty_plot(FR, REV, knee_d, path="mc_uncertainty.png"):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    data = [("FR", FR), ("Revenue", REV), ("Knee density", knee_d)]
    for a, (label, y) in zip(ax, data):
        a.hist(y, bins=35, color="#5aa0d6", edgecolor="white")
        for q, style in [(2.5, ":"), (50, "-"), (97.5, ":")]:
            a.axvline(np.percentile(y, q), color="#c0392b", ls=style, lw=1.4)
        a.set_title(f"{label}", fontsize=10)
        a.set_xlabel(label)
        a.set_ylabel("count")
    fig.suptitle(f"Output uncertainty", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130)
    print(f"saved -> {path}")



def prcc(sample, y):
    p = sample.shape[1]
    R = np.column_stack([rankdata(sample[:, j]) for j in range(p)] + [rankdata(y)])
    C = np.corrcoef(R, rowvar=False)
    P = np.linalg.inv(C + 1e-12 * np.eye(C.shape[0]))
    yi = p
    prccs = np.array([-P[j, yi] / np.sqrt(P[j, j] * P[yi, yi]) for j in range(p)])
    N  = len(y)
    df = N - 2 - (p - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = prccs * np.sqrt(df / (1 - prccs**2))
    pval = 2 * student_t.sf(np.abs(tstat), df)
    return prccs, pval


def prcc_bootstrap_ci(sample, y, n_boot=1000, seed=1):
    rng  = np.random.default_rng(seed)
    N    = len(y)
    boots = np.empty((n_boot, sample.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        boots[b], _ = prcc(sample[idx], y[idx])
    return np.percentile(boots, [2.5, 97.5], axis=0)


def sensitivity_report(names, sample, y, label):
    pr, pv = prcc(sample, y)
    lo, hi = prcc_bootstrap_ci(sample, y)
    order  = np.argsort(-np.abs(pr))
    print(f"\nSensitivity {label}")
    print(f"{'param':<10}{'PRCC':>8}{'95% CI':>18}")
    for j in order:
        print(f"{names[j]:<10}{pr[j]:>8.3f}"
              f"  [{lo[j]:.2f}, {hi[j]:.2f}]")
    return pr, order


def sensitivity_plot(names, pr_fr, pr_rev, pr_knee, path="mc_prcc.png"):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (pr, label) in zip(ax, [(pr_fr, "FR"), (pr_rev, "Revenue"), (pr_knee, "Knee density")]):
        order        = np.argsort(np.abs(pr))          # ascending → largest ends at top
        sorted_names = [names[i] for i in order]
        sorted_pr    = pr[order]
        y            = np.arange(len(names))
        colors       = ["#27ae60" if v >= 0 else "#c0392b" for v in sorted_pr]
        a.barh(y, sorted_pr, color=colors)
        a.axvline(0, color="0.3", lw=0.8)
        a.set_yticks(y)
        a.set_yticklabels(sorted_names)
        a.set_title(f"PRCC: {label}", fontsize=11)
        a.set_xlabel("PRCC")
    fig.suptitle("Parameter sensitivity — tornado plot",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130)
    print(f"\nsaved -> {path}")


def knee_uncertainty_report(knee_d):
    n = len(knee_d)
    print(f"\nKnee point uncertainty")
    for label, y, unit in [
        ("density", knee_d,   "animals/m²")
    ]:
        lo, med, hi = np.percentile(y, [2.5, 50, 97.5])
        print(f"  {label:<10}  median={med:,.4g} {unit}   "
              f"95% band=[{lo:,.4g}, {hi:,.4g}]   "
              f"sd={np.std(y):,.4g}")


if __name__ == "__main__":
    names, sample = draw_lhs(N_SIM, seed=0)
    FR, REV, PR, MORT, CUM_IR, CUM_IS, IR_FINAL = evaluate(names, sample, DENSITY_SA)
    lhs_scatter_plot(FR, REV)

    print(f"Uncertainty analysis at density {DENSITY_SA}")
    uncertainty_report(FR=FR, revenue=REV, PR=PR, final_mortality=MORT,
                        cum_IR=CUM_IR, cum_IS=CUM_IS, final_IR=IR_FINAL)

    pr_fr,  _ = sensitivity_report(names, sample, FR,  "F_R")
    pr_rev, _ = sensitivity_report(names, sample, REV, "revenue")

    # Knee-point uncertainty: sweep the full density range for each LHS draw
    print(f"\nKnee point uncertainty")
    _, knee_sample      = draw_lhs(N_KNEE, seed=42)
    density_sweep       = np.linspace(DENSITY_MIN, DENSITY_MAX, N_KNEE_DENSITIES)
    kd_list, kd_rows = [], []

    for i, row in enumerate(knee_sample):
        knee_d_i, knee_fr_i, knee_rev_i = find_knee(dict(zip(names, row)), density_sweep)
        if knee_d_i is not None:
            kd_list.append(knee_d_i)
            kd_rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N_KNEE} done — {len(kd_list)} valid knee points so far")

    knee_d            = np.array(kd_list)
    knee_sample_valid = np.array(kd_rows)

    knee_uncertainty_report(knee_d)
    pr_knee, _ = sensitivity_report(names, knee_sample_valid, knee_d, "knee density")

    uncertainty_plot(FR, REV, knee_d)
    sensitivity_plot(names, pr_fr, pr_rev, pr_knee)

    plt.show()
