import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# 1. Basis-Parameter und Funktionen
# ---------------------------------
np.random.seed(42)

patch_yields = np.array([32.5, 45.0, 57.5])
patch_dist   = {'rich': np.array([0.5,0.3,0.2]), 'poor': np.array([0.2,0.3,0.5])}
decay, tau   = 0.075, 6.0
beta, c      = 3.0, 0.0
dt           = 0.2
n_runs       = 200
n_agents     = 50

# Compute BRR-Fixpunkt
def compute_brr(p, yields):
    br = np.dot(p, yields) / (np.sum(p)*(1/decay + tau))
    for _ in range(300):
        T = -np.log(br/yields)/decay
        R = (yields - br)/decay
        br_new = np.dot(p,R)/np.dot(p,(T+tau))
        if abs(br_new-br)<1e-6: break
        br = br_new
    return br

brr = {env: compute_brr(dist, patch_yields) for env, dist in patch_dist.items()}

# Simulate Mittel-Leave-Zeit
def mean_leave(alpha, env, S):
    br_val = brr[env]
    leaves = []
    for _ in range(n_runs):
        t=0.0
        while True:
            g = S * np.exp(-decay*t)
            vs, vl = g**alpha, br_val**alpha
            exp = beta*(vl-vs)-c
            exp = np.clip(exp,-500,500)
            p_leave = 1/(1+np.exp(-exp))
            if np.random.rand()<p_leave:
                leaves.append(t)
                break
            t += dt
    return np.mean(leaves)

# MVT-Optimum T* für jeden Yield
T_star = {env: -np.log(brr[env]/patch_yields)/decay for env in brr}

# Daten sammeln
alphas = [0.8, 1.2]
mean_data = {env: {α: [] for α in alphas} for env in patch_dist}
for env in patch_dist:
    for α in alphas:
        for S in patch_yields:
            mean_data[env][α].append(mean_leave(α, env, S))

# ---------------------------------
# 6. Plot Rich vs. Poor mit englischem Titel
# ---------------------------------
fig, axes = plt.subplots(1,2,figsize=(12,4), sharey=True)
for ax, env in zip(axes, ['rich','poor']):
    x = np.arange(len(patch_yields))
    ax.plot(x, T_star[env], '--k', lw=2, label='MVT Optimum')
    ax.plot(x, mean_data[env][0.8], '-o', color='#1f77b4', label='α=0.8')
    ax.plot(x, mean_data[env][1.2], '-o', color='#ff7f0e', label='α=1.2')
    ax.set_xticks(x)
    ax.set_xticklabels(['Low\n(32.5)','Mid\n(45.0)','High\n(57.5)'])
    ax.set_title(env.capitalize())
    ax.set_xlabel('Patch Yield')
    ax.grid(True, linestyle='--', alpha=0.4)

axes[0].set_ylabel('Leave Time (s)')
axes[1].legend(frameon=False, loc='upper right')
plt.suptitle('Comparison: MVT Optimum vs. Risk-averse & Risk-seeking')
plt.tight_layout(rect=(0,0,1,0.95))
plt.show()
