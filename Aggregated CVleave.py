import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Parameters for stability
# -------------------------------
np.random.seed(0)

patch_yields = np.array([32.5, 45.0, 57.5])
patch_dist = {
    'rich': np.array([0.5, 0.3, 0.2]),
    'poor': np.array([0.2, 0.3, 0.5])
}

decay = 0.075
tau = 6.0
dt = 0.5      # coarse step for speed
n_runs = 200  # samples per agent
n_agents = 50 # number of agents

beta = 3.0
c = 0.0

# ---------------------------------
# 2. Compute BRR
# ---------------------------------
def compute_brr(p, yields):
    br = np.dot(p, yields)/(np.sum(p)*(1/decay + tau))
    for _ in range(200):
        T = -np.log(br/yields)/decay
        R = (yields-br)/decay
        br_new = np.dot(p, R)/np.dot(p, (T+tau))
        if abs(br_new-br) < 1e-5: break
        br = br_new
    return br

brr = {env: compute_brr(dist, patch_yields) for env,dist in patch_dist.items()}

# ---------------------------------
# 3. Simulate leave times (random yields)
# ---------------------------------
def simulate_times(alpha, env):
    br_val = brr[env]
    times = np.zeros(n_runs)
    # initialize unfinished
    unfinished = np.arange(n_runs)
    while unfinished.size > 0:
        t = times[unfinished]
        S = np.random.choice(patch_yields, size=unfinished.size, p=patch_dist[env])
        g = S * np.exp(-decay*t)
        vs = g**alpha
        vl = br_val**alpha
        exp_term = beta*(vl-vs) - c
        exp_term = np.clip(exp_term, -500, 500)
        p_leave = 1/(1+np.exp(-exp_term))
        # random draw
        r = np.random.rand(unfinished.size)
        leave_mask = r < p_leave
        # record leave times
        times[unfinished[leave_mask]] = t[leave_mask]
        # update unfinished
        unfinished = unfinished[~leave_mask]
        times[unfinished] += dt
    return times

# ---------------------------------
# 4. Aggregated CVleave
# ---------------------------------
def aggregated_cv(alpha, env):
    cvs = np.zeros(n_agents)
    for i in range(n_agents):
        t = simulate_times(alpha, env)
        cvs[i] = np.std(t)/np.mean(t)
    return np.mean(cvs)

# ---------------------------------
# 5. Compute for α=0.8 & 1.2
# ---------------------------------
alphas = [0.8, 1.2]
results = {alpha: {env: aggregated_cv(alpha, env) for env in patch_dist} for alpha in alphas}

# Print
for alpha in alphas:
    print(f"α={alpha}: Rich CV={results[alpha]['rich']:.3f}, Poor CV={results[alpha]['poor']:.3f}")

# ---------------------------------
# 6. Bar Plot
# ---------------------------------
labels = [f"α={a}" for a in alphas]
rich_vals = [results[a]['rich'] for a in alphas]
poor_vals = [results[a]['poor'] for a in alphas]
x = np.arange(len(labels))
w = 0.35

fig, ax = plt.subplots(figsize=(6,4))
ax.bar(x-w/2, rich_vals, w, label='Rich', color='#2E86AB')
ax.bar(x+w/2, poor_vals, w, label='Poor', color='#F18F01')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Aggregated CVleave')
ax.set_title('Aggregated CVleave')
ax.legend(frameon=False)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
