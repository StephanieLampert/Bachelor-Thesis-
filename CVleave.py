import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# 1. Parameters (further reduced)
# ---------------------------------
np.random.seed(42)

patch_yields = np.array([32.5, 45.0, 57.5])
patch_labels = ['Low\n(32.5)', 'Mid\n(45.0)', 'High\n(57.5)']
patch_dist = {'rich': np.array([0.5, 0.3, 0.2]),
              'poor': np.array([0.2, 0.3, 0.5])}

decay_rate = 0.075
travel_time = 6.0
dt = 0.5  # larger step for speed

# Simulation settings (further reduced)
n_runs = 100  # leave-times per agent
n_agents = 50  # agents per condition

beta = 3.0
bias = 0.0


# ---------------------------------
# 2. Compute BRR
# ---------------------------------
def compute_brr(probs, yields):
    br = np.dot(probs, yields) / (np.sum(probs) * (1 / decay_rate + travel_time))
    for _ in range(200):
        T = -np.log(br / yields) / decay_rate
        R = (yields - br) / decay_rate
        br_new = np.dot(probs, R) / np.dot(probs, (T + travel_time))
        if abs(br_new - br) < 1e-5:
            break
        br = br_new
    return br


brr = {env: compute_brr(dist, patch_yields) for env, dist in patch_dist.items()}


# ---------------------------------
# 3. Simulation functions
# ---------------------------------
def simulate_leave_times(alpha, env, S):
    br = brr[env]
    times = []
    for _ in range(n_runs):
        t = 0.0
        while True:
            g = S * np.exp(-decay_rate * t)
            vs, vl = g ** alpha, br ** alpha
            exponent = beta * (vl - vs) - bias
            exponent = np.clip(exponent, -500, 500)
            p_leave = 1 / (1 + np.exp(-exponent))
            if np.random.rand() < p_leave:
                times.append(t)
                break
            t += dt
    return np.array(times)


def aggregated_cv_patch(alpha, env, S):
    cvs = []
    for _ in range(n_agents):
        times = simulate_leave_times(alpha, env, S)
        cvs.append(np.std(times) / np.mean(times))
    return np.mean(cvs)


# ---------------------------------
# 4. Stable CVleave Plots
# ---------------------------------
for alpha in [0.8, 1.2]:
    cvs_rich = [aggregated_cv_patch(alpha, 'rich', S) for S in patch_yields]
    cvs_poor = [aggregated_cv_patch(alpha, 'poor', S) for S in patch_yields]

    x_pos = np.arange(len(patch_yields)) + 1
    plt.figure(figsize=(6, 4))
    plt.plot(x_pos, cvs_rich, '-o', label='Rich', color='#1f77b4', markersize=8)
    plt.plot(x_pos, cvs_poor, '-o', label='Poor', color='#ff7f0e', markersize=8)

    plt.xticks(x_pos, patch_labels)
    plt.xlim(0.5, len(patch_yields) + 0.5)
    plt.ylim(0, max(max(cvs_rich), max(cvs_poor)) * 1.1)

    plt.xlabel('Patch Yield')
    plt.ylabel('CVleave')
    plt.title(f'CVleave over patch yield (α={alpha})')
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

