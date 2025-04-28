import numpy as np
import matplotlib.pyplot as plt

# 1. Decay function
def g(x, s):
    return s * np.exp(-0.075 * x)

# 2. Patch yields
s_vals = np.array([32.5, 45.0, 57.5])
labels = ['Low (32.5)', 'Mid (45.0)', 'High (57.5)']

# 3. Precomputed environment rates
rich_rate = 21.9
poor_rate = 18.6
env_rates = {'rich': rich_rate, 'poor': poor_rate}
colors = {'rich': 'gold', 'poor': 'green'}

# 4. Compute intersection points
intersection_points = []
for env, rate in env_rates.items():
    for s_val in s_vals:
        if rate <= s_val:
            x_cross = -np.log(rate / s_val) / 0.075
            y_cross = g(x_cross, s_val)
            intersection_points.append((x_cross, y_cross, colors[env]))

# 5. Plotting with origin-centered and equal aspect axes
fig, ax = plt.subplots(figsize=(8, 5))

# a) Decay curves
x = np.linspace(0, 100, 400)
for s_val, label in zip(s_vals, labels):
    ax.plot(x, g(x, s_val), label=label, linewidth=2)

# b) Environment lines
for env, rate in env_rates.items():
    ax.hlines(rate, xmin=0, xmax=100, colors=colors[env], linestyles='--', label=env.capitalize())

# c) Intersection markers
for x_cross, y_cross, color in intersection_points:
    ax.scatter(x_cross, y_cross, color=color, marker='o', s=50, zorder=5)

# d) Move spines to (0,0)
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# e) Axis limits and aspect
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.set_aspect('equal', adjustable='box')

# f) Labels, title, legend, grid
ax.set_xlabel('Time in patch')
ax.set_ylabel('Patch reward rate')
ax.set_title('Optimal Foraging Behaviour')
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', linewidth=0.5)

plt.show()






