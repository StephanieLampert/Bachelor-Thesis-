import numpy as np
import matplotlib.pyplot as plt

# 1) Four parameter combinations
combos = {
    'Low c, Low β': {'c': -2.0, 'β': 1.0},
    'Low c, High β': {'c': -2.0, 'β': 5.0},
    'High c, Low β': {'c': +2.0, 'β': 1.0},
    'High c, High β': {'c': +2.0, 'β': 5.0},
}

# 2) ΔV axis: V_leave − V_stay
deltaV = np.linspace(-3, 3, 400)

# 3) Plot each combination with orange curves
for label, params in combos.items():
    c_val = params['c']
    beta_val = params['β']
    P_leave = 1 / (1 + np.exp(c_val - beta_val * deltaV))

    plt.figure(figsize=(6, 4))
    plt.plot(deltaV, P_leave, color='orange', lw=2)  # Orange curve
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='P = 0.5 (indifference)')

    plt.xlabel(r'$\Delta V = V_{\rm leave} - V_{\rm stay}$')
    plt.ylabel(r'$P_{\rm leave}$')
    plt.title(f'Softmax Curve for\n{label}', pad=15)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



