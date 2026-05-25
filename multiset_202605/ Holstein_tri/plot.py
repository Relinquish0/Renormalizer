import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt


data = np.load("HolsteinTri3.npz", allow_pickle=True)
time = np.array(data["time series"], dtype=float)
occupations = np.array(data["electron occupations array"], dtype=float)

plt.figure(figsize=(8, 5))
for i in range(occupations.shape[1]):
    plt.plot(time, occupations[:, i], label=f"Site {i + 1}", linewidth=3)

plt.xlabel("Time (a.u.)", fontsize=16)
plt.ylabel("Population", fontsize=16)
plt.xlim(left=0)
plt.ylim(-0.1, 1.05)
plt.legend(fontsize=13)
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig("HolsteinTri3_population.png", dpi=150)
