import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("logs/rewards.npy")

plt.plot(rewards)
plt.xlabel("Episodio")
plt.ylabel("Recompensa total")
plt.title("Curva de recompensa por episodio")
plt.grid(True)
plt.savefig("logs/reward_curve.png")
plt.show()
