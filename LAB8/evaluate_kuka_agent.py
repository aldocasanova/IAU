from stable_baselines3 import PPO
from robotic_env_real import KukaArmEnv
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# Inicializar entorno con renderizado
env = KukaArmEnv(render=True)
model = PPO.load("ppo_kuka_arm", env=env)

total_episodes = 50
success_count = 0
results = []

os.makedirs("logs", exist_ok=True)

for ep in range(total_episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step = 0
    hit = False  # indicador de si se logró voltear el vaso

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if reward == 1.0:
            hit = True
        done = terminated or truncated
        step += 1

    success = 1 if hit else 0
    success_count += success
    print(f"[EPISODIO {ep+1:02d}] Recompensa total: {total_reward:.2f} → {'✅ ÉXITO' if success else '❌ FALLA'}")
    results.append([ep+1, float(total_reward), "Éxito" if success else "Falla"])

# Guardar CSV
with open("logs/eval_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episodio", "Recompensa", "Resultado"])
    writer.writerows(results)

# Imprimir resumen
print("\n📊 EVALUACIÓN COMPLETA")
print(f"Total de episodios: {total_episodes}")
print(f"Episodios exitosos: {success_count}")
print(f"Tasa de éxito: {100 * success_count / total_episodes:.2f}%")

# Gráfica de recompensas
rewards = [row[1] for row in results]
rolling_avg = np.convolve(rewards, np.ones(5)/5, mode='valid')

plt.figure(figsize=(8, 4))
plt.plot(rewards, label="Recompensa por episodio")
plt.plot(rolling_avg, label="Promedio móvil (5)", linestyle='--')
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.title("Evaluación del agente PPO")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/eval_rewards_plot.png")
plt.show()

# Cerrar entorno
env.close()



