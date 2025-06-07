import time
import gymnasium as gym
from stable_baselines3 import PPO

# Umgebung mit visueller Darstellung (nicht vektorisiert!)
env = gym.make("LunarLander-v2", render_mode="human")

# Lade den trainierten Agenten
learner = PPO.load("trained_ensemble_learner")

# Hol dir die Policy
policy = learner.policy

# Initialisiere Episode
obs, _ = env.reset()
done = False
total_reward = 0
steps = 0

# Episode ausführen
while not done:
    action, _ = policy.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    total_reward += reward
    steps += 1

    time.sleep(0.02)  # für flüssige Animation

env.close()

# Ergebnisse anzeigen
print(f"Episode beendet nach {steps} Schritten. Gesamt-Reward: {total_reward:.2f}")
