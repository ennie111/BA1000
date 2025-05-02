import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm


def load_ensemble(num_models=3, path_prefix="models/reward_net_", finetuned=True):
    """Lädt die gespeicherten Reward-Netzwerke."""
    reward_nets = []
    suffix = "_finetuned.pth" if finetuned else ".pth"

    for i in range(num_models):
        net = BasicRewardNet(
            observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,)),  # LunarLander-v2
            action_space=gym.spaces.Discrete(4),
            normalize_input_layer=RunningNorm,
        )
        net.load_state_dict(torch.load(f"{path_prefix}{i}{suffix}"))
        net.eval()
        reward_nets.append(net)

    print(f"{num_models} Reward-Netzwerke geladen.")
    return reward_nets


def make_ensemble_reward_fn(reward_nets):
    """Kombiniert ein Ensemble von RewardNet-Modellen zu einer mittleren Reward-Funktion."""
    def ensemble_reward(obs, act):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            # Diskrete Aktion anpassen
            if isinstance(act, (int, np.integer)):
                act_tensor = torch.tensor([[act]], dtype=torch.float32)
            else:
                act_tensor = torch.tensor(act, dtype=torch.float32).unsqueeze(0)

            rewards = [
            torch.tensor(
                net.predict(obs_tensor, act_tensor, obs_tensor, torch.tensor([False]))
            )
            for net in reward_nets
        ]
        return torch.stack(rewards).mean().item()


    return ensemble_reward


def evaluate_agent(agent_path="models/ppo_lunarlander.zip", use_finetuned=True):
    env = gym.make("LunarLander-v2", render_mode="human")
    reward_nets = load_ensemble(finetuned=use_finetuned)
    reward_fn = make_ensemble_reward_fn(reward_nets)

    agent = PPO.load(agent_path)
    print("PPO-Agent geladen.")

    obs, _ = env.reset()
    done = False
    total_env_rew = 0.0
    total_model_rew = 0.0

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        next_obs, env_rew, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        model_rew = reward_fn(obs, action)

        print(f"Env-Reward: {env_rew:.2f} | Model-Reward: {model_rew:.2f}")
        total_env_rew += env_rew
        total_model_rew += model_rew

        obs = next_obs
        env.render()

    print("\nEvaluation abgeschlossen.")
    print(f" Gesamt-Env-Reward: {total_env_rew:.2f}")
    print(f"Gesamt-Model-Reward: {total_model_rew:.2f}")
    env.close()


if __name__ == "__main__":
    evaluate_agent()
