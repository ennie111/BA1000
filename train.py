import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.serialization import add_safe_globals

import gymnasium as gym
from stable_baselines3 import PPO

from imitation.util.util import make_vec_env
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
from imitation.util.networks import RunningNorm
from imitation.algorithms import preference_comparisons

from reward_vec_env_wrapper_feedback import RewardVecEnvWrapperWithFeedback
from feedback.dataset import SingleStepFeedbackDataset

# Für Torch Unpickling (Feedback-Dataset)
add_safe_globals([SingleStepFeedbackDataset])
def load_models(num_models=3):
    """Lädt Reward-Netze, PPO-Agent und gespeichertes Feedback-Dataset."""
    venv = make_vec_env("LunarLander-v2", rng=np.random.default_rng(0))
    observation_space = venv.observation_space
    action_space = venv.action_space

    reward_nets = []
    for i in range(num_models):
        net = BasicRewardNet(
            observation_space=observation_space,
            action_space=action_space,
            normalize_input_layer=RunningNorm,
        )
        net.load_state_dict(torch.load(f"models/reward_net_{i}.pth"))
        net.eval()
        reward_nets.append(net)

    agent = PPO.load("models/ppo_lunarlander")
    feedback_dataset = torch.load("models/step_feedback_dataset.pt")
    return reward_nets, agent, feedback_dataset



class EnsembleRewardFn:
    def __init__(self, reward_nets):
        self.reward_nets = reward_nets

    def __call__(self, obs, act, next_obs, dones):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32)
            act = torch.tensor(act, dtype=torch.float32)
            next_obs = torch.tensor(next_obs, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            rewards = [net.predict(obs, act, next_obs, dones) for net in self.reward_nets]
            rewards = [torch.tensor(r) if not isinstance(r, torch.Tensor) else r for r in rewards]
            return torch.stack(rewards).mean(dim=0).cpu().numpy()

    def predict_processed_all(self, obs, act, next_obs, dones):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32)
            act = torch.tensor(act, dtype=torch.float32)
            next_obs = torch.tensor(next_obs, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            all_rews = [net.predict(obs, act, next_obs, dones) for net in self.reward_nets]
            all_rews = [torch.tensor(r) if not isinstance(r, torch.Tensor) else r for r in all_rews]
            return torch.stack(all_rews, dim=1).cpu().numpy()


def train_on_step_feedback(reward_net, dataset, epochs=3, batch_size=64):
    """Feinjustierung des RewardNet durch direktes Feedback."""
    if len(dataset) == 0:
        print("Kein direktes Feedback – SSFB-Training übersprungen.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(reward_net.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0.0
        for obs, act, next_obs, done, feedback in loader:
            obs = torch.tensor(obs, dtype=torch.float32)
            # Falls act eindimensional ist (z. B. [64]) → umformen zu [64, 1]
            if act.ndim == 1:
                act = act.unsqueeze(1)
            act = act.to(dtype=torch.float32)

            next_obs = torch.tensor(next_obs, dtype=torch.float32)
            done = torch.tensor(done, dtype=torch.float32)
            feedback = torch.tensor(feedback, dtype=torch.float32)
            obs_p, act_p, next_obs_p, done_p = reward_net.preprocess(obs, act, next_obs, done)
            pred = reward_net(obs_p, act_p, next_obs_p, done_p).squeeze()
            loss = F.mse_loss(pred, feedback)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"🎓 SSFB Epoch {epoch+1}, Loss: {total_loss:.4f}")


def train():
    rng = np.random.default_rng(42)
    num_models = 3
    reward_nets, agent, feedback_dataset = load_models(num_models)
    ensemble_reward_fn = EnsembleRewardFn(reward_nets)

    # Neue Umgebung für das Training
    venv = make_vec_env("LunarLander-v2", rng=rng)
    wrapped_env = RewardVecEnvWrapperWithFeedback(
        venv,
        reward_fn=ensemble_reward_fn,
        step_feedback_dataset=feedback_dataset,
        uncertainty_threshold=0.5,
    )

    # PbRL-Komponenten
    fragmenter = preference_comparisons.RandomFragmenter(rng=rng)
    gatherer = preference_comparisons.SyntheticGatherer(rng=rng)

    for i in range(num_models):
        print(f"\n PbRL-Feintraining für Modell {i}")
        trainer = preference_comparisons.BasicRewardTrainer(
            preference_model=preference_comparisons.PreferenceModel(reward_nets[i]),
            loss=preference_comparisons.CrossEntropyRewardLoss(),
            epochs=3,
            rng=np.random.default_rng(i),
        )

        trajectory_gen = preference_comparisons.AgentTrainer(
            algorithm=agent,
            reward_fn=ensemble_reward_fn,
            venv=wrapped_env,
            exploration_frac=0.05,
            rng=rng,
        )

        pref_comp = preference_comparisons.PreferenceComparisons(
            trajectory_generator=trajectory_gen,
            reward_model=reward_nets[i],
            num_iterations=10,
            fragmenter=fragmenter,
            preference_gatherer=gatherer,
            reward_trainer=trainer,
            fragment_length=50,
            transition_oversampling=1,
            allow_variable_horizon=True,
            initial_epoch_multiplier=1,
            initial_comparison_frac=0.1,
        )

        pref_comp.train(total_timesteps=100_000, total_comparisons=100)
        train_on_step_feedback(reward_nets[i], feedback_dataset)
        torch.save(reward_nets[i].state_dict(), f"models/reward_net_{i}_finetuned.pth")
        print(f"Reward-Netzwerk {i} (finetuned) gespeichert!")

    print("Training abgeschlossen!")


if __name__ == "__main__":
    train()
