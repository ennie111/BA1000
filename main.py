# main.py
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.algorithms import preference_comparisons
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from reward_vec_env_wrapper_feedback import RewardVecEnvWrapperWithFeedback
from feedback.dataset import SingleStepFeedbackDataset

os.makedirs("models", exist_ok=True)

rng = np.random.default_rng(0)
NUM_MODELS = 3

# Environment
venv = make_vec_env("LunarLander-v2", rng=rng)
feedback_dataset = SingleStepFeedbackDataset()

wrapped_venv = RewardVecEnvWrapperWithFeedback(
    venv,
    reward_fn=None,  # wird später gesetzt
    step_feedback_dataset=feedback_dataset,
    uncertainty_threshold=0.5,
)

# Reward-Nets & Trainer
reward_nets, reward_models, reward_trainers = [], [], []
for i in range(NUM_MODELS):
    net = BasicRewardNet(
        wrapped_venv.observation_space,
        wrapped_venv.action_space,
        normalize_input_layer=RunningNorm,
    )
    model = preference_comparisons.PreferenceModel(net)
    trainer = preference_comparisons.BasicRewardTrainer(
        preference_model=model,
        loss=preference_comparisons.CrossEntropyRewardLoss(),
        epochs=3,
        rng=np.random.default_rng(i),
    )
    reward_nets.append(net)
    reward_models.append(model)
    reward_trainers.append(trainer)

# Ensemble-Reward-Funktion
class EnsembleRewardFn:
    def __init__(self, models):
        self.models = models

    def __call__(self, obs, act, next_obs, dones):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32)
            act = torch.tensor(act, dtype=torch.float32)
            next_obs = torch.tensor(next_obs, dtype=torch.float32)
            rewards = [m.reward_net(*m.reward_net.preprocess(obs, act, next_obs, dones)) for m in self.models]
            return torch.stack(rewards).mean(dim=0).cpu().numpy()

    def predict_processed_all(self, obs, act, next_obs, dones):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32)
            act = torch.tensor(act, dtype=torch.float32)
            next_obs = torch.tensor(next_obs, dtype=torch.float32)
            all_rews = [m.reward_net(*m.reward_net.preprocess(obs, act, next_obs, dones)) for m in self.models]
            return np.stack(all_rews, axis=1)

reward_fn = EnsembleRewardFn(reward_models)
wrapped_venv.reward_fn = reward_fn

# PPO-Agent
agent = PPO(
    policy=FeedForward32Policy,
    policy_kwargs=dict(
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
    ),
    env=wrapped_venv,
    seed=0,
    n_steps=2048 // wrapped_venv.num_envs,
    batch_size=64,
    ent_coef=0.01,
    learning_rate=2e-3,
    clip_range=0.1,
    gae_lambda=0.95,
    gamma=0.97,
    n_epochs=10,
)

# Speichern
agent.save("models/ppo_lunarlander")
for i, net in enumerate(reward_nets):
    torch.save(net.state_dict(), f"models/reward_net_{i}.pth")
torch.save(feedback_dataset, "models/step_feedback_dataset.pt")

print("Setup abgeschlossen.")
