import os
import numpy as np
import torch
from stable_baselines3 import PPO
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.algorithms import preference_comparisons
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from reward_vec_env_wrapper_feedback import RewardVecEnvWrapperWithFeedback
from feedback.dataset import SingleStepFeedbackDataset
from feedback.preference_comparison import FeedbackPreferenceComparisons
from stable_baselines3.common.evaluation import evaluate_policy

# Konfiguration
NUM_MODELS = 3
TOTAL_TIMESTEPS = 300_000
TOTAL_COMPARISONS = 3_000
FINAL_TRAINING_STEPS = 100_000
EVAL_EPISODES = 10
UNCERTAINTY_THRESHOLD = 0.25
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Zufallsquelle
rng = np.random.default_rng(0)

def main():
    # Umgebung und Feedback vorbereiten
    venv = make_vec_env("LunarLander-v2", rng=rng)
    feedback_dataset = SingleStepFeedbackDataset()

    # Reward-Netze + Trainer
    reward_nets, reward_trainers = [], []
    for i in range(NUM_MODELS):
        net = BasicRewardNet(
            venv.observation_space, venv.action_space,
            normalize_input_layer=RunningNorm
        )
        trainer = preference_comparisons.BasicRewardTrainer(
            preference_model=preference_comparisons.PreferenceModel(net),
            loss=preference_comparisons.CrossEntropyRewardLoss(),
            epochs=5,
            rng=np.random.default_rng(i),
        )
        reward_nets.append(net)
        reward_trainers.append(trainer)

        # Ensemble Reward-Funktion
        class EnsembleRewardFn:
            def __init__(self, reward_nets):
                self.reward_nets = reward_nets

            def __call__(self, obs, act, next_obs, dones):
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    act_tensor = torch.tensor(act, dtype=torch.float32)
                    next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
                    dones_tensor = torch.tensor(dones, dtype=torch.float32)

                    rewards = [
                        net.predict(obs_tensor, act_tensor, next_obs_tensor, dones_tensor)
                        for net in self.reward_nets
                    ]
                    return np.mean(rewards, axis=0)

            def predict_processed_all(self, obs, act, next_obs, dones):
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32)
                    act_t = torch.tensor(act, dtype=torch.float32)
                    next_obs_t = torch.tensor(next_obs, dtype=torch.float32)
                    dones_t = torch.tensor(dones, dtype=torch.float32)

                    all_rews = [
                        net.predict(obs_t, act_t, next_obs_t, dones_t)
                        for net in self.reward_nets
                    ]

                    # Stelle sicher, dass alles NumPy ist
                    all_rews = [r.cpu().numpy() if isinstance(r, torch.Tensor) else r for r in all_rews]

                    # Stacke zu (batch_size, num_models)
                    return np.stack(all_rews, axis=1)



    ensemble_reward_fn = EnsembleRewardFn(reward_nets)

    # Umgebung mit Feedback-Wrapper
    wrapped_venv = RewardVecEnvWrapperWithFeedback(
        venv,
        reward_fn=ensemble_reward_fn,
        step_feedback_dataset=feedback_dataset,
        uncertainty_threshold=UNCERTAINTY_THRESHOLD,
    )

    # PPO-Agent
    agent = PPO(
        policy=FeedForward32Policy,
        policy_kwargs=dict(
            features_extractor_class=NormalizeFeaturesExtractor,
            features_extractor_kwargs=dict(normalize_class=RunningNorm),
        ),
        env=wrapped_venv,
        seed=0,
        n_steps=4096 // wrapped_venv.num_envs,
        batch_size=128,
        ent_coef=0.005,
        learning_rate=3e-4,
        clip_range=0.2,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=5,
    )

    # PbRL-Komponenten
    fragmenter = preference_comparisons.RandomFragmenter(warning_threshold=0, rng=rng)
    gatherer = preference_comparisons.SyntheticGatherer(rng=rng)

    # PbRL-Training für jedes Reward-Netz
    for i in range(NUM_MODELS):
        print(f">>> Trainiere Reward-Netz {i}")
        trajectory_generator = preference_comparisons.AgentTrainer(
            algorithm=agent,
            reward_fn=ensemble_reward_fn,
            venv=wrapped_venv,
            exploration_frac=0.05,
            rng=rng,
        )

        pref_comp = FeedbackPreferenceComparisons(
            trajectory_generator=trajectory_generator,
            reward_model=reward_nets[i],
            num_iterations=80,
            fragmenter=fragmenter,
            preference_gatherer=gatherer,
            reward_trainer=reward_trainers[i],
            fragment_length=40,
            transition_oversampling=2,
            initial_comparison_frac=0.15,
            allow_variable_horizon=True,
            initial_epoch_multiplier=5,
            feedback_dataset=feedback_dataset,
        )

        pref_comp.train(
            total_timesteps=TOTAL_TIMESTEPS,
            total_comparisons=TOTAL_COMPARISONS,
        )

        torch.save(reward_nets[i].state_dict(), f"{MODEL_DIR}/reward_net_{i}.pth")
        print(f">>> Reward-Netz {i} gespeichert.")

    # Finaltraining mit Ensemble-Reward
    learned_reward_venv = RewardVecEnvWrapper(
        venv,
        reward_fn=ensemble_reward_fn,
    )

    learner = PPO(
        policy=FeedForward32Policy,
        policy_kwargs=dict(
            features_extractor_class=NormalizeFeaturesExtractor,
            features_extractor_kwargs=dict(normalize_class=RunningNorm),
        ),
        env=learned_reward_venv,
        seed=0,
        n_steps=4096 // learned_reward_venv.num_envs,
        batch_size=128,
        ent_coef=0.005,
        learning_rate=3e-4,
        clip_range=0.2,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=5,
    )

    print(">>> Starte Finaltraining...")
    learner.learn(FINAL_TRAINING_STEPS)
    print(">>> Finaltraining abgeschlossen.")

    learner.save(f"{MODEL_DIR}/ppo_lunarlander")
    torch.save(feedback_dataset, f"{MODEL_DIR}/step_feedback_dataset.pt")

    # Evaluation mit echtem Reward
    reward_mean, reward_std = evaluate_policy(learner.policy, venv, n_eval_episodes=EVAL_EPISODES)
    reward_stderr = reward_std / np.sqrt(EVAL_EPISODES)
    print(f"Reward: {reward_mean:.0f} +/- {reward_stderr:.0f}")


if __name__ == "__main__":
    main()
