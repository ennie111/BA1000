import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from imitation.algorithms import preference_comparisons
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from imitation.util.logger import configure as imitation_logger_configure
from imitation.util import logger as imit_logger
import os
import pandas as pd


class SimpleLogger:
    def __init__(self, keys):
        self.keys = keys
        self.data = {key: [] for key in keys}
        self.steps = []
        self.current_step = 0

    def record(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)
        if len(self.steps) <= len(self.data[key]):
            self.steps.append(self.current_step)
            self.current_step += 1

    def plot(self):
        import matplotlib.pyplot as plt
        for key in self.keys:
            if key in self.data and len(self.data[key]) > 0:
                plt.figure()
                plt.plot(self.steps[:len(self.data[key])], self.data[key], label=key)
                plt.title(key)
                plt.xlabel("Iteration")
                plt.ylabel(key.split("/")[-1])
                plt.legend()
                plt.grid(True)
                plt.show()

class RewardTracker:
    def __init__(self):
        self.accuracies = []
        self.losses = []
        self.current_metrics = {}  # Wieder hinzugefügt für das Monitoring
        print("RewardTracker initialized")  # Debug output

    def update_metrics(self, stats):
        """Update metrics from the training statistics"""
        print("\nProcessing stats:", stats)  # Debug output
        
        # Sammle alle Epochen-Metriken dieser Iteration
        epoch_accuracies = []
        epoch_losses = []
        
        # Finde die höchste Epochennummer
        max_epoch = -1
        for key in stats.keys():
            if 'reward/epoch-' in key and 'train/accuracy' in key:
                epoch_num = int(key.split('epoch-')[1].split('/')[0])
                max_epoch = max(max_epoch, epoch_num)
        
        # Sammle die Metriken für alle Epochen
        if max_epoch >= 0:
            for epoch in range(max_epoch + 1):
                acc_key = f'reward/epoch-{epoch}/train/accuracy'
                loss_key = f'reward/epoch-{epoch}/train/loss'
                
                if acc_key in stats:
                    print(f"Found accuracy for epoch {epoch}: {stats[acc_key]}")
                    epoch_accuracies.append(float(stats[acc_key]))
                if loss_key in stats:
                    print(f"Found loss for epoch {epoch}: {stats[loss_key]}")
                    epoch_losses.append(float(stats[loss_key]))
        
        # Wenn wir finale Metriken haben, speichern wir sie
        if 'reward/final/train/accuracy' in stats:
            final_accuracy = float(stats['reward/final/train/accuracy'])
            print(f"Saving final accuracy: {final_accuracy}")
            self.accuracies.append(final_accuracy)
            
        if 'reward/final/train/loss' in stats:
            final_loss = float(stats['reward/final/train/loss'])
            print(f"Saving final loss: {final_loss}")
            self.losses.append(final_loss)
            
        # Aktualisiere current_metrics mit allen Stats
        self.current_metrics.update(stats)

# Zufallszahlengenerator für Reproduzierbarkeit
rng = np.random.default_rng(0)

# Umgebung erstellen
venv = make_vec_env("LunarLander-v2", rng=rng)

# Ensemble von Reward-Netzwerken
NUM_MODELS = 3
reward_nets = []
reward_trainers = []
trackers = []

# Create a directory for logs
log_dir = "training_logs"
os.makedirs(log_dir, exist_ok=True)

# Initialize the main logger
main_logger = imit_logger.configure(folder=log_dir, format_strs=["stdout", "log", "csv"])

for i in range(NUM_MODELS):
    net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
    )
    model = preference_comparisons.PreferenceModel(net)
    
    trainer = preference_comparisons.BasicRewardTrainer(
        preference_model=model,
        loss=preference_comparisons.CrossEntropyRewardLoss(),
        epochs=5,
        rng=np.random.default_rng(i),
    )

    reward_nets.append(net)
    reward_trainers.append(trainer)
    trackers.append(RewardTracker())

# PPO-Agent wie im Originalcode
agent = PPO(
    policy=FeedForward32Policy,
    policy_kwargs=dict(
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
    ),
    env=venv,
    seed=0,
    n_steps=4096 // venv.num_envs,
    batch_size=128,
    ent_coef=0.005,
    learning_rate=3e-4,
    clip_range=0.2,
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=5,
)

# Ensemble-Reward-Funktion (Mittelwert der Vorhersagen)
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

ensemble_reward_fn = EnsembleRewardFn(reward_nets)

# Fragmenter & Gatherer
fragmenter = preference_comparisons.RandomFragmenter(warning_threshold=0, rng=rng)
gatherer = preference_comparisons.SyntheticGatherer(rng=rng)

# Trajektorien-Generator mit Ensemble-Reward
trajectory_generator = preference_comparisons.AgentTrainer(
    algorithm=agent,
    reward_fn=ensemble_reward_fn,
    venv=venv,
    exploration_frac=0.05,
    rng=rng,
)

# PbRL-Training für jedes Reward-Netzwerk
for i in range(NUM_MODELS):
    print(f"\n>>> Trainiere Reward-Netzwerk {i}")
    
    pref_comp = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_nets[i],
        num_iterations=60,
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainers[i],
        fragment_length=75,
        transition_oversampling=1,
        initial_comparison_frac=0.067,
        allow_variable_horizon=True,
        initial_epoch_multiplier=5,
        query_schedule="hyperbolic",
    )
    
    # Monkey patch the trainer's logger to capture metrics
    original_record = reward_trainers[i].logger.record
    def record_wrapper(key, value, *args, **kwargs):
        if isinstance(value, (int, float, np.float32, np.float64)):
            trackers[i].current_metrics[key] = float(value)
            #print(f"Recording metric for network {i}: {key} = {value}")  # Debug output
        return original_record(key, value, *args, **kwargs)
    reward_trainers[i].logger.record = record_wrapper

    original_dump = reward_trainers[i].logger.dump
    def dump_wrapper(*args, **kwargs):
        # Process the collected metrics before dumping
        print(f"\nDumping metrics for network {i}:")  # Debug output
        print("Current metrics to process:", trackers[i].current_metrics)  # Debug output
        trackers[i].update_metrics(trackers[i].current_metrics)
        print(f"Current state after update - Accuracies: {trackers[i].accuracies}, Losses: {trackers[i].losses}")  # Debug output
        trackers[i].current_metrics = {}  # Clear for next iteration
        return original_dump(*args, **kwargs)
    reward_trainers[i].logger.dump = dump_wrapper
    
    # Train the model
    pref_comp.train(total_timesteps=300_000, total_comparisons=30_000)
    print(f"\n>>> Training für Reward-Netzwerk {i} abgeschlossen")
    print(f"Gesammelte Metriken für Netzwerk {i}:")
    print(f"Accuracies: {trackers[i].accuracies}")
    print(f"Losses: {trackers[i].losses}")

    torch.save(reward_nets[i].state_dict(), f"reward_net_{i}.pth")
    print(f">>> Reward-Netzwerk {i} gespeichert!")

# Verwende den gelernten Ensemble-Reward in der finalen Umgebung
learned_reward_venv = RewardVecEnvWrapper(
    venv,
    ensemble_reward_fn,
)

# Finaltraining mit PPO auf dem Ensemble-Reward
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
    ent_coef=0.00053,
    learning_rate=0.00018,
    clip_range=0.2,
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=5,
)

print(">>> Starte Finaltraining...")
learner.learn(100_000)
print(">>> Finaltraining abgeschlossen.")
learner.save("trained_ensemble_learner")

# Evaluation
n_eval_episodes = 10
reward_mean, reward_std = evaluate_policy(learner.policy, venv, n_eval_episodes)
reward_stderr = reward_std / np.sqrt(n_eval_episodes)
print(f"Reward: {reward_mean:.0f} +/- {reward_stderr:.0f}")

def plot_metrics():
    print("\nPreparing to plot metrics:")  # Debug output
    
    # Plot accuracies
    plt.figure(figsize=(12, 6))
    has_data = False
    for i, tracker in enumerate(trackers):
        print(f"\nNetwork {i} metrics summary:")
        print(f"Accuracies ({len(tracker.accuracies)} values): {tracker.accuracies}")
        print(f"Losses ({len(tracker.losses)} values): {tracker.losses}")
        
        if tracker.accuracies:
            has_data = True
            plt.plot(range(len(tracker.accuracies)), 
                    tracker.accuracies, 
                    marker='o', 
                    label=f'Netzwerk {i}')
    
    if has_data:
        plt.title('Accuracy während des Trainings')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('accuracy_plot.png')
        plt.show()
    else:
        print("\nKeine Accuracy-Daten zum Plotten verfügbar")
        for i, tracker in enumerate(trackers):
            print(f"Netzwerk {i} - Anzahl Accuracy Werte: {len(tracker.accuracies)}")

    # Plot losses
    plt.figure(figsize=(12, 6))
    has_data = False
    for i, tracker in enumerate(trackers):
        if tracker.losses:
            has_data = True
            plt.plot(range(len(tracker.losses)), 
                    tracker.losses, 
                    marker='o', 
                    label=f'Netzwerk {i}')
    
    if has_data:
        plt.title('Loss während des Trainings')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('loss_plot.png')
        plt.show()
    else:
        print("\nKeine Loss-Daten zum Plotten verfügbar")
        for i, tracker in enumerate(trackers):
            print(f"Netzwerk {i} - Anzahl Loss Werte: {len(tracker.losses)}")

# Plot the metrics
print("\nPlotte die Trainingsmetriken...")
plot_metrics()

