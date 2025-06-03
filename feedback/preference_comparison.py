import math
import numpy as np
from typing import Optional, Callable, Mapping, Any

from imitation.algorithms.preference_comparisons import PreferenceComparisons
from imitation.util import util

from feedback.train_utils import train_on_step_feedback


class FeedbackPreferenceComparisons(PreferenceComparisons):
    def __init__(self, *args, feedback_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.feedback_dataset = feedback_dataset

    def train(
        self,
        total_timesteps: int,
        total_comparisons: int,
        callback: Optional[Callable[[int], None]] = None,
    ) -> Mapping[str, Any]:
        initial_comparisons = int(total_comparisons * self.initial_comparison_frac)
        total_comparisons -= initial_comparisons

        # Zeitplan für Abfragen berechnen
        vec_schedule = np.vectorize(self.query_schedule)
        unnormalized_probs = vec_schedule(np.linspace(0, 1, self.num_iterations))
        probs = unnormalized_probs / np.sum(unnormalized_probs)
        shares = util.oric(probs * total_comparisons)
        schedule = [initial_comparisons] + shares.tolist()

        print(f"Query schedule: {schedule}")

        timesteps_per_iteration, extra_timesteps = divmod(
            total_timesteps, self.num_iterations
        )
        reward_loss = None
        reward_accuracy = None

        for i, num_pairs in enumerate(schedule):
            ##########################
            # Daten sammeln
            ##########################
            num_steps = math.ceil(
                self.transition_oversampling * 2 * num_pairs * self.fragment_length
            )
            self.logger.log(f"Collecting {2 * num_pairs} fragments")
            trajectories = self.trajectory_generator.sample(num_steps)
            self._check_fixed_horizon(len(t) for t in trajectories if t.terminal)

            self.logger.log("Creating fragment pairs")
            fragments = self.fragmenter(trajectories, self.fragment_length, num_pairs)

            self.logger.log("Gathering preferences")
            preferences = self.preference_gatherer(fragments)
            self.dataset.push(fragments, preferences)
            self.logger.log(f"Dataset now contains {len(self.dataset)} comparisons")

            ##########################
            # Training: PbRL + SSFB
            ##########################
            epochs = getattr(self.reward_trainer, "epochs", 3)
            if i == 0:
                epochs = int(self.initial_epoch_multiplier)

            for joint_epoch in range(epochs):
                print(f"\n>>> Gemeinsame Reward-Training Epoche {joint_epoch + 1} (Iteration {i})")

                # PbRL-Training: keine accumulate_means-Wrapper außenrum!
                self.logger.log("Training reward model (PbRL)")
                self.reward_trainer.train(self.dataset, epoch_multiplier=1.0)

                # SSFB-Training
                if self.feedback_dataset is not None:
                    print("→ SSFB-Training (1 Epoche)")
                    train_on_step_feedback(self.model, self.feedback_dataset, epochs=1)

            # Logging-Ergebnisse extrahieren
            base_key = self.logger.get_accumulate_prefixes() + "reward/final/train"
            reward_loss = self.logger.name_to_value.get(f"{base_key}/loss", None)
            reward_accuracy = self.logger.name_to_value.get(f"{base_key}/accuracy", None)

            ##########################
            # Agent trainieren
            ##########################
            steps = timesteps_per_iteration
            if i == self.num_iterations - 1:
                steps += extra_timesteps

            self.logger.log(f"Training agent for {steps} timesteps")
            self.trajectory_generator.train(steps=steps)

            self.logger.dump(self._iteration)

            if callback:
                callback(self._iteration)
            self._iteration += 1

        return {
            "reward_loss": reward_loss,
            "reward_accuracy": reward_accuracy,
        }
