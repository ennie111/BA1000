from imitation.util import util  
import math
from typing import Any, Callable, Mapping, Optional
from imitation.algorithms.preference_comparisons import PreferenceComparisons
import numpy as np

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

        vec_schedule = np.vectorize(self.query_schedule)
        unnormalized_probs = vec_schedule(np.linspace(0, 1, self.num_iterations))
        probs = unnormalized_probs / np.sum(unnormalized_probs)
        shares = util.oric(probs * total_comparisons)
        schedule = [initial_comparisons] + shares.tolist()

        timesteps_per_iteration, extra_timesteps = divmod(
            total_timesteps,
            self.num_iterations,
        )
        reward_loss = None
        reward_accuracy = None

        for i, num_pairs in enumerate(schedule):
            num_steps = math.ceil(
                self.transition_oversampling * 2 * num_pairs * self.fragment_length,
            )
            self.logger.log(f"Collecting {2 * num_pairs} fragments")
            trajectories = self.trajectory_generator.sample(num_steps)
            horizons = (len(traj) for traj in trajectories if traj.terminal)
            self._check_fixed_horizon(horizons)

            fragments = self.fragmenter(trajectories, self.fragment_length, num_pairs)
            preferences = self.preference_gatherer(fragments)
            self.dataset.push(fragments, preferences)

            # HIER Training über mehrere Epochen hinweg, abwechselnd PbRL und SSFB
            epochs = self.reward_trainer.epochs if hasattr(self.reward_trainer, "epochs") else 3
            epochs = int(self.initial_epoch_multiplier) if i == 0 else epochs

            for joint_epoch in range(epochs):
                print(f"\n>>> Gemeinsame Reward-Training Epoche {joint_epoch + 1} (Iteration {i})")

                # PbRL-Training (eine Epoche)
                self.reward_trainer.train(self.dataset, epoch_multiplier=1.0)

                # SSFB-Training (eine Epoche)
                if self.feedback_dataset is not None:
                    print("→ SSFB-Training (1 Epoche)")
                    train_on_step_feedback(self.model, self.feedback_dataset, epochs=1)


            num_steps = timesteps_per_iteration
            if i == self.num_iterations - 1:
                num_steps += extra_timesteps
            self.trajectory_generator.train(steps=num_steps)

            self.logger.dump(self._iteration)
            if callback:
                callback(self._iteration)
            self._iteration += 1

        return {
            "reward_loss": reward_loss,
            "reward_accuracy": reward_accuracy,
        }
