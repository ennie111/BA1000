import collections
from typing import Deque, Optional
import numpy as np

from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from imitation.rewards import reward_function
from imitation.data import types
from imitation.rewards.reward_nets import RewardEnsemble

from feedback.dataset import SingleStepFeedbackDataset


class RewardVecEnvWrapperWithFeedback(VecEnvWrapper):
    """
    VecEnv-Wrapper mit Ensemble-Reward-Funktion,
    der bei hoher Unsicherheit synthetisches Feedback speichert.
    """

    def __init__(
        self,
        venv: VecEnv,
        reward_fn: reward_function.RewardFn,
        step_feedback_dataset: SingleStepFeedbackDataset,
        uncertainty_threshold: float = 0.25,
        ep_history: int = 100,
    ):
        super().__init__(venv)
        self.reward_fn = reward_fn
        self.step_feedback_dataset = step_feedback_dataset
        self.uncertainty_threshold = uncertainty_threshold

        self.episode_rewards: Deque[float] = collections.deque(maxlen=ep_history)
        self._cumulative_rew = np.zeros((venv.num_envs,))
        self._old_obs = None
        self._actions = None
        self.reset()

    def reset(self, **kwargs):
        self._old_obs = self.venv.reset(**kwargs)
        return self._old_obs

    def step_async(self, actions):
        self._actions = actions
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, env_rews, dones, infos = self.venv.step_wait()

        obs_fixed = []
        obs = types.maybe_wrap_in_dictobs(obs)
        for single_obs, single_done, single_infos in zip(obs, dones, infos):
            if single_done:
                single_obs = single_infos["terminal_observation"]
            obs_fixed.append(types.maybe_wrap_in_dictobs(single_obs))

        obs_fixed = (
            types.DictObs.stack(obs_fixed)
            if isinstance(obs, types.DictObs)
            else np.stack(obs_fixed)
        )

        rews = self.reward_fn(
            self._old_obs,
            self._actions,
            types.maybe_unwrap_dictobs(obs_fixed),
            np.array(dones),
        )

        # === UNCERTAINTY-BASED FEEDBACK HOOK ===
        if hasattr(self.reward_fn, "predict_processed_all"):
            obs_old = types.assert_not_dictobs(self._old_obs)
            acts = self._actions
            next_obs = types.assert_not_dictobs(obs_fixed)
            dones_arr = np.array(dones)

            all_model_rews = self.reward_fn.predict_processed_all(obs_old, acts, next_obs, dones_arr)
            for i in range(len(obs)):
                model_rews = all_model_rews[i]
                variance = np.var(model_rews)
                if variance > self.uncertainty_threshold:
                    step_rew = rews[i]
                    feedback = +1 if env_rews[i] >1  else -1

                    self.step_feedback_dataset.push(
                        obs=obs_old[i],
                        act=acts[i],
                        next_obs=next_obs[i],
                        done=dones_arr[i],
                        feedback=feedback,
                    )
                    print(f"[Feedback] env={i} | var={variance:.3f} | env_rew={env_rews[i]:.3f} |step_rew={step_rew:.3f}| feedback={feedback}")


        # Statistiken aktualisieren
        done_mask = np.asarray(dones, dtype="bool").reshape((len(dones),))
        self._cumulative_rew += rews
        for single_done, single_ep_rew in zip(dones, self._cumulative_rew):
            if single_done:
                self.episode_rewards.append(single_ep_rew)
        self._cumulative_rew[done_mask] = 0

        self._old_obs = types.maybe_unwrap_dictobs(obs)
        for info_dict, old_rew in zip(infos, env_rews):
            info_dict["original_env_rew"] = old_rew

        return types.maybe_unwrap_dictobs(obs), rews, dones, infos
