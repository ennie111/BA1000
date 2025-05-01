from typing import List, Tuple
import numpy as np
import torch.utils.data as data_th
from imitation.data import types


class SingleStepFeedbackDataset(data_th.Dataset):
    """
    Datensatz für einzelne Schritte mit binärem Feedback (-1 oder +1),
    """

    def __init__(self) -> None:
        self.obs: List[np.ndarray] = []
        self.acts: List[np.ndarray] = []
        self.next_obs: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.feedbacks: List[int] = []  # -1 oder +1

    def push(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        next_obs: np.ndarray,
        done: bool,
        feedback: int,
    ) -> None:
        assert feedback in (-1, 1), "Feedback muss -1 oder +1 sein."
        self.obs.append(obs)
        self.acts.append(act)
        self.next_obs.append(next_obs)
        self.dones.append(done)
        self.feedbacks.append(feedback)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, int]:
        return (
            self.obs[idx],
            self.acts[idx],
            self.next_obs[idx],
            self.dones[idx],
            self.feedbacks[idx],
        )

    def __len__(self) -> int:
        return len(self.feedbacks)

    def to_transitions(self) -> types.Transitions:
        """Konvertiert in ein Transitions-Objekt (z.B. für das RewardNet)."""
        return types.Transitions(
            obs=np.stack(self.obs),
            acts=np.stack(self.acts),
            next_obs=np.stack(self.next_obs),
            dones=np.array(self.dones),
        )

    def to_tensor_dataset(self):
        """Optional: als TensorDataset für direktes PyTorch-Training."""
        return data_th.TensorDataset(
            np.stack(self.obs),
            np.stack(self.acts),
            np.stack(self.next_obs),
            np.array(self.dones),
            np.array(self.feedbacks),
        )
