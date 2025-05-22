# feedback/train_utils.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def train_on_step_feedback(reward_net, dataset, epochs=1, batch_size=64):
    if len(dataset) == 0:
        print("Kein direktes Feedback – SSFB-Training übersprungen.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(reward_net.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0.0
        for obs, act, next_obs, done, feedback in loader:
            obs = torch.tensor(obs, dtype=torch.float32)
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

        print(f"SSFB Epoch {epoch+1}, Loss: {total_loss:.4f}")

