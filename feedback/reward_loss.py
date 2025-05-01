import torch.nn as nn
import torch

class FeedbackRewardLoss(nn.Module):
    def forward(self, observations, actions, feedback):
        preds = reward_net(observations, actions)
        loss = nn.MSELoss()(preds, feedback)
        return loss
