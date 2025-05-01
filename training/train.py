# Beispielgerüst:
from imitation.algorithms.preference_comparisons import AgentTrainer

# 1. AgentTrainer initialisieren mit RewardVecEnvWrapperWithFeedback
# 2. run .train() für einige Schritte
# 3. aus trainer.venv.step_feedback Feedback sammeln
# 4. in FeedbackDataset pushen
# 5. trainiere RewardNet mit FeedbackLoss
# 6. wiederhole
