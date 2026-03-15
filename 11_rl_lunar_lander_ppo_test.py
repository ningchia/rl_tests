import os
import gymnasium as gym
import torch
import numpy as np
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F

class PPOModel(nn.Module): # 必須與訓練時結構一致
    def __init__(self, state_dim, action_dim):
        super(PPOModel, self).__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.actor_mu = nn.Linear(256, action_dim)
        self.actor_sigma = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)
    def forward(self, state):
        x = self.fc(state)
        return torch.tanh(self.actor_mu(x)), F.softplus(self.actor_sigma(x)) + 1e-5, self.critic(x)

MODEL_SAVE_PATH = "dqn_model"
CHECKPOINT_FILE = "ppo_lunar_lander.pth"
checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)

def test():
    env = gym.make("LunarLanderContinuous-v3", render_mode="human")
    model = PPOModel(env.observation_space.shape[0], env.action_space.shape[0])
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    for i in range(5):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_t = torch.FloatTensor(state)
            with torch.no_grad():
                mu, _, _ = model(state_t)
            action = mu.numpy() # 測試時直接用均值 mu，不採樣
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Test Episode {i+1}, Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    test()