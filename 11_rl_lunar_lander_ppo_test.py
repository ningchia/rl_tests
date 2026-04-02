import os
import gymnasium as gym
import torch
import numpy as np
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F

# --- PPO 網路架構 (Actor-Critic 分離 backbone) ---
class PPOModel(nn.Module): # 必須與訓練時結構一致
    def __init__(self, state_dim, action_dim):
        super(PPOModel, self).__init__()
        # --- Actor 網路：專門負責輸出動作分布 ---
        self.actor_backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(256, action_dim)
        self.sigma_head = nn.Linear(256, action_dim)

        # --- Critic 網路：專門負責估計狀態價值 (Value) ---
        self.critic_backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # 直接輸出 Value
        )

    def forward(self, state):
        # 1. Actor 路徑
        a_x = self.actor_backbone(state)
        mu = torch.tanh(self.mu_head(a_x))
        # 這裡建議繼續用你的 clamp 邏輯或 softplus
        log_sigma = torch.clamp(self.sigma_head(a_x), -2.0, 0) 
        sigma = torch.exp(log_sigma)

        # 2. Critic 路徑 (完全獨立於 Actor 的特徵提取)
        value = self.critic_backbone(state)

        return mu, sigma, value
    
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