import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import os
from torch.distributions import Normal

DETERMINISTIC = True    # 是否直接使用mu作為動作，還是從分布中採樣動作。通常測試時會設為 True，訓練時則為 False。

class Actor(nn.Module): # 結構必須與訓練一致
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mu = self.mu_head(x)
        log_std = torch.clamp(self.log_std_head(x), -20, 2)
        return mu, torch.exp(log_std)

    def sample(self, state):
        mu, std = self.forward(state)
        dist = Normal(mu, std)
        x_t = dist.rsample()        # 這裡的 x_t 是 mu + sigma * epsilon, epsilon 是從標準正態分布中採樣的雜訊。
        action = torch.tanh(x_t)    # tanh(mu + sigma * epsilon), tanh的使用只是為了把動作限制在 [-1, 1] 的範圍內.
        # Log-probability 修正 (Jacobian correction for tanh)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(1, keepdim=True)

def test():
    env = gym.make("LunarLanderContinuous-v3", render_mode="human")
    model = Actor(8, 2)
    model_path = "dqn_model/sac_actor.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("Model loaded successfully.")
    else:
        print("No model found!")
        return

    model.eval()
    for i in range(5):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                # action = model(state_t).numpy()[0]
                if DETERMINISTIC:
                    # 若要 deterministic 測試：直接取 mu 的值就好，並經過 tanh 限制在 [-1, 1] 範圍內。
                    mu, _ = model(state_t)
                    action = torch.tanh(mu).squeeze(0).cpu().numpy()
                else:
                    # 若要跟 SAC 訓練邏輯一致，則直接使用 sample()來採樣 action。
                    action, _ = model.sample(state_t)
                    action = action.squeeze(0).cpu().numpy()
                
            state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            total_reward += reward
        print(f"Test Episode {i+1}, Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    test()