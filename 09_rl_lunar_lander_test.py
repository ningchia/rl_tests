import gymnasium as gym
import torch
import numpy as np
import time
import torch.nn as nn

# 記得保留相同的 DuelingDQN 定義
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.value_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim))
    
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))

def evaluate_lander(num_tests=10):
    env = gym.make("LunarLander-v3", render_mode="human")
    model = DuelingDQN(env.observation_space.shape[0], env.action_space.n)
    
    if not os.path.exists("best_lunarlander_model.pth"):
        print("尚未找到權重檔案！")
        return

    model.load_state_dict(torch.load("best_lunarlander_model.pth"))
    model.eval()

    results = []
    for i in range(num_tests):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = model(torch.FloatTensor(state)).argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        results.append(total_reward)
        print(f"測試 {i+1}: {total_reward:.2f}")

    print(f"\n平均分數: {np.mean(results):.2f} (標準差: {np.std(results):.2f})")
    env.close()

if __name__ == "__main__":
    import os
    evaluate_lander()