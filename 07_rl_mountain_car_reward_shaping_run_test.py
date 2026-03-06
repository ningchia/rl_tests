import os
import gymnasium as gym
import torch
import numpy as np
import time

import torch
import torch.nn as nn

MODEL_SAVE_PATH = "dqn_model"
CHECKPOINT_FILE = "best_mountaincar_model.pth"
checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)

# --- Dueling DQN 架構 (從 06_rl_mountain_car_reward_shaping.py複製過來, 保持不變) ---
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, action_dim))
    
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))

def run_evaluation(num_tests=10):
    env = gym.make("MountainCar-v0", render_mode="human")
    model = DuelingDQN(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    all_rewards = []
    print(f"\n--- 開始模型評估：連續執行 {num_tests} 次 ---")

    for i in range(num_tests):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = model(torch.FloatTensor(state)).argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            time.sleep(0.01)
        all_rewards.append(total_reward)
        print(f"測試 {i+1}, 總得分 (Score): {total_reward}")
    
    # 輸出統計結果
    avg_score = np.mean(all_rewards)
    std_score = np.std(all_rewards)
    print("\n" + "="*30)
    print(f"評估完成！")
    print(f"平均得分: {avg_score:.2f}")
    print(f"標準差: {std_score:.2f}")
    print(f"最高分: {np.max(all_rewards)}")
    print(f"最低分: {np.min(all_rewards)}")
    print("="*30)

    env.close()

if __name__ == "__main__":
    run_evaluation(num_tests=10)