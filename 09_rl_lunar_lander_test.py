import os
import gymnasium as gym
import torch
import numpy as np
import time
import torch.nn as nn

MODEL_SAVE_PATH = "dqn_model"
CHECKPOINT_FILE = "best_lunarlander_model.pth"
checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)

# 設定設備 (GPU 或 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"目前使用的設備: {device}")

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
    
    if not os.path.exists(checkpoint_path):
        print("尚未找到權重檔案！")
        return

    # PyTorch 在儲存模型權重（.pth 檔案）時，預設會連同權重所在的「設備資訊（Device Information）」一起存進去.
    # 當需要從"檔案"來load state dict時, 要使用 map_location 將權重dict對應到目前的設備 (不論是 CPU 還是 GPU)
    # 這樣即使模型是在 GPU 上訓練的，你也可以在只有 CPU 的電腦上跑測試。
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # 模型本體也要搬到該設備
    model.to(device)

    model.eval()

    results = []
    for i in range(num_tests):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            # 從 env.reset() 或 env.step() 拿到的 state 是 NumPy array，必須先轉成 Tensor 並搬到與模型相同的設備上，
            # 否則模型會抱怨「資料在 CPU，但權重在 GPU」。
            # 將 state 轉成 tensor 並搬到同一設備上，然後用模型來選擇動作。
            state_tensor = torch.FloatTensor(state).to(device)
            action = model(state_tensor).argmax().item()
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