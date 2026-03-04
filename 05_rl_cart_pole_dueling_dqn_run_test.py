import os
import gymnasium as gym
import torch
import time
import numpy as np

# 將Dueling DQN 定義從04_rl_cart_pole_dueling_dqn.py 複製過來---
import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        # 共享特徵提取層
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )
        
        # 狀態價值流 (Value Stream) - 輸出該狀態的基礎分數 (V)
        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 動作優勢流 (Advantage Stream) - 輸出每個動作的相對優勢 (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        # 確保輸入至少是 2D (batch_size, state_dim)
        # 如果 x 只有 1 維，會自動補成 (1, state_dim)，這樣就可以同時處理單筆和多筆輸入了。
        # 什麼時候 state 會是 1 維的？在訓練過程中，我們是一次訓練多筆經驗 (batch)，所以輸入通常是 2 維的 (batch_size, state_dim)。
        # 但是在評估過程中，我們可能會直接輸入單一的狀態 (state)，這時它就是 1 維的 (state_dim,)，需要補成 (1, state_dim) 才能通過網路。
        # 特別是之後在使用policy_net來預測"當前狀態下"每個動作的Q值然後選擇Q值最高的動作, 或是在執行測試時。
        # ex. 之後會執行到 action = policy_net(state).argmax().item()
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 為什麼原本的 02_rl_cart_pole.py 沒有補維度也沒事?
        # 因為在 02_rl_cart_pole.py 中，使用的是 nn.Sequential 直接輸出。
        # 對於 (4,) 的輸入，它會輸出 (2,)。執行到 action = policy_net(state).argmax().item()時, .argmax() 在一維向量上執行沒有問題。
        # 在 Dueling DQN 這裡不這樣會出問題的原因是, 因為我們在 forward 裡面寫了 advantage.mean(dim=1)。
        # 這個「指定維度求平均」的操作，本身就強制要求輸入必須具備 Batch 維度。

        features = self.feature_layer(x)            # 形狀為 (batch, 64)
        value = self.value_stream(features)         # 形狀為 (batch, 1)，代表每個狀態的基礎價值 V(s)
        advantage = self.advantage_stream(features) # 形狀為 (batch, action_dim)，代表每個動作的相對優勢 A(s,a)
        
        # 結合公式: Q(s,a) = V(s) + (A(s,a) - Mean(A(s,a)))
        # 減去平均值可以增加訓練的穩定性與模型識別度
        # q_values 的形狀為 (batch, action_dim)，代表每個狀態下每個動作的 Q 值。
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))    # 在 action_dim 上求平均，保持維度以便相減
        return q_values

MODEL_SAVE_PATH = "dqn_model"
CHECKPOINT_FILE = "best_cartpole_model_dueling_dqn.pth"
checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"未找到檢查點檔案: {checkpoint_path}")

def run_evaluation(num_tests=10):
    # 1. 初始化環境 (使用 render_mode="human" 讓你親眼看到它在玩)
    env = gym.make("CartPole-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 2. 載入訓練好的大腦
    model = DuelingDQN(state_dim, action_dim)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()  # 切換到評估模式

    all_rewards = []
    print(f"\n--- 開始模型評估：連續執行 {num_tests} 次 ---")

    for i in range(num_tests):

        state, _ = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0
        done = False
        # path = [] # 紀錄動作路徑

        print("\n--- 開始測試最佳模型 ---")
        
        while not done:
            with torch.no_grad():
                # 完全不探索 (Epsilon=0)，只選 Q 值最高的最優動作
                action = model(state).argmax().item()   # 注意這邊的 state 沒有batch維度 !! 進model時會自動補成 (1, state_dim).
            
            # path.append(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = torch.FloatTensor(next_state)
            total_reward += reward
            done = terminated or truncated
            
            # 稍微暫停一下，才不會飛快閃過
            # 測試時可適度縮短 sleep 或拿掉以節省時間
            # time.sleep(0.02)

        # print(f"測試結束，總得分: {total_reward}")
        # print(f"最佳路徑動作序列: {path}")
        all_rewards.append(total_reward)
        print(f"第 {i+1} 次測試，總得分: {total_reward}")

    # 3. 輸出統計結果
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
    run_evaluation(10)