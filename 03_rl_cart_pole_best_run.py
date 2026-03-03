import os
import gymnasium as gym
import torch
import time

# 這裡要引用你定義的 DQN 類別 (或直接在腳本裡重新定義一次)
# 但是, 在 Python 中，import 語法本質上要求模組名稱必須符合 「標識符 (Identifier)」 的命名規則：不能以數字開頭。
# 因此, 你不能直接 import "02_rl_cart_pole.py" 這樣的檔案，因為它以數字開頭。
# --- Option-1 : 如果堅持不改檔名，可以使用 importlib 來動態加載模組。---
#   import importlib
#
#   # 動態載入檔名為 '02_rl_cart_pole.py' 的模組
#   # 注意：這裡不需要寫 .py 後綴
#   rl_module = importlib.import_module("02_rl_cart_pole")
#
#   # 從載入的模組中取得 DQN 類別
#   DQN = rl_module.DQN
#
# --- Option-2 (專業建議) : 將檔名改為符合 Python 命名規則(-> "rl_02_cart_pole.py")，然後直接使用 import 語法。---
# from rl_02_cart_pole import DQN  
#
# --- Option-3 : 將DQN 定義從02_rl_cart_pole.py 抽出來成為一隻獨立檔案並做成module ---
# 針對這個簡單 practice 感覺有點 lousy ...
#
# --- Option-4 : 將DQN 定義從02_rl_cart_pole.py 複製過來---
# 以目前02_rl_cart_pole.py的寫法, 由於沒有類似 if __name__ == "__main__": 的語法, 直接 import python file 
# 所有import的方式都會造成執行該檔案的side effect, 等於重新訓練一次的意思. 
# 正常應該要為02_rl_cart_pole.py加上 if __name__ == "__main__": 的語法來防止import時再度執行訓練, 
# 但這樣也會造成額外的import dependency 例如 tensorboard, setuptools等.
#
# 我們這邊偷懶一下, 直接複製過來就好. 
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim) # 輸出每個動作的 Q 值
        )
    
    def forward(self, x):
        return self.net(x)

MODEL_SAVE_PATH = "dqn_model"
CHECKPOINT_FILE = "best_cartpole_model.pth"
checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"未找到檢查點檔案: {checkpoint_path}")

def test_best_model():
    # 1. 初始化環境 (使用 render_mode="human" 讓你親眼看到它在玩)
    env = gym.make("CartPole-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 2. 載入訓練好的大腦
    model = DQN(state_dim, action_dim)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()  # 切換到評估模式

    state, _ = env.reset()
    state = torch.FloatTensor(state)
    total_reward = 0
    done = False
    path = [] # 紀錄動作路徑

    print("\n--- 開始測試最佳模型 ---")
    
    while not done:
        with torch.no_grad():
            # 完全不探索 (Epsilon=0)，只選 Q 值最高的最優動作
            action = model(state).argmax().item()
        
        path.append(action)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = torch.FloatTensor(next_state)
        total_reward += reward
        done = terminated or truncated
        
        # 稍微暫停一下，才不會飛快閃過
        time.sleep(0.02)

    print(f"測試結束，總得分: {total_reward}")
    print(f"最佳路徑動作序列: {path}")
    env.close()

if __name__ == "__main__":
    test_best_model()