# need additional package : pip install gymnasium[box2d].
from numbers import Real
import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import datetime

# Lunar Lander 環境模型
# ref: https://gymnasium.farama.org/environments/box2d/lunar_lander/
#
# 狀態空間 (8 維): 座標 (x, y)、速度 (vx, vy)、角度、角速度，以及兩隻腳是否著地的布林值.
# 空間限制：Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ], [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)
#   x: [-2.5, 2.5]
#   y: [-2.5, 2.5]
#   vx: [-10., 10.]
#   vy: [-10., 10.]
#   angle: [-6.2831855, 6.2831855]
#   angular velocity: [-10., 10.]
#   left/right leg contact: [0., 1.]
# 動作空間 (4 維): 不動(0)、往左噴(1), 往下噴(2)、往右噴(3).
# 獎勵：
#   靠近降落點：得分增加。
#   移動太快：扣分（鼓勵減速）。
#   傾斜太嚴重：扣分（鼓勵垂直下降）。
#   腳著地：各 +10 分。
#   平穩降落：+100 分。
#   往左右噴的每一步：減 0.03 分（鼓勵快速完成任務）。
#   往下噴的每一步：減 0.3 分（鼓勵快速完成任務）。
#   墜毀：-100 分。
#   訓練目標：拿到 200 分 以上就被視為解決了該環境。

# --- Dueling DQN 架構 ---
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        # 共享特徵層 (輸出加大到 128 維並增加一層以提升表達能力)
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # 狀態價值流 (Value Stream)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 動作優勢流 (Advantage Stream)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        # 如果 x 只有 1 維，會自動補成 (1, state_dim)，這樣就可以同時處理單筆和多筆輸入了。
        if x.dim() == 1: x = x.unsqueeze(0)     

        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # 輸出 Q 值的計算公式: Q(s,a) = V(s) + (A(s,a) - Mean(A(s,a)))
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))

# --- 超參數 ---
ENV_NAME = "LunarLander-v3"
GAMMA = 0.99
LEARNING_RATE = 0.0005     # 稍微調低學習率以求穩定
MEMORY_SIZE = 50000        # 大記憶池. 幫助模型在長時間的嘗試中記住「偶然成功降落」的經驗。
BATCH_SIZE = 128           # 大 Batch 增加梯度穩定性. 如果有GPU可以更大一些。
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.996      # 較慢的衰減. 前期非常容易墜毀，所以讓探索時間拉長一點。
TARGET_UPDATE = 10
EPISODES = 1000            # 一般需要run 2 到 3次 1000 局(第二次後都讓EPSILON_START回到0.3)的訓練後效果才比較好

USE_DDQN = True            # 是否使用 Double DQN 來減少 DQN 的過度估計問題。Double DQN 通過分離動作選擇和動作評估來提供更穩定的學習目標。
USE_SOFT_UPDATE = True     # 是否使用軟更新 (Soft Update) 來平滑地更新目標網路的權重。軟更新會將目標網路的權重逐步向當前網路的權重靠攏，這有助於穩定訓練過程。
TAU = 0.005                # 軟更新的係數，值越小表示更新越慢，通常在 0.001 到 0.01 之間選擇。

MODEL_SAVE_PATH = "dqn_model"
CHECKPOINT_FILE = "best_lunarlander_model.pth"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)

# 設定設備 (GPU 或 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    BATCH_SIZE = 256  # 如果有 GPU，可以使用更大的 Batch Size 來加速訓練
print(f"目前使用的設備: {device}, Batch Size: {BATCH_SIZE}")

# 初始化.
# 在訓練過程中不需要 render，等訓練完成後再用測試腳本來觀看模型表現。
# env = gym.make(ENV_NAME, render_mode="human")     
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DuelingDQN(state_dim, action_dim).to(device)
target_net = DuelingDQN(state_dim, action_dim).to(device)

# --- 載入舊有的經驗 ---
if os.path.exists(checkpoint_path):
    print(f"--- 偵測到既有模型權重 {checkpoint_path}，載入中... ---")
    # 記得使用 map_location 確保設備正確
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy_net.load_state_dict(state_dict)
    target_net.load_state_dict(state_dict)
    
    # 既然已經有基礎了，我們可以把初始探索率 EPSILON_START 調低
    # 例如從 0.3 開始，而不是 1.0，這樣可以減少前期亂噴氣的時間
    EPSILON_START = 0.3 
else:
    print("--- 未發現既有權重，將從隨機初始化開始訓練 ---")

target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(f'runs/DuelingDQN_LunarLander_{current_time}')

best_reward = -np.inf

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    total_shaped_reward = 0  # 紀錄加權後的獎勵供觀察
    done = False
    
    # 計算目前的 epsilon
    epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** episode))
    
    while not done:
        # Epsilon-Greedy 選擇動作
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                # 注意這邊的 state 沒有batch維度 !! 進model時會自動補成 (1, state_dim).
                # 推論時也要把 state 搬到 GPU
                state_t = torch.FloatTensor(state).to(device)
                action = policy_net(state_t).argmax().item() 
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # ------------------------------------------------------------
        # Reward shaping:
        #
        # 注意 Reward Shaping 太重了會蓋過環境原本的目標，甚至讓 AI 學壞（Reward Hacking）。
        # 量級對齊法則 (Magnitude Alignment):
        #   最基本的規則是：Shaping 的總和，不應該超過原始環境單步 Reward 的量級。
        #   觀察原始值：Lunar Lander 每步的 Reward 通常在 -1.0 到 +1.0 之間（墜毀或降落除外）。
        #   所以 reward shaping 建議控制在原始值的 10% ~ 50% 之間。
        #   如果原始 Reward 是 1.0，你的 Shaping 設定在 0.1 ~ 0.5 比較安全。
        #   如果給一個高達 10.0 的 Shaping，Agent 會發現：「只要在那邊練習拿 Shaping 就好了，幹嘛要冒險去降落？」
        # 總分守恆原則 (Total Potential Rule):
        #   這是一個比較進階的視角：不管加了多少 Shaping，一局下來的「額外獎勵」總和，應該遠小於「成功完成任務」的大獎.
        #   Lunar Lander 成功降落有 +100 ~ +200 分。
        #   守恆檢查：如果你每一步都給 0.5 的「高度獎勵」，一局跑 400 步，AI 就拿到了 200 分。
        #   這時 AI 會覺得：「我不降落也拿到了滿分，那我就在空中跳舞吧！」
        #   經驗法則：Shaping 的累積總分，建議不要超過最終大獎的 20% ~ 30%。
        # 使用Potential-based Reward Shaping:
        #   這是學術界最推薦的方法。為了避免 AI 為了刷分而來回刷獎勵，Shaping 應該以「狀態的差異」來給予，而不是給「固定值」。
        #   錯誤寫法：if angle < 0.1: reward += 0.1 (AI 可能會在那邊抖動來反覆領這 0.1 分)。
        #   正確寫法：shaping = (prev_dist - current_dist)。
        #   意義：這代表「進步了才給獎勵」。如果 AI 走回頭路，這個值就會變負的，自動抵銷。
        # 根據經驗，不同類型的行為建議給予不同的權重：(相對於單步)
        #   目的生存/時間懲罰 -0.01 ~ -0.1, 逼它快點結束任務
        #   精細控制 (如角度/速度) -0.05 ~ -0.2, 微調動作，不宜過大以免變膽小
        #   里程碑 (如腳觸地) +1.0 ~ +5.0, 標示階段性成功
        #   關鍵懲罰 (如翻覆/墜毀)-50 ~ -100, 必須讓它「刻骨銘心」
        # 記得在 TensorBoard 分開紀錄兩個數值：
        #   Real Reward：環境給的原始分數（真理）。
        #   Shaped Reward：加料後的分數（誘導）。
        #   如果 Shaped Reward 一直升，但 Real Reward 沒動，說明 AI 在「鑽漏洞（Reward Hacking）」，你的 Shaping 給太重或邏輯有誤。
        #   如果 Shaped Reward 升得很慢，說明 Shaping 太淡，AI 感覺不到引導。
        # -------------------------------------------------------------
        x = next_state[0]               # 水平位置, range : -2.5 ... 2.5
        height = next_state[1]          # 垂直高度, range : -2.5 ... 2.5
        vel_x = next_state[2]           # 水平速度, range : -10 ... 10
        vel_y = next_state[3]           # 垂直速度, range : -10 ... 10
        angle = next_state[4]           # 角度, range : -6.2831855 ... 6.2831855 (360度的弧度表示)
        angular_velocity = next_state[5]   # 角速度, range : -10 ... 10
        leg_left_contact = next_state[6]   # 左腳接觸地面
        leg_right_contact = next_state[7]  # 右腳接觸地面
        
        custom_reward = reward  # 預設為原始獎勵，後續會加上 shaping 的部分

        # 稍微強化著地的正面回饋, 鼓勵及早降落. 
        if leg_left_contact == 1 or leg_right_contact == 1:
            custom_reward += 1.0

        # 修正「初期不調整」：
        # 讓它在高度還很高時就趕快往中央（x=0）靠攏。
        # 如果水平位置偏離中心 (x 不等於 0)，且高度還很高 (y > 0.5, 最高約 1.5)，就給予負分，鼓勵它早點調整。
        # 高度越高時，對位移的忍受度越低，強迫它在高處就對準目標
        dist_from_center = abs(x)
        if height > 0.5:
            custom_reward -= dist_from_center * 0.1    # x range : -2.5 ... 2.5

        # 我們希望角度盡量接近 0 (垂直向上)
        custom_reward -= abs(angle) * 0.05              # angle range : -6.2831855 ... 6.2831855

        # 改善「翻倒墜毀」：
        # 1. 強化角速度懲罰 (防止瘋狂旋轉)
        custom_reward -= abs(angular_velocity) * 0.05   # angular velocity range : -10 ... 10

        # 朝向中心移動的速度獎勵:
        # 如果距離中心遠，我們就獎勵「朝向中心移動的速度」；如果已經在中心上方，則獎勵「水平靜止」。
        # 如果 dist_x 為正(在右邊)，vel_x 為負(往左移)才是正確方向. 二者相乘為負代表方向正確.
        # 在高空時鼓勵往中心靠
        if height > 0.3 :
            custom_reward += -x * vel_x * 0.1  # 獎勵朝中心移動的速度. x range : -2.5 ... 2.5, vel_x range : -10 ... 10
        
        # 修正「下降太快」：垂直速度控制
        # 理想的下降速度隨高度減少。高處可以快一點，低處必須慢。也要避免停在空中太久。
        # 如果在低空 (height < 0.4)且下降太快才罰 (vel_y 是負的代表下降)
        if vel_y < -0.4 and height < 0.5:
            custom_reward -= abs(vel_y) * 0.1

        # 修正「著陸後亂噴」：速度與觸地懲罰
        # 一旦腳部接觸地面，我們應該強烈要求它靜止。
        if leg_left_contact > 0 or leg_right_contact > 0:
            # 懲罰觸地後的殘餘速度，強迫它安分下來
            custom_reward -= (abs(vel_x) + abs(vel_y)) * 0.1   # vel_x range : -10 ... 10, vel_y range : -10 ... 10
            
            # [關鍵] 如果腳觸地了，卻還在發動引擎 (action != 0)，給予額外懲罰，鼓勵它一旦著陸就不要再亂噴氣了。
            if action != 0:
                custom_reward -= 1.0

        # 如果墜毀 (reward == -100)，我們手動加重一點點
        if terminated and reward <= -90:
            custom_reward = -200 # 讓它對墜毀更反感

        # -------------------------------------------------------------

        # 儲存經驗並更新狀態
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        total_shaped_reward += custom_reward

        # 訓練邏輯
        if len(memory) > BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            s_batch, a_batch, r_batch, ns_batch, d_batch = zip(*batch)
            
            s_batch = torch.FloatTensor(np.array(s_batch)).to(device)
            a_batch = torch.LongTensor(a_batch).to(device).unsqueeze(1)
            r_batch = torch.FloatTensor(r_batch).to(device)
            ns_batch = torch.FloatTensor(np.array(ns_batch)).to(device)
            d_batch = torch.FloatTensor(d_batch).to(device)
            
            # policy_net(s_batch) 的輸出是 (BATCH_SIZE, action_dim)，我們用 gather 取出對應動作的 Q 值.
            # current_q 的 shape 是 (BATCH_SIZE, 1)，而 target_q 的 shape 是 (BATCH_SIZE,)，所以在計算 loss 時要把 current_q 壓平成 (BATCH_SIZE,)
            # 這裡的current_q沒有包在torch.no_grad() context中, 所以會被列入計算圖中, 在後面叫用loss.backward()更新梯度時
            # 就會被更新到. 這樣一來current_q 計算圖上產生的梯度會回傳到 policy_net 的參數，從而更新網路。
            current_q = policy_net(s_batch).gather(1, a_batch)
            
            with torch.no_grad():
                if not USE_DDQN:
                    # 這邊還是使用 DQN 的 target 計算方式 (由target net直接預測下一個狀態可能拿到的最大Q值)，
                    # 而非像 DDQN 那樣先由 policy net 選下一個狀態可能最高Q值的動作, 再由 target net 評估該動作的可能評分。
                    next_q = target_net(ns_batch).max(1)[0].detach()

                else:
                    # Double DQN 的 target 計算方式
                    # 1. 使用 Policy Net (當前網路) 選出"下一個狀態"的最佳動作 (A')
                    best_actions = policy_net(ns_batch).argmax(dim=1, keepdim=True)
                    
                    # 2. 使用 Target Net (目標網路) 來評估這個動作 A' 的 Q 值
                    # gather(1, best_actions) 會從 Target Net 的輸出中提取 A' 對應的分數
                    next_q = target_net(ns_batch).gather(1, best_actions).squeeze(1)

                # 計算目標 Q 值 (Bellman Equation)
                target_q = r_batch + (1 - d_batch) * GAMMA * next_q

            loss = nn.MSELoss()(current_q.squeeze(), target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    if USE_SOFT_UPDATE:
        # 軟更新目標網路
        for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
    else:
        # 每 10 局同步一次目標網路
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
    # 儲存最佳模型
    if total_reward > best_reward and episode > 100:
        best_reward = total_reward
        torch.save(policy_net.state_dict(), checkpoint_path)
    
    writer.add_scalar('Performance/Reward', total_reward, episode)
    writer.add_scalar('Performance/Shaped_Reward', total_shaped_reward, episode)
    print(f"Episode {episode}, Reward: {total_reward:.2f}, Shaped: {total_shaped_reward:.2f}, Epsilon: {epsilon:.2f}")

writer.close()