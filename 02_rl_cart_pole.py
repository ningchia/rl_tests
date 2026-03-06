# 需要 pip install torch, gymnasium[toy-text] 和 numpy. 

import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# 使用 PyTorch 內建的 SummaryWriter (TensorBoard) 來監控訓練, 來解決「如何看成果」與「診斷問題」的需求.
# 需要安裝 tensorboard 與 setuptools 套件 (pip install tensorboard "setuptools<82.0.0")，
#
# 注意, 根據這個網頁(https://discuss.frappe.io/t/bench-new-site-throws-error-no-module-named-pkg-resources/160415/9),
# setuptools v82.0.0 也移除了 pkg_resources這個模組, 所以安裝的setuptools也要< v82.0.0. (pip install "setuptools<82.0.0")
#
# 然後在終端機輸入 tensorboard --logdir=runs 來啟動 TensorBoard 伺服器，
# 最後在瀏覽器中打開 http://localhost:6006 就可以看到訓練的過程和結果了。
from torch.utils.tensorboard import SummaryWriter
import datetime

# r1. 初始化tensorboard紀錄器 (會建立一個 runs 資料夾)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(f'runs/DQN_CartPole_{current_time}')

# 1. 定義類神經網路 (大腦), 用來預測某一個observation/state下每個action的Q值 (價值)
#    所以 input tensor 是 state_dim (4個觀察值, 位置、速度、角度、角速度)，output是action_dim (2個動作, 左右)，
#    中間有兩層隱藏層，每層64個神經元，激活函數是ReLU。
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

# 2. 超參數設定
BATCH_SIZE = 64
GAMMA = 0.99            # 折扣因子 (gamma)，用來平衡當前獎勵和未來獎勵的重要性。接近1表示未來獎勵幾乎和當前獎勵一樣重要，接近0表示只關心當前獎勵。
EPSILON_START = 1.0     # 一開始完全探索 (epsilon = 1.0)，隨著訓練逐漸減少探索，增加利用已學習的知識。
EPSILON_END = 0.01      # 最小探索率，保證即使訓練後期也有少量隨機探索。
EPSILON_DECAY = 0.995   # 每一回合結束後，epsilon 會乘以這個衰減因子，逐漸減少探索。
LR = 0.001              # 學習率 (learning rate)，控制模型權重更新的幅度。過大可能導致訓練不穩定，過小可能導致收斂過慢。
MEMORY_SIZE = 10000     # 經驗回放池的大小，存儲過去的經驗以供訓練使用。過小可能導致模型無法學習到足夠的經驗，過大可能佔用過多記憶體。
TARGET_UPDATE = 10      # 每多少回合更新一次目標網路，這是 DQN 中用來穩定訓練的一個技巧，通過減少目標網路更新的頻率來降低訓練過程中的震盪。

EPISODES = 500          # 玩幾局，每一局都是從環境的初始狀態開始，直到遊戲結束。更多的回合通常可以讓模型學習得更好，但也會增加訓練時間。
ACTS_PER_EPISODE = 200  # 每局最多執行多少個動作，這是為了防止某些情況下遊戲無法結束而導致訓練無限進行下去。

USE_DDQN = True         # 是否使用 Double DQN 來減少 DQN 的過度估計問題。Double DQN 通過分離動作選擇和動作評估來提供更穩定的學習目標。

MODEL_SAVE_PATH = "dqn_model"
CHECKPOINT_FILE = "best_cartpole_model.pth"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)
best_reward = 0  # 紀錄歷史最高分

# 3. 初始化環境與模型
#    ref: https://gymnasium.farama.org/environments/classic_control/cart_pole/
env = gym.make("CartPole-v1")
# state_dim 取決於環境。在 CartPole 中是 4 維（位置、速度、角度、角速度）
# action_dim 取決於環境。在 CartPole 中是 2（向左或向右）。
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# DQN 的核心思想：使用 "當前的model" 來提供 "當前狀態" 所有可能動作的Q值，但使用 "目標model" 來評估 "下一個動作" 的價值(Q值)作為學習目標，
# 而且多個batch後才同步一次"當前的model"與"目標model"來減少訓練過程中的震盪
policy_net = DQN(state_dim, action_dim) # 正在學習的網路, 用來預測"當前的狀態下"採取每一種可能action的Q值.
target_net = DQN(state_dim, action_dim) # 前一次的policy_net, 被用來預測"下一個狀態可能的最大Q值", 做為學習目標.
target_net.load_state_dict(policy_net.state_dict())     # 使用之前學習的網路來初始化目標 Q 值，讓訓練更穩定
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

# 另一個減少震盪的功臣 —— 經驗回放 (Experience Replay)
# deque 是一種雙端(double-ended)隊列，可以在兩端快速添加和刪除元素，這裡用來存儲過去的經驗，當超過 MEMORY_SIZE 時會自動丟棄最舊的經驗。
memory = deque(maxlen=MEMORY_SIZE)      # 經驗回放池. 

epsilon = EPSILON_START

# 4. 訓練迴圈
for episode in range(EPISODES):
    state, _ = env.reset()
    state = torch.FloatTensor(state)    # 將狀態(觀察值, 包含位置、速度、角度、角速度)轉成張量形式
    total_reward = 0

    total_loss = 0
    train_steps = 0
    
    for t in range(ACTS_PER_EPISODE):
        # Epsilon-Greedy 選擇動作
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                # 使用policy_net來預測"當前狀態下"每個動作的Q值，然後選擇Q值最高的動作.
                # 這是一個推論的動作,不會更新權重，所以用torch.no_grad()來避免計算梯度，節省記憶體和運算時間。
                action = policy_net(state).argmax().item()  # 注意這邊的 state 沒有batch維度 !!
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state)
        
        # 儲存經驗. 每一筆經驗包含當前狀態、所選擇的動作、這個動作所獲得的獎勵、下一個狀態和是否結束的標記。
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        
        # 當記憶夠多時(>64筆)，開始訓練(逐筆sample更新Q值).
        if len(memory) > BATCH_SIZE:
            # 隨機從記憶中抽取一批(64筆)"抉擇"的經驗來訓練. 
            # input與output都會多使用一個batch維度，這樣就可以一次訓練多筆經驗，提高效率。
            # batch 是一個列表，裡面有 64 筆經驗，每筆經驗是一個 tuple (state, action, reward, next_state, done)。
            batch = random.sample(memory, BATCH_SIZE)

            # 將經驗中的狀態、動作、獎勵、下一個狀態和是否結束的標記分別打包成張量，方便後續計算。 
            # *batch 是將 batch 解包成五個元素，分別對應 states, actions, rewards, next_states, dones
            # zip(*batch) 是將這五個元素分別打包成五個獨立的列表(list)，然後再轉換成帶有batch維度的張量。
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.stack(states)                        # 將狀態列表轉換成一個張量，形狀是 (64, state_dim)，每一行是一個狀態。
            actions = torch.LongTensor(actions).unsqueeze(1)    # 將動作列表轉換成張量，形狀是 (64, 1). 
                                                                # unsqueeze(1) 是將動作張量從 (64,) 變成 (64, 1)，這樣才能在後面用
                                                                # gather 函數來選取對應的 Q 值。
            rewards = torch.FloatTensor(rewards)                # 將獎勵列表轉換成張量，形狀是 (64,)
            next_states = torch.stack(next_states)              # 將下一個狀態列表轉換成張量，形狀是 (64, state_dim)
            dones = torch.FloatTensor(dones)                    # 將是否結束的標記轉換成張量，形狀是 (64,)
            
            # 計算當前的 Q 值
            # policy_net(states) 的輸出是 (batch, action_dim)，每一行代表batch中的一個sample在對應的狀態下可以採取的每個動作的 Q 值。
            # policy_net(states).gather(1, actions) 就是從這些 Q 值中選取 agent 實際執行的動作的 Q 值。
            # gather函數的原型是 input.gather(dim, index) 的意思是從輸入張量的第 dim 維度上，根據 index 中的索引來選取元素。:
            #   輸入張量在這裡是 policy_net(states) 的輸出。形狀是 (64, action_dim), 即 (64,2).
            #   第一個參數是維度，這裡我們在維度 1 (action_dim) 上根據 actions 選取 Q 值。
            #   第二個參數是索引張量, 代表batch中每一筆sample採取的action index.
            # 因為 gather 要求 index 的維度必須與 input 一致。input 是二維的 (64, 2)，所以 actions 也必須是二維的 (64, 1)。
            # 結果的形狀在index的維度上 length 會變成 1, 因為只選1個.
            # 這裡的 current_q 是一個形狀為 (64, 1) 的張量，代表 batch 中每一筆經驗所選擇的動作對應的 Q 值。
            #
            # 這裡有一個關鍵：DQN運練需要的梯度是在哪裡產生的？
            # 答案是：這裡的current_q沒有包在torch.no_grad() context中, 所以會被列入計算圖中, 在後面叫用loss.backward()更新梯度時
            # 就會被更新到. 這樣一來current_q 計算圖上產生的梯度會回傳到 policy_net 的參數，從而更新網路。
            current_q = policy_net(states).gather(1, actions)   # 形狀為 (64, 1). 這邊的 states 是有 batch 維度的 !!

            # torch.no_grad() : 
            #   真實動作是將這些張量的計算過程直接不列入computation graph.
            # requires_grad = False：
            #   通常用於模型微調 (Fine-tuning, transfer learning)。
            #   計算圖依然會建立，但在後面叫用loss.backward()更新計算圖中的梯度時就不要更新它們.
            #   相當於梯度流到這一層時會「斷掉」，不會計算該權重的變化。
            
            # 計算目標 Q 值 (使用 Target Network 讓訓練更穩定)
            with torch.no_grad():
                # 尋找下一個狀態的最大 Q 值：一樣用推論模式不計算梯度.
                if not USE_DDQN:
                    # --- Option-1 : DQN, 由 target_net 選擇並評估下一個動作的價值, 可能有過度估計 (Overestimation) 的問題 -----------
                    # 這是 DQN 的核心思想：使用當前的網路來選擇動作，但使用目標網路來評估下一個動作與其價值，這樣可以減少訓練過程中的震盪。
                    # target_net(next_states)的輸出形狀是 (batch, action_dim), 也就是 (64, 2).
                    # max(1) 是指在維度1 (action_dim) 上尋找最大(Q)值以及其位置index，會返回兩個形狀為 (64,) 的張量：最大值 和 對應的索引，
                    # 我們只需要最大值，所以用 [0] 來選取。
                    next_q = target_net(next_states).max(1)[0]      # next_q 是一個形狀為 (64,) 的張量，代表 batch 中每一筆經驗對應的
                                                                    # 下一個狀態的最大 Q 值。 這邊的 next_states 也是有 batch 維度的 !!
                else:
                    # --- Option-2 : Double DQN (DDQN), 用 policy_net 選擇下一個動作, 再由target_net 評估其價值 -----------
                    # 用來解決 DQN 的 Overestimation 的問題.
                    # 好比問兩個不太準確的專家：
                    # DQN：問專家 A (target_net) 哪個方案最好，專家 A 說方案一，並且說它值 100 分。你直接信了。
                    # DDQN：問專家 A (target_net) 哪個方案最好，專家 A 說方案一。
                    #       你轉頭問專家 B (policy_net)：「方案一你覺得值幾分？」專家 B 比較冷靜，說：「我覺得只值 60 分。」

                    # 1. 由 Policy Net 選出「看起來最強」的動作 (Selection). 
                    #    keepdim=True 是為了保持維度，這樣 best_actions 的形狀就會是 (64, 1)，方便後面用 gather 函數來選取對應的 Q 值。
                    #    如果不使用 keepdim=True，best_actions 的形狀會是 (64,)，這樣在後面用 gather 函數時就會出現維度不匹配的問題。
                    best_actions = policy_net(next_states).argmax(dim=1, keepdim=True)      # 形狀: (64, 1). 
                    
                    # 2. 由 Target Net 來評估「那個動作」到底值多少分 (Evaluation)
                    #    使用 gather 從 Target Net 的輸出中挑選 Policy Net 選中的動作
                    next_q = target_net(next_states).gather(1, best_actions).squeeze(1)     # (64, 1) -> squeeze(1) 變成 (64,)

                # 計算目標 Q 值，根據 DQN 的更新公式：
                # 如果 done = 0 (還沒結束)：(1 - 0) = 1。目標值 = 當前獎勵 + 未來的折現價值。這是標準邏輯。
                # 如果 done = 1 (掉進洞裡或遊戲結束)：(1 - 1) = 0。這會把後面的未來價值「歸零」。
                # 意義是：如果這一步已經讓遊戲結束了，那後面就沒有「未來」可言，得分就僅限於當下的獎勵。
                target_q = rewards + (1 - dones) * GAMMA * next_q   # 形狀為 (64,) 的張量，代表 batch 中每一筆經驗的目標 Q 值。
            
            # 計算損失並更新參數
            # MSELoss 是均方誤差損失函數，計算當前 Q 值和目標 Q 值之間的差距，這個差距就是我們要最小化的損失。
            # 監督式學習的backward update是從loss一次性的回推更新所有的權重與Bias,
            # 而強化學習的backward update則是逐步地在每一步中, 用下一步的預測Q值扮演target值, 來進行當前這一步的更新。
            loss = nn.MSELoss()(current_q.squeeze(), target_q)  # current_q 的形狀是 (64, 1)，需要用 squeeze() 將其變成 (64,) 以匹配 target_q 的形狀。
            optimizer.zero_grad()           # 在 PyTorch 中，每次反向傳播之前都需要先將梯度歸零，否則梯度會累積。
            loss.backward()                 # 反向傳播，計算損失對模型參數的梯度。
            optimizer.step()                # 更新模型參數

            total_loss += loss.item()
            train_steps += 1
            
        if done: break
    
    # 在每一局結束後，判斷是否需要儲存模型
    if total_reward >= best_reward:
        best_reward = total_reward
        # 儲存目前的最佳模型權重
        torch.save(policy_net.state_dict(), checkpoint_path)
        print(f"--- 發現更優模型！得分: {total_reward}，已儲存權重 ---")

    # 每 TARGET_UPDATE (10) 個 Episode 同步一次目標網路
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    # 隨著訓練進行，逐漸減少 epsilon 的值，讓 agent 越來越傾向於利用已學習的知識，而不是隨機探索。
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    # ---- tensorboard 紀錄 ----
    # r2. 紀錄每局的總回報 (Total Reward)
    writer.add_scalar('Performance/Reward', total_reward, episode)
    
    # r3. 紀錄訓練中的 Loss (假設你在訓練步驟中存了 loss_val)
    #     在 Python 中，for 迴圈或 if 判斷式並不會產生新的作用域（Scope）。
    #     這意味著在 for 迴圈內定義的變數（如 loss），在迴圈結束後依然可以被外部存取。
    #     但是為了避免混淆, 我們還是改成使用 total_loss/train_steps 來紀錄平均 Loss, 這樣就不會有「變數作用域」的問題了。
    """
    if 'loss' in locals():
        writer.add_scalar('Optimization/Loss', loss.item(), episode)
    """
    # 在每一局結束後紀錄平均 Loss
    if train_steps > 0:
        writer.add_scalar('Optimization/Loss', total_loss / train_steps, episode)

    # r4. 紀錄目前的探索率 (Epsilon)
    writer.add_scalar('Parameters/Epsilon', epsilon, episode)

# 訓練結束後關閉
writer.close()

env.close()