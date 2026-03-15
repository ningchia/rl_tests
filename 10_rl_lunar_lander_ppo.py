# 在DQN中, DQN model輸入是state/observation的維度, 輸出則是action的維度, 代表的是採取哪一個action會獲得的reward. 
# 比如說lunar lander, action有4個維度, 分別代表不噴, 往左噴, 往下噴, 往右噴, 而model的輸出有4維, 分別代表做這4種
# 動作能獲得的reward (Q值). 在 DQN 的標準邏輯中，我們使用 argmax，也就是 「四選一」。所以，若需要「同時往左與往下噴」
# 在目前的Lunar Lander 的環境設定中是做不到的。
# 所以如果當 Agent 需要「同時做兩件事」，通常有三種解決方案：
# 1. 擴展動作空間 (Action Space Expansion)
#    這是最簡單的方法。我們可以把「組合動作」定義成一個新的整數。例如： 4: 同時左噴 + 主噴, 5: 同時右噴 + 主噴.
#    這樣輸出層就會變成 6 維。
# 2. 多重離散空間 (Multi-Discrete)
#    有些環境（如足球機器人或複雜賽車）會把動作拆成多個維度。例如： 維度 1：[不噴, 左噴, 右噴], 維度 2：[不噴, 主噴]
#    這時候模型的輸出就不是簡單的 4 維，而是兩組不同的輸出頭。
# 3. 連續動作空間 (Continuous Action Space)
#    這就是為什麼會有另一個環境叫 LunarLanderContinuous-v3。
#    在連續版中，動作不再是整數，而是一個包含兩個數值的向量：
#    主引擎：-1.0 到 +1.0（負數不噴，正數代表噴力大小）。
#    側邊引擎：-1.0 到 +1.0（負代表右噴，正代表左噴）。
#    在這種情況下，Agent 可以同時輸出「主引擎推力 0.8」與「側邊引擎推力 0.5」。
#    但注意，DQN 無法處理這種連續空間，必須換成 PPO, DDPG 或 SAC 等演算法。

# PPO (Proximal Policy Optimization)
#   核心關鍵字：穩定、通用、保險。
#   PPO 是一種策略梯度方法，適用於連續動作空間，能夠同時輸出多個動作維度（如主引擎和側邊引擎的推力）。
#   屬於 On-policy 演算法（邊跑邊學，跑完的資料就丟掉）。
#   剪裁功能（Clipped Objective）: 把更新策略幅度限制在一個安全的範圍內（通常是 20%），以免步子跨太大導致策略崩潰。
#   優點：非常穩定，不容易因為一次爛訓練就讓模型廢掉。
#   缺點：資料利用率低（資料用一次就丟，所以訓練需要非常多局）。
#
# DDPG (Deep Deterministic Policy Gradient)
#   核心關鍵字：DQN 的連續版、激進。把 DQN 的概念強行擴展到連續空間。
#   屬於 Off-policy（可以存記憶池重複學習）。由一個 Actor 負責決定動作，和一個 Critic（評論家） 負責打分（Q 值）。
#   Actor輸出一個確定的數值（例如：主噴推力 0.72）。Critic 告訴演員：這樣噴，預期分數是 80 分，再噴強一點可能會變 90 分。
#   Actor就根據評論家的意見去改進。
#   優點：資料利用率高，因為有記憶池。
#   缺點：極度不穩定，非常依賴超參數調整，且容易陷入局部最佳解（例如只會瘋狂往左噴）。
#
# SAC (Soft Actor-Critic)
#   核心關鍵字：效率高、愛探索、目前最強。
#   SAC 是目前處理連續控制任務（如機器人手臂、月球登陸器連續版）的最佳演算法之一。
#   是 DDPG 的升級版，但加入了一個關鍵概念——熵（Entropy）。
#   SAC 的目標不只是「拿到最高分」，還要「動作儘可能隨機/多樣化」。
#   AI 會想：「我要一邊拿高分，一邊嘗試各種不同的飛行姿勢。」這防止了 AI 太早認定某個平庸的策略就是最好的。
#   優點：非常耐操（Robust），探索能力極強，通常比 DDPG 快且穩。
#     最大熵原理 (Maximum Entropy RL)：
# #     SAC 不只追求高分，還追求動作的「多樣性」（即熵）。
#       這能防止 Agent 太早卡在某種奇怪的飛行姿勢（例如只會歪著飛），讓它能探索出最優雅、最省燃料的降落方式。
#     離策學習 (Off-policy)：
#       SAC 擁有像 DQN 一樣的記憶池（Replay Buffer）。
#       這意味著它會反覆利用過去的經驗，訓練效率遠高於 PPO。在同樣的訓練局數下，SAC 通常能更快達到 200 分。
#     自動調整熱度 (Automatic Entropy Tuning)：
#       現代的 SAC 版本會自動調整「探索」與「利用」的平衡，你不需要像 DQN 那樣痛苦地調整 Epsilon 衰減。
#   缺點：計算量比 PPO 大，實現起來比較複雜。

# 這裡我們實作 PPO，因為它在連續控制任務中表現穩定，且相對容易調整超參數。
# 基本上 PPO 的 model 輸出相當於 action 維度的 "幾組(2組, 左右噴, 往下噴) 機率分布" (分別由各組的 mu(mean) 
# 與 sigma(standard error) 定義出來), 然後後面在選擇動作時, 因為每個動作的大小"程度"是連續的值, 
# 就用這幾組"機率分布"中隨機取樣一個值出來, 作為選定的該動作的大小"程度". 
# 而之前在DQN裡, 當取樣的random值小於epsilon就隨機選一個動作, 在PPO裡則相當於sigma的意思, sigma大就代表
# "可能嘗試"的動作程度大小值範圍比較大的意思, mu就相當於目前"學得最好"的動作程度大小值的意思.

import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime

# --- PPO 網路架構 ---
class PPOModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOModel, self).__init__()
        # 共同特徵層
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Actor: 輸出動作的機率分佈 (由均值 mu 與 標準差 sigma 定義出來的機率分布)
        # 注意這邊的輸出(actor.mu與actor.sigma) 本身就是 長度為 2 的向量.
        # 第一維度 (Main Engine)：控制垂直方向噴力。數值範圍：[0, 1]（但模型輸出後通常映射到這裡）。
        # 第二維度 (Side Engines)：控制水平方向噴力。數值範圍：[-1, 1]。
        # 負數代表向右噴（推力向左），正數代表向左噴（推力向右）。
        self.actor_mu = nn.Linear(256, action_dim)      # head0: 輸出動作的均值 (mu)
        self.actor_sigma = nn.Linear(256, action_dim)   # head1: 輸出動作的標準差 (sigma)
        
        # Critic: 輸出狀態價值 (V-value)
        self.critic = nn.Linear(256, 1)                 # head2: 輸出狀態本身與動作無關的價值 (value)

    def forward(self, state):
        x = self.fc(state)
        mu = torch.tanh(self.actor_mu(x))               # 將head0的輸出(mu)限制在 [-1, 1]
        sigma = F.softplus(self.actor_sigma(x)) + 1e-5  # 將head1的輸出(sigma)限制在 [0, ∞)
        value = self.critic(x)
        return mu, sigma, value

# --- PPO 核心演算法 ---
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用兩組網路：
        # 一組是正在訓練的 policy 網路 (self.policy)，
        # 另一組是用來選擇動作的舊 policy 網路 (self.policy_old)。
        self.policy = PPOModel(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = PPOModel(state_dim, action_dim).to(self.device)
        # 初始時，讓舊網路的權重與正在訓練的網路相同。
        self.policy_old.load_state_dict(self.policy.state_dict())
        # PPO 的損失函數包含兩部分：策略損失和價值損失。
        # 策略損失使用剪裁的 surrogate objective, 意思是 「如果新策略和舊策略的比率 (pi_theta / pi_theta_old) 
        # 超過 1 + eps_clip 或低於 1 - eps_clip，就不再增加損失」，以保持更新的穩定性。
        # 而價值損失則使用均方誤差 (MSE)。
        self.mse_loss = nn.MSELoss()

    def select_action(self, state, storage):
        state = torch.FloatTensor(state).to(self.device)    # 將當前狀態轉換為 Tensor，並移動到 GPU（如果可用）。
        with torch.no_grad():
            # 用舊 policy 網路 (self.policy_old) 的輸出(當前狀態下的動作分佈 (mu, sigma) 和狀態價值 (value))來選擇動作。
            # 注意這邊的輸出(mu, sigma, value) 本身就是 長度為 2 的向量.
            mu, sigma, value = self.policy_old(state)
            
        # 根據 mu 和 sigma 定義一個正態分佈 (Normal distribution)，然後從這個分佈中隨機取樣一個動作 (action)。
        # 這裡的 action 是一個包含兩個數值的向量，分別代表主引擎和側邊引擎的推力大小。
        dist = Normal(mu, sigma)
        action = dist.sample()      # 以這個機率分布隨機取樣一個該動作 (action)的"值"。
        # 計算該動作是這個"值"的 log 機率 (log probability)，這在 PPO 的更新過程中會用到。
        # .sum(dim=-1) 是因為 action 是一個向量（包含兩個動作維度- 往下噴, 左右噴），我們需要把這兩個動作維度
        # 的 log 機率(在最後的維度上)加總起來，來得到整體動作的 log 機率。
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        storage.states.append(state)
        storage.actions.append(action)
        storage.logprobs.append(action_logprob)
        storage.values.append(value)
        # 將動作從 Tensor 轉換回 NumPy 陣列，以便在環境中使用。
        # .cpu() 是將 Tensor 從 GPU 移回 CPU，numpy() 是將 Tensor 轉換為 NumPy 陣列。
        return action.cpu().numpy() 

    # 把過去一陣子(2000步)的紀錄翻出來，看看哪幾步走得好，哪幾步走得爛。好動作多學一點，爛動作少做一點，
    def update(self, storage):
        # 計算回報 (Rewards-to-go)
        rewards = []
        discounted_reward = 0
        # is_terminal 是一個布林值，表示該步是否結束了這一局。
        # 當 is_terminal 為 True 時，表示這一步是該局的最後一步，後續的回報不應該再累積之前的獎勵，
        # 因此 discounted_reward 被重置為 0。
        for reward, is_terminal in zip(reversed(storage.rewards), reversed(storage.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            # 這裡的 discounted_reward 是從最後一步開始往前累積的回報值。
            # 每一步的回報等於當前的 reward 加上之前累計回報的折扣值 (gamma * discounted_reward)。
            # 即在時間點 t 採取動作後，最終能拿到的累積總分是多少（即真實目標值）
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # 轉換為 Tensor
        # torch.stack 是將列表中的 Tensor 沿著新的維度堆疊起來，形成一個新的 Tensor。
        # 這裡的 storage.states, storage.actions, storage.logprobs, storage.values 都是列表，
        # 每個元素都是一個 Tensor。
        # detach() 是將 Tensor 從計算圖中分離出來，這樣在更新過程中就不會對這些 Tensor 進行梯度計算。
        old_states = torch.stack(storage.states).detach()
        old_actions = torch.stack(storage.actions).detach()
        old_logprobs = torch.stack(storage.logprobs).detach()
        
        # 注意 storage.values 是一個列表，每個元素都是一個形狀是 [1] 的 Tensor（因為 critic 輸出的是一個scaler）。
        # squeeze() 是將 Tensor 中的單維度（即大小為 1 的維度）去掉，這樣 old_values 就會變成一維的 Tensor。
        old_values = torch.stack(storage.values).detach().squeeze()     # 舊的policy算出的狀態價值 (V-value)。
        
        # 這裡的 target_values 是 PPO 更新過程中的「真實得分」(來自environment的reward, 不是policy預測出來的)，
        # 而前面的 old_values 則是 PPO 更新過程中的「預期得分」(policy_old 預測出來的 "狀態價值")。
        target_values = torch.FloatTensor(rewards).to(self.device)      
        
        # 優勢函數 (Advantage). 優勢 (A) = 真實得分 - 預期得分。
        # 如果 A > 0：實際 reward 比 policy_old 預期的還要好，所以會在 policy 模型上強化產生這個動作的機率。
        # 如果 A < 0：實際 reward 比 policy_old 預期的還要差，所以會在 policy 模型上壓抑產生這個動作的機率。
        # 
        # 為何要用優勢函數?
        #
        # DQN時model輸出的是動作的Q值(reward), 而現在PPO model有兩組head, 其中:
        #   1. Actor輸出的是動作設定值的機率, 與該設定能獲得的reward沒有直接關係, 所以得要改透過優勢函數
        #      藉由"與狀態無關的狀態價值(reward)"來與reward掛勾以決定調整方向. 這就是所謂的 Policy Loss.
        #   2. 而Critic 直接就輸出動作無關的狀態價值(reward), 所以直接用MSE就好.
        advantages = target_values - old_values

        for _ in range(self.K_epochs):      # K_epochs 是每次 PPO 更新的迭代次數，通常是 10。
            # 在每次更新迭代中，使用正在訓練的 policy 網路 (self.policy) 計算當前狀態下的動作分佈 (mu, sigma) 
            # 和狀態價值 (values)。
            mu, sigma, values = self.policy(old_states)
            dist = Normal(mu, sigma)
            logprobs = dist.log_prob(old_actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            
            # PPO 核心：計算 Ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Clipped Surrogate Objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # PPO 的損失函數包含：
            #   Policy Loss = -torch.min(surr1, surr2) ：加負號是因為我們要「最大化」優點。
            #   Value Loss = 0.5 * MSELoss             ：這是 。訓練 Critic（評論家）讓它預測得分越來越準。
            #   混亂獎勵 = -0.01 * Entropy              ：獎勵那些 sigma 比較大的行為，防止 AI 太快變死板。
            # PPO的訓練目標: 
            #   Advantage 慢慢趨近於 0，但 Reward 卻很高.
            #   那就代表 Critic 已經變成了「預言家」，而 Actor 已經變成了「頂尖飛行員」。
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(values.squeeze(), target_values) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # 學習結束後，把新學好的權重同步給 policy_old，作為下一批收集資料時的參考基準。
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 清空 Storage：PPO 是 On-Policy，舊資料學完就不能再用了（因為策略已經變了），必須清空重抓。
        storage.clear()

class Storage:
    def __init__(self):
        self.states, self.actions, self.logprobs, self.rewards, self.is_terminals, self.values = [], [], [], [], [], []
    def clear(self):
        del self.states[:], self.actions[:], self.logprobs[:], self.rewards[:], self.is_terminals[:], self.values[:]

# --- 訓練主程式 ---
ENV_NAME = "LunarLanderContinuous-v3"
MAX_EPISODES = 1000
UPDATE_TIMESTEP = 2000 # 每累積 2000 步更新一次

MODEL_SAVE_PATH = "dqn_model"
CHECKPOINT_FILE = "ppo_lunar_lander.pth"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)

env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = PPOAgent(state_dim, action_dim)
storage = Storage()
writer = SummaryWriter(f'runs/PPO_Lander_{datetime.datetime.now().strftime("%H%M%S")}')

timestep = 0
for episode in range(1, MAX_EPISODES + 1):
    state, _ = env.reset()
    episode_reward = 0
    
    for t in range(500): # 每局上限步數
        timestep += 1
        # 在每一步中，Agent 根據 policy_old 為當前狀態選擇一個動作，然後將這個動作應用到環境中，獲得新的狀態、獎勵以及是否結束的資訊。
        action = agent.select_action(state, storage)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        storage.rewards.append(reward)
        storage.is_terminals.append(done)   # 記錄這一步是否結束了這一局，這對於計算回報 (Rewards-to-go) 非常重要。
        episode_reward += reward
        
        # 每累積 UPDATE_TIMESTEP (2000) 步就更新一次 PPO 模型。
        if timestep % UPDATE_TIMESTEP == 0:
            agent.update(storage)
            
        if done: break
        
    writer.add_scalar('Reward', episode_reward, episode)
    # 每 10 局輸出一次當前的獎勵，並且將模型權重保存到指定的檔案中。
    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")
        torch.save(agent.policy.state_dict(), checkpoint_path)