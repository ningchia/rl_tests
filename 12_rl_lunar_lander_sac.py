#--------------------------------------------------------------------------------
# 這裡我們實作 SAC，一個適用於連續動作空間的 Off-policy 演算法。
# 主要特色：
# 1. Actor-Critic 架構：同時訓練一個策略網路 (Actor) 和兩個價值網路 (Critic)。
# 2. 自動調整溫度參數 (Alpha)：平衡探索與利用。
# 3. 重參數化技巧：讓策略網路能夠直接輸出動作分布的參數，並從中採樣。
# 4. Twin Critic：使用兩個 Critic 網路來減少過度估計的問題。
# 5. SAC 是 Off-policy，需要記憶池來存儲過去的經驗，並從中隨機抽取樣本進行訓練。
#--------------------------------------------------------------------------------

import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import datetime

# --- 1. Replay Buffer (SAC 是 Off-policy，需要記憶池) ---
# 在 PPO 中，我們用 storage 存完就刪；但在 SAC 中，我們使用 ReplayBuffer。
# 機制：它是一個循環隊列（Capacity 通常設為 10 萬到 100 萬步）。
# 目的：打破資料之間的相關性（Correlation）。從中隨機抽樣來訓練，能讓神經網路看到各種不同時期的經驗，訓練會更穩定。
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    # 從 Replay Buffer 中隨機抽取一批經驗, output 為 state, action, reward, next_state, done 的 batch (numpy array)。
    # batch 是一個 list，裡面有 batch_size 個 (state, action, reward, next_state, done) 的 tuple。
    # *batch 是 Python 的解包語法，會把 list 中的 tuple 分別解包成"五個獨立的 list"，分別對應 state, action, reward, next_state, done。
    # map(np.stack, zip(*batch)) 的作用是把這個 list 轉換成五個 numpy array，分別對應 state, action, reward, next_state, done。
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# --- 2. 網路架構 ---
class Actor(nn.Module):
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
        # log_std 的輸出被限制在 [-20, 2] 的範圍內，這樣可以確保 std 的值不會太大或太小，避免數值不穩定。
        log_std = torch.clamp(self.log_std_head(x), -20, 2)
        return mu, torch.exp(log_std)   # Actor 的輸出是動作分布的參數 (mu 和 std)，而不是直接輸出動作。

    # 用 重參數化技巧 選取一個動作並回傳 log-probability。
    #
    # 在 PPO 中，由於"動作"是採樣出來的，無法求導數, 所以梯度無法傳到model的weight與bias。
    # 但在 SAC 的 Actor.sample() 裡, rsample()代表的是 "重參數化採樣",
    # 意思是, 我們不是直接從分布中採樣"動作", 而是先從標準正態分布中採樣一個"雜訊" epsilon, 然後計算 Action = mu + sigma * epsilon。
    # ( 但其實, 在這裡的實作中 Action 是 tanh(mu + sigma * epsilon), tanh的使用只是為了把動作限制在 [-1, 1] 的範圍內 )
    # 由於 loss 是關於 Action 的, 而 Action 又是 mu 和 sigma 的函數 (把epsilon作為常數代入), 這樣就能讓梯度通過 Action 傳到
    # mu 和 sigma 的參數上, 進而更新 Actor 網路的權重。
    # 而當我們更新 Actor 時，目標是極大化 Q(s, a). 所以loss function 是 -Q(s, a), 這樣 Actor 就會學習產生更高 Q 值的動作。
    def sample(self, state):
        mu, std = self.forward(state)
        dist = Normal(mu, std)
        # rsample() 是"重參數化採樣"而非"一般的採樣"，讓我們能夠從分布中採樣同時保持"可微分"。
        x_t = dist.rsample()        # 這裡的 x_t 是 mu + sigma * epsilon, epsilon 是從標準正態分布中採樣的雜訊。
        action = torch.tanh(x_t)    # tanh(mu + sigma * epsilon), tanh的使用只是為了把動作限制在 [-1, 1] 的範圍內.
        
        # Log-probability 修正 (Jacobian correction for tanh)
        # 這個公式來自於 變數變換（Change of Variables）的機率論原理.
        # 對一個隨機變數做非線性轉換時，它的機率密度函數（PDF）會因為空間的擠壓或拉伸而改變。
        # 假設我們有一個隨機變數 x，其機率密度為 p(x)。現在我們定義一個新的隨機變數 a = f(x)（在這裡 f 是 tanh）。
        # 根據機率密度變換公式，新變數 a 的機率密度 p(a) 必須滿足 p(a) da = p(x) dx。
        # 因此，我們需要計算 da/dx 的絕對值（也就是Jacobian行列式的絕對值）來修正 log-probability。
        # log p(a) = log p(x) - log | f'(x) |, 而 f'(x) = 1 - tanh^2(x) = 1 - a^2, 
        # 因此 log | f'(x) | = log(1 - a^2 + 1e-6) (加上小常數避免除以零)
        # 這個修正確保了我們在計算 log-probability 時考慮了 tanh 函數對機率密度的影響，讓 Actor 能夠正確地學習到動作分布。
        # 原始的 dist.log_prob(x_t)，是在 x_t 空間（高斯分佈）的"機率密度", x_t的範圍是+/-infini,
        # 但由於我們實際上使用的是 action = tanh(x_t)，在靠近+/-1的地方機率密度比起原本的高斯分佈會因為擠壓而變高, 
        # 所以減去 log(1 - action^2) 來修正這個機率密度分布, 來讓在+/-1範圍內的action其機率密度接近x_t在+/-1範圍內, 高斯分布時的機率密度。
        # 如果不做這個修正，Agent 會誤以為在接近 +/-1 的action點 "機率密度"非常高, 學習時action會變得比較偏向這兩個邊界點。
        # 這會導致梯度計算錯誤，讓 Actor 變得很偏激，只敢往邊界噴火，最後導致訓練崩潰。
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(1, keepdim=True)

# 為何要有兩個一模一樣的Critic ? 解決高估偏差（Overestimation Bias）
#
# 在 Q-Learning 的過程中，我們不斷地在更新 Q 值，讓它去逼近未來的獎勵。
# 但在更新公式中，我們通常會取「未來狀態中最好的動作」的 Q 值作為目標 (Bellman Equation):
#   new_Q(s, a) = 
#     old_Q(s,a) 
#     + learning_rate * { 
#       actual_reward_from_env(s,a) 
#       + discount_factor * max[ old_Q(next_s, next_a) for all possible next_a ] 
#       - old_Q(s,a)
#     }
# 或者, 簡化成
#   Target ~ Reward + gamma * max[ Q(next_s, next_a) for all possible next_a ]
#
# 但問題來了：神經網路是不完美的，它總是有誤差。想像某個狀態下，所有動作的真實價值都是 0 分。
# 但因為網路的隨機誤差，它可能會覺得：動作 A 是 0.02, 動作 B 是 -0.01, 動作 C 是 0.05.
# 當使用 max 操作時，網路會永遠盯著那個「因為誤差而偏高」的 0.05。
# 這導致 Agent 會像一個過度樂觀的賭徒，它並不是真的找到了好動作，只是因為它對誤差「信以為真」，不斷累積這些正向誤差，最後導致 Q 值爆炸。
#
# 為了降溫這種過度樂觀的情緒，SAC 引入了兩個一模一樣的 Critic，讓它們"獨立"初始化並獨立訓練。
# 雖然它們的目標一樣，但因為初始權重不同，它們產生的隨機誤差方向也會不同。
# 更新邏輯：當我們要計算目標值（Target）時，我們同時詢問兩個 Critic，然後取"最小值"：(兩者都輸出"下一個state"採取action的"最大Q值")
#   Target = Reward + gamma * min( Q_1(s', a'), Q_2(s', a'))
# 這種做法被稱為 Clipped Double-Q。它確保了 Agent 看到的 Q 值是「保守且穩健」的估計。
#
# SAC 的 Critic 同時看 state 和 action。這就是為什麼 SAC 能進行 Off-policy 訓練的原因.
# 它可以評估「過去隨機採樣到的動作」在「現在的眼光」看來值多少分。
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2 (Twin Critic)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    # 注意 Critic 的輸入是 state 和 action 的串接，輸出是兩個 Q 值 (Q1 和 Q2)，用於減少過度估計的問題。
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)  
        return self.q1(sa), self.q2(sa)

# --- 3. SAC Agent ---
class SACAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.gamma = 0.99
        self.tau = 0.005 # Soft update 係數
        
        self.actor = Actor(state_dim, action_dim).to(device)
        # 兩個 Critic (critic, critic_target). 每個Critic 有兩組Q輸出.
        # critic 是用來訓練的網路，critic_target 是用來計算 Target Q 的網路。
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)

        # 自動調整 Alpha (溫度參數)
        #
        # 為什麼 self.target_entropy 被初始成 -action_dim ? 
        # 如果機率<1, entropy又是 - log_prob, 不應該是正值嗎? 為何會是負(-action_dim)的?
        #
        # 機率密度（Density）不等於 機率（Probability）
        # 在離散空間（例如玩機台按 A 或 B），機率 p 永遠介於 0 到 1 之間，所以 -log(p) 永遠是正值。
        # 但在連續空間中，我們看的是機率密度函數 (PDF) f(x), 密度可以大於 1.
        # Entropy H(X) = - integration[ f(x)*log f(x) ]dx。
        # 想像一個標準差 sigma 極小的高斯分佈, 中心點的高度（密度）會飆升到遠大於 1。
        # 當 f(x) > 1 時, log(f(x)) 就會變成正數, Entropy 就會是負值。
        #
        # 目標要設為 -action_dim 是 SAC 論文作者 Haarnoja 提出的一個非常強大的經驗法則（Heuristic）.
        # 每個維度我們都希望它維持一定的「確定性」.
        # 在數學上，一個 d 維高斯分佈的微分熵公式為：H(X) = d/2*(1 + ln(2*pi*sigma^2))
        # 當 sigma 變小時，這個值會一直往下降，並穿過零變成負數。
        # 作者發現，將目標熵設定為 動作空間維度的負值（-d），在大多數環境中都能讓 Agent 
        # 在「保持探索」與「穩定降落」之間取得完美的平衡。
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=3e-4)

    # 自動熵調整 (Automatic Entropy Tuning)
    # 
    # 這解決了之前在 PPO 實驗中糾結的「熵權重到底要 0.001 還是 0.01」的問題。
    # 現在, self.log_alpha 是一個可學習的參數。
    # 運作邏輯：
    # 如果目前的策略「太死板」（Entropy 太低），alpha_loss 會讓 alpha 變大，強迫 Agent 增加探索。
    # 如果已經「夠亂了」，alpha 就會變小，讓 Agent 專心優化獎勵。
    #
    # target_entropy：通常設為 -action_dim（在 Lunar Lander 是 -2），這是我們希望 Agent 保持的最基本好奇心水準。

    # @property: 這是一個 Python 的裝飾器，讓我們可以像訪問屬性一樣訪問 alpha，而不需要每次都寫 self.log_alpha.exp()。
    #   如果沒有 @property，每次想拿到目前的 alpha 值就必須寫成 agent.alpha()（像調用函數一樣）。
    #   使用了 @property 後，就可以像讀取普通變數一樣寫成 agent.alpha。
    #
    # 目的：透過 @property 來把 exp() 這個轉換過程"隱藏"起來。
    #   在優化神經網路時，我們通常會優化 log_alpha 而不是直接優化 alpha。這是因為 alpha 必須永遠是"正數".
    #   如果直接優化 alpha，優化器可能會把它算出負值導致訓練崩潰。
    #   優化 log_alpha 後再取 exp()，能確保結果永遠是正數。
    #   外部代碼（如更新 Actor 的地方）只需要知道它在用 alpha。
    #   內部邏輯（計算過程）會自動幫你把存儲的 log_alpha 轉回 alpha。
    #
    # 這裡的 alpha 是溫度參數，控制著探索與利用的平衡。
    # 當 alpha 越大時，Agent 越傾向於探索（因為 log_prob 的影響被放大了），
    # 當 alpha 越小時，Agent 越傾向於利用（因為 log_prob 的影響被縮小了）。
    @property
    def alpha(self):
        return self.log_alpha.exp()

    # Target Q 計算：
    #   Q_target = Reward + gamma * [ min(Q_1, Q_2) - alpha * log(pi) ]
    #   這裡減去 alpha * log(pi) 就是在實踐「最大熵獎勵」：動作越不確定，獎勵越高。
    #   在 RL 中, theta 指的是神經網路的「權重與偏置」, 
    #   而 pi 指的是 Policy（策略）。它是一個函數，定義了 Agent 在特定狀態 s 下會採取什麼動作 a。
    #   pi(a|s) 指的是: 「在狀態 s 的條件下，採取動作 a 的機率或機率密度」。
    #   所以上述的公式中, log(pi) 就是 log(pi(a|s)), 基本上就是 log_prob.
    #
    # Actor 損失函數：
    #   actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
    #   Actor 的目標是：讓 Q 值極大化(用減的)，同時讓 Entropy（-log_prob）也極大化 (用減的)
    #   Entropy : -log_prob 是動作的不確定性，當 log_prob 越小（動作越不確定）時，-log_prob 就越大，這會給 Actor 更多的獎勵去保持探索。
    #
    # Soft Update (平滑更新)：
    #   我們不直接把 critic 複製給 target_critic，而是每次挪動一點點（tau = 0.005）。
    #   這讓目標值像「移動緩慢的終點線」，防止模型震盪。

    def update(self, buffer, batch_size):
        # 從 Replay Buffer 中隨機抽取一批經驗, 
        # output 為 s(state), a(action), r(reward), s_next(next_state), d(done) 的 batch (numpy array)。
        s, a, r, s_next, d = buffer.sample(batch_size)
        s = torch.FloatTensor(s).to(self.device)        # 將numpy array 轉換成 PyTorch 的 Tensor，並移動到 GPU（如果可用）。
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        # 計算target Q 時通知 torch 這段計算不須列入計算圖, 因為critic_target只是被用來預測要給critic訓練的目標.
        with torch.no_grad():  
            # 用 重參數化技巧 選取一個動作並回傳 log-probability。
            #
            # Target Q 計算：
            #   Q_target = Reward + gamma * [ min(Q_1, Q_2) - alpha * log_prob ]
            #   這裡"減"去 alpha * log_prob 就是在實踐「最大熵獎勵」：動作越不確定，獎勵越高。 alpha本身也是學習出來的.
            #
            # critic 是用來訓練的網路，critic_target 是用來計算 Target Q 的網路。
            # 注意在計算 Target Q 時，我們是使用 "critic_target" 來評估在"下一個狀態"採取所有可能動作所能獲得的最大 Q 值.
            a_next, log_prob_next = self.actor.sample(s_next)   # 形狀: (batch_size, action_dim), (batch_size, 1)
            q1_t, q2_t = self.critic_target(s_next, a_next)
            min_q_t = torch.min(q1_t, q2_t) - self.alpha * log_prob_next
            target_q = r + (1 - d) * self.gamma * min_q_t       # (1-d) : 只為尚未結束(terminated)的狀態計算未來獎勵。

        # 1. Update Critic : 訓練 critic 網路去逼近 critic_target。
        # 計算 MSE loss between critic_target 與 critic 的預測值, 用以更新 critic 網路的權重。
        # 這裡的 critic_loss 是兩個 MSE loss 的和，分別對應 Q1 和 Q2 的預測值與 target_q 之間的差距。
        curr_q1, curr_q2 = self.critic(s, a)
        critic_loss = F.mse_loss(curr_q1, target_q) + F.mse_loss(curr_q2, target_q) 
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 2. Update Actor
        # Actor 損失函數：
        #   actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
        #   Actor 的目標是：讓 Q 值極大化(用減的)，同時讓 Entropy（-log_prob）也極大化 (用減的)。
        #   Entropy : -log_prob 是動作的不確定性，當 log_prob 越小（動作越不確定）時，-log_prob 就越大，這會給 Actor 更多的獎勵去保持探索。

        # 在 Actor 損失函數中的這一項: torch.min(q1_new, q2_new) 是剛更新後的critic所預測採取行動後能獲得的q值.
        # 為什麼不用像在 PPO 中一樣減掉什麼類似"原本的值"的東西 ? (Value function的預測值, advantage = real_reward - V(s))
        #
        # 因為在 PPO 中 (Score Function Gradient), PPO 的梯度像是在進行「事後評論」。
        # 它不知道 Q 函數長什麼樣子(因為動作無法關連到actor的參數)，它只知道「剛才那個動作拿了幾分」。
        # 
        # 意思是, PPO 的 Actor 不知道「Q 值是如何對動作 a 做反應的」。
        # PPO只看見「動作 a -> 高分」。Actor 只能說：「喔，那以後多做 a」
        # 
        # 為什麼要減掉 V(s)？ 
        # 因為如果不減掉一個基準值，如果所有動作都是正分（例如 100, 101, 102），網路會試圖把"所有動作"的機率都調高，
        # 這會導致梯度很大且不穩定。減掉 V(s) 是為了知道哪個動作比「平均」更好。
        #
        # SAC (Pathwise Derivative Gradient)：SAC 透過重參數化，讓動作 a 變成 Actor 參數 theta 的連續函數。
        # 這時，Actor Loss 實際上是在做：「沿著 Critic 給出的 Q 值坡度往上爬」。.
        #
        # 意思是, SAC 的 actor 看見了「Q(s, a) 這個函數的斜率」。
        # Actor 說：「喔，Q 函數在 a 這個點的斜率是正的，那我要把 mu 向正方向推，這樣分數才會更高」。
        
        # 用原本的actor網路在當前狀態 s 上採樣一個動作 new_a，並計算它的 log-probability。
        new_a, log_prob = self.actor.sample(s)
        # 接著把這個 new_a 串接到剛剛更新後的 critic 網路中，得到兩個新的 Q 值 (q1_new, q2_new)，
        # 這代表在當前狀態 s 下採取 new_a 這個動作的價值。
        # mean() 是對整個 batch 的損失取平均，這樣 Actor 就會學習產生更高 Q 值的動作，同時保持足夠的探索性。
        q1_new, q2_new = self.critic(s, new_a)
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 3. Update Alpha
        # 在 PPO 中，我們必須手動測試 0.001 或 0.01 來決定混亂獎勵（Entropy Loss）的權重。
        # 而在 SAC 中，我們透過這個 alpha_loss 讓模型自己學會如何調整這個權重。
        # Entropy(熵) 就是 -log(prob).
        #
        # alpha_Loss = alpha * (Current Entropy - Target Entropy)
        #            = self.log_alpha * (-log_prob - self.target_entropy)
        #            = - self.log_alpha * (log_prob + self.target_entropy)
        #
        # 這個alpha_loss計算式只是一個讓優化器幫我們調整alpha的工具, 跟縮小current_entropy
        # 與target_entropy的距離沒關係. 
        #
        # detach() 很重要的原因是它讓 log_prob + self.target_entropy 在計算圖中被視為常數, 
        # 這樣在優化 log_alpha 時，梯度不會傳回 log_prob + self.target_entropy。(梯度到此變成0)
        # 所以這個loss function中的唯一可優化變數是 log_alpha, 
        # 而這使得梯度 d(alpha_loss)/d(log_alpha) = - (log_prob + self.target_entropy).
        # 而優化器的動作是對"變數"值做"負梯度方向"的調整, 所以當探索不足時, current entropy 
        # 小於 target_entropy, 梯度就會是"負"的。優化器就會調"高"log_alpha, 達到我們的"原始目的".
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # 4. Soft Update Target Networks
        # 之前在步驟1已經訓練 critic 網路去逼近 critic_target。
        # 現在我們要讓 critic_target 慢慢地跟上 critic 的更新，用來評估下一個run時所選取動作的q值。
        # 
        # 在步驟1之前, actor選好動作後, 是使用critic_target來評估可獲得的q值, 作為在步驟1 中 critic的訓練目標, 
        # 來訓練 critic 網路去逼近 critic_target。 
        # 但是在步驟4中, 又讓 critic_target 跟上 critic 的更新，用來評估下一個run時所選取動作的目標q值。
        # A學習B, 然後B往A移動, 不會怪怪的嗎?
        # 這在強化學習中被稱為 「自舉（Bootstrapping）」。
        # 如果只是 A 追 B、B 追 A，那 Q 值確實會陷入無限循環，最後可能一起漂移到正無窮大。
        # 但請注意 Target Q 的公式：Q_target = r + gamma * (min(Q_t1, Q_t2) - alpha * log_prob).
        # 關鍵就在那個 r (即時獎勵)。 r 是來自環境的真實回饋（Ground Truth），它是不可被神經網路操縱的「硬指標」。
        # 雖然我們是用「未來的預測」來補足目前的評估，但每一輪更新都會被真實獎勵 r 往「現實」拉回一點點。
        # 如果用同一個 Critic 計算 Target，那麼每更新一次參數，「終點線（Target）」也會跟著動。
        # 這會導致正回饋震盪：如果網路不小心高估了某個動作，它會立刻把這個高估的值當作下一次的目標，導致錯誤被無限放大。
        # Soft Update：讓目標變成「慢動作」.
        # Critic (A)：正在瘋狂跑步，每一步都在根據最新的經驗調整方向。
        # Critic Target (B)：像是一個非常有耐心的導師，它只會吸收 A 表現出的「長期趨勢」，而忽略掉 A 瞬時的震盪。
        # Target (B)的行為邏輯是緩慢地、平滑地被 A 同步。它提供一個穩定的參考基準，防止 Q 值爆炸。
        #
        # 這個公式 tau * current + (1 - tau) * target, 就像是 數位信號處理中的 recursive filter, 
        # 當 tau 向我們的例子中的 0.005 時, 本身的行為就像是一個低通濾波器, 減緩高頻震盪(critic)造成的變動.
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            # Soft Update 的公式: target_param = tau * param + (1 - tau) * target_param
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return critic_loss.item(), actor_loss.item(), self.alpha.item()

# --- 4. Main Training Loop ---
if __name__ == "__main__":#
    env = gym.make("LunarLanderContinuous-v3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(8, 2, device)
    buffer = ReplayBuffer(1000000)
    writer = SummaryWriter(f'runs/SAC_Lander_{datetime.datetime.now().strftime("%H%M%S")}')
    
    MAX_EPISODES = 1000
    BATCH_SIZE = 256
    total_steps = 0

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        ep_reward = 0

        # 什麼前 1000 步要「純隨機」取樣？
        # 雖然 Actor Model 的重參數化採樣（Reparameterization Trick）也能提供隨機性，但在訓練初期，直接使用
        # env.action_space.sample() 有以下幾個不可替代的好處：
        # 真正的「冷啟動」全覆蓋：
        #   剛初始化的神經網路權重是隨機的，但這種隨機往往帶有「結構性偏差」。例如，經過 tanh 層後，動作可能傾向
        #   於擠在中間或兩端。而 env.action_space.sample() 通常是均勻分佈（Uniform Distribution），它能確保
        #   Agent 在一開始更公平地探索整個動作空間的所有角落。
        # 填充經驗回放池（Replay Buffer）：
        #   SAC 這種離策（Off-policy）演算法需要先有一堆「垃圾資料」作為基礎，Critic 才能開始學會分辨什麼是好、
        #   什麼是壞。如果前 1000 步就讓 Actor 決定，萬一 Actor 的隨機權重讓它一直往牆上撞，你的 Buffer 裡就
        #   全是「撞牆」的經驗。純隨機探索能幫你存進更豐富的 state-action 組合。
        # 避免「先入為主」的崩潰：
        #   如果在網路還沒對環境有任何概念時就開始優化，模型很容易因為幾次偶然的成功（例如剛好歪打正著降落）
        #   而迅速陷入局部最優解（Local Optimum）。這 1000 步就像是「先看地圖」，而不是「邊跑邊想」。

        # 每一輪最多走500步.
        for t in range(500):
            total_steps += 1
            if total_steps < 1000: # 前期隨機探索
                action = env.action_space.sample()      # 均勻分佈（Uniform Distribution）
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                action, _ = agent.actor.sample(state_t) # 透過 actor model 進行 重參數化 取樣
                # .detach() : 切斷計算圖. 這個動作是要給環境的，不需要算梯度。
                # .cpu()	: 搬移裝置.	Actor 的輸出通常在 GPU（CUDA）上，但 env.step() 函數是在 CPU 上跑的。
                # .numpy()	: 轉換格式. PyTorch 的 Tensor 環境不認得。env.step() 預期接收的是 NumPy 的 ndarray。
                # [0]	    : 移除 Batch 維度. action 形狀是 (1, action_dim)。[0] 將 action 還原成 (action_dim,)。
                action = action.detach().cpu().numpy()[0]

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            # done 有兩種情況 : terminated (term) 或 truncated (trunc).
            # Terminated (終止)：
            #   Agent 達到了「終止狀態」。例如：飛船墜毀了，或是成功降落在目標點。
            #   數學意義：這個狀態之後真的沒有任何未來獎勵了。V(s_terminal) = 0。
            # Truncated (截斷)：
            #   Agent 還在跑，但「時間到了」。例如：你限制 Lunar Lander 只能飛 500 步，第 501 步時它還在空中飄，但環境強制停止了。
            #   數學意義：這個狀態之後其實還有未來獎勵，只是我們不讓它跑了。飛船在第 501 步的價值（Value）不應該是 0。
            #
            # 在 update() 裡計算 target_q 時：
            #   target_q = r + (1 - d) * gamma * next_q
            # 如果我們在 Truncated（時間到）時把 d 設為 1，公式會變成 target_q = r + 0。
            # 這會誤導模型以為「在空中飄」這個狀態的未來價值是 0，導致模型學會「在時間快到時自殺」，
            # 因為它誤以為在那之後沒有任何好處。
            # 因此，我們只在 Terminated（真正結束）時才讓 d=1。
            
            # ------------------------------------------------------------
            # 實測過後, 尚未apply 任何的 reward shaping, 效果就已經相當的不錯, lander都能很順利地停在範圍內, 
            # 唯一還需要改進的點只剩下著陸之後不要再噴火就可以了.
            #
            # 可加入 Reward Shaping 邏輯 here.
            # ------------------------------------------------------------
            
            # 注意這裡是 term, 不是 done. 
            # 因為 SAC 是離策演算法, 我們需要知道 episode 是因為「終止條件」結束的還是「截斷條件」結束的。
            buffer.push(state, action, reward, next_state, term) 
            state = next_state
            ep_reward += reward

            if len(buffer) > BATCH_SIZE and total_steps % 1 == 0:
                c_loss, a_loss, alpha = agent.update(buffer, BATCH_SIZE)
                writer.add_scalar("Loss/Critic", c_loss, total_steps)
                writer.add_scalar("Loss/Actor", a_loss, total_steps)
                writer.add_scalar("Alpha", alpha, total_steps)

            if done: break
        
        writer.add_scalar("Reward/Episode", ep_reward, episode)
        print(f"Episode: {episode}, Reward: {ep_reward:.2f}")

        # 每 50 輪保存一次模型權重。
        if episode % 50 == 0:
            os.makedirs("dqn_model", exist_ok=True)
            torch.save(agent.actor.state_dict(), "dqn_model/sac_actor.pth")