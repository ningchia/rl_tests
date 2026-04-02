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

#--------------------------------------------------------------------------------
# 這裡我們實作 PPO，因為它在連續控制任務中表現穩定，且相對容易調整超參數。
# 基本上 PPO 的 model 輸出由3個head組成, 分成兩組 :
#
#   Actor : 由兩個head組成輸出一個機率曲線,  代表 action 設定值範圍內的"偏好機率分布".
#           head0 輸出 mu (平均值), head1 輸出 sigma(標準差)
#           注意 mu 與 sigma 的維度就是"action 設定值"的維度. 
#           以這個例子為例, action有2組 (左右噴, 往下噴), 所以mu與sigma其實定義了兩條 機率分布曲線,
#           分別代表 左右噴 與 往下噴 引擎的設定值範圍. 
#           在選擇動作時, 因為每個動作的大小"程度"是連續的值, 就用這個"偏好機率分布"取樣一個動作設定值. 
#           而之前在DQN裡, 當取樣的random值小於epsilon就隨機選一個動作, 在PPO裡則相當於sigma的意思, 
#           sigma 大就代表有比較高的機率選到非平均值的設定值, 代表可能嘗試的"非預期最好"的動作設定值範圍比較大的意思, 
#           而 mu 就相當於目前最prefer的, "預期效果最好"的動作程度設定值的意思.
#
#  Critic : 由一個head組成, 輸出一個數值, 代表"當前狀態"一直到終局, 最有可能的平均狀態價值 (V-value).
#           精確地說, 是
#
#               "從現在這個狀態開始，一直到這局結束，預期能拿到的所有 Reward 總和（打折後的累計值）"
#
#           Critic 在「預測」時雖然不看 Action，但它「學習」的對象卻深深受到 Actor 的影響。
#           它是狀態價值 (V)，不是動作價值 (Q).
#           相對於 DQN 的輸出是 Q(s, a)：它預測「在狀態 s 下，做動作 a 有幾分」。所以它必須考慮 Action。
#           而 PPO Critic 的輸出是 V(s)：它預測「處於狀態 s 下，平均而言最後能拿幾分」。
#           這就是為什麼 Critic 的輸入層只有 state_dim，而不像有些演算法會把 action 也餵進去。
#           它就像一個看盤的分析師，只要看一眼現在的盤勢（飛船的高度、速度、角度），不論你打算怎麼噴火，
#           它心中就會有一個「這局大概能拿幾分」的底價。 
#           其實，Critic 預測的是：「在目前這個 Actor 的水平下，這個狀態值多少錢。」
#           如果 Actor 很爛：只要稍微歪掉就會墜毀。這時 Critic 看到「傾斜 10 度」的狀態，就會給出很低的分數.
#           如果 Actor 是高手：就算歪掉也能救回來。這時 Critic 看到同樣「傾斜 10 度」的狀態，會給出較高的分數.

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

ENV_NAME = "LunarLanderContinuous-v3"
MAX_EPISODES = 1000
UPDATE_TIMESTEP = 4000 # 每累積 4000 步更新一次

MODEL_SAVE_PATH = "dqn_model"
CHECKPOINT_FILE = "ppo_lunar_lander.pth"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)


# 實驗發現若是讓 Actor 與 Critic 共用同一個特徵提取層 (backbone), 會導致訓練過程中 Critic 的預測能力變得非常差, 
# 以至於 Actor 的調整方向完全錯誤, 最終導致整個訓練過程崩潰.
# Critic 的巨大loss (value_loss)造成梯度更新時淹沒 Actor 的loss (policy_loss) 梯度, 讓 Actor 的輸出變得非常不穩定, 
# 以至於 Actor 的調整方向完全錯誤, 即使讓 value_loss的計算權重調小 0.5 -> 0.02, 也無法完全解決這個問題, 
# 會發現 Sigma_Main被clamp在上限. 因此需要讓 Actor 與 Critic 使用完全獨立的特徵提取層 (backbone).
'''
# --- PPO 網路架構 (Actor-Critic 分享 backbone) ---
class PPOModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOModel, self).__init__()
        # 共同特徵層
        self.fc = nn.Sequential(                        # 輸入形狀: [batch_size, state_dim]
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
        self.actor_mu = nn.Linear(256, action_dim)      # head0: 輸出動作設定值的均值 (mu), 形狀: [batch_size, action_dim]
        # nn.init.constant_(self.actor_mu.bias, 0.5)      # 初始 bias 為 0.5 來強迫初期 mu 偏向正值（噴火）

        self.actor_sigma = nn.Linear(256, action_dim)   # head1: 輸出動作設定值的標準差 (sigma), 形狀: [batch_size, action_dim]
        
        # Critic: 輸出預期從目前狀態到終局的累計(打折後的)狀態價值 (V-value)
        self.critic = nn.Linear(256, 1)                 # head2: 輸出預期到終局能獲得的,與動作無關的狀態價值 (value), 形狀: [batch_size, 1]

    def forward(self, state):
        x = self.fc(state)
        mu = torch.tanh(self.actor_mu(x))               # 將head0的輸出(mu)限制在 [-1, 1]

        # sigma = F.softplus(self.actor_sigma(x)) + 1e-5  # 將head1的輸出(sigma)限制在 [0, ∞)

        # 以下這個做法發生過訓練過程中 sigma 為 NaN 的情況. 意思是某些權重在更新後變得NaN, 以至於 sigma 的輸出也變成NaN.
        # sigma = torch.exp(self.actor_sigma(x) - 0.5) + 1e-5  # 將head1的輸出(sigma) = e^(x - 0.5) 讓初始 sigma 大約在 e^-0.5, 0.6 左右
        
        # 將 log_sigma 限制在 [exp(-2), exp(0)] -> [0.13, 1.0] 之間. 
        # log_sigma的clamp上限太小也不好, 像是e^-0.5 = 0.6, 觀察到 reward 還是負的.
        log_sigma = torch.clamp(self.actor_sigma(x), -2.0, 0)
        sigma = torch.exp(log_sigma)

        value = self.critic(x)

        return mu, sigma, value
'''
# --- PPO 網路架構 (Actor-Critic 分離 backbone) ---
class PPOModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOModel, self).__init__()
        # --- Actor 網路：專門負責輸出動作分布 ---
        self.actor_backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(256, action_dim)
        self.sigma_head = nn.Linear(256, action_dim)

        # --- Critic 網路：專門負責估計狀態價值 (Value) ---
        self.critic_backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # 直接輸出 Value
        )

    def forward(self, state):
        # 1. Actor 路徑
        a_x = self.actor_backbone(state)
        mu = torch.tanh(self.mu_head(a_x))
        # 這裡建議繼續用你的 clamp 邏輯或 softplus
        log_sigma = torch.clamp(self.sigma_head(a_x), -2.0, 0) 
        sigma = torch.exp(log_sigma)

        # 2. Critic 路徑 (完全獨立於 Actor 的特徵提取)
        value = self.critic_backbone(state)

        return mu, sigma, value

# --- PPO 核心演算法 ---
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用兩組網路：
        # 舊 policy 網路 (self.policy_old) : 在 "蒐集資料階段" 使用.
        #   前一個UPDATE_TIMESTEP週期(2000步)freeze的網路, 在"蒐集資料階段"用來選擇action設定值
        #   並記錄 logprobs, values 等資料以供後續對新的policy做訓練。
        # 新的 policy 網路 (self.policy) : 在 "update 新policy階段" 使用.
        #   在目前這個 UPDATE_TIMESTEP週期(2000步) 被調整的網路.
        #   使用新 policy 來算出新的 logprobs, values 等資料來跟舊的資料做比較. 根據這個loss 進行的調整與更新.
        #   更新的對象是 policy 的權重, 而不是 policy_old 的權重.
        
        self.policy = PPOModel(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = PPOModel(state_dim, action_dim).to(self.device)
        # 初始時，讓舊網路的權重與正在訓練的網路相同。
        self.policy_old.load_state_dict(self.policy.state_dict())
        # 初始化 均方誤差 (MSE) instance. 等一下 計算價值損失時會用到。
        self.mse_loss = nn.MSELoss()

    # 蒐集資料階段：根據舊 policy 網路 (self.policy_old) 的輸出來選擇動作，並記錄相關資料。
    def select_action(self, state, storage):
        state = torch.FloatTensor(state).to(self.device)    # 將當前狀態轉換為 Tensor，並移動到 GPU（如果可用）。
        with torch.no_grad():
            # 用舊 policy 網路 (self.policy_old) 輸出:
            #   1. 下一個action設定值的偏好機率分布
            #   2. 預測以目前的actor能力, 從目前狀態到終局預期能拿到的所有Reward(狀態價值V)的總和(打折後的累計值）.
            # 注意這邊的輸出(mu, sigma) 本身就是 長度為 2 的向量, value 是長度為 1 的向量.
            mu, sigma, value = self.policy_old(state)           # 形狀: [1, action_dim], [1, action_dim], [1, 1]
            
        # 根據 mu 和 sigma 定義一個map在action設定值範圍的偏好正態機率分佈 (Normal distribution)，
        # 然後從這個分佈中隨機取樣一個動作設定值 (action)。
        # 這裡的 action 是一個包含兩個數值的向量，分別代表主引擎和側邊引擎的推力大小。
        dist = Normal(mu, sigma)
        action = dist.sample()      # 以這個機率分布隨機取樣一個動作 (action)的"設定值"。
        # 將該設定值的 log 機率 (log probability) 記錄下來，這個等一下在計算優勢函數時會用到。
        # .sum(dim=-1) 是因為 action 是一個向量（包含兩個動作維度- 往下噴, 左右噴），我們需要把這兩個動作維度
        # 的 log 機率(在最後的維度上)加總起來，來得到整體動作的 log 機率。
        action_logprob = dist.log_prob(action).sum(dim=-1)          # 形狀: [1], 這邊的1是batch維度.
        
        # 將蒐集到的資料 (state, 選擇的下一個動作設定值, 該設定值的logprobs, 以及該state預期到終局的value) 
        # 都記錄到 Storage 中。
        storage.states.append(state)
        storage.actions.append(action)
        storage.logprobs.append(action_logprob)
        storage.values.append(value)
        # 將動作從 Tensor 轉換回 NumPy 陣列並限制在 [-1, 1] 確保符合 Gym 規範，以便在環境中使用。
        # .cpu() 是將 Tensor 從 GPU 移回 CPU，numpy() 是將 Tensor 轉換為 NumPy 陣列。
        act_out = action.cpu().numpy()
        # 主引擎 (act_out[0]) 在環境中 [0, 1] 噴火，負數不噴。
        # 側邊引擎 (act_out[1]) 在環境中 [-1, 1]。
        return np.clip(act_out, -1.0, 1.0) 

    # 訓練 policy 階段.
    # 訓練過程使用的loss分成幾個部分 : 
    #   1. Actor heads 使用 Policy Loss : 這是 PPO 的核心損失函數。
    #        ratios = pi_theta / pi_theta_old
    #        surr1 = ratios * advantages
    #        surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
    #        loss = -torch.min(surr1, surr2)
    #
    #      這個比例 ratios = torch.exp(logprobs - old_logprobs), 可以理解成 : 
    #
    #        這個動作設定值(之前蒐集到的紀錄中, 透過old policy採取的動作), 其 被map到新policy曲線上的機率
    #        與 被map到舊policy曲線上的機率 的比率, 就代表正在調整中的 "新policy" 相對於 "舊policy" 對同樣
    #        這個動作設定值的偏愛程度趨勢, 是更加偏愛(>1), 還是更不偏愛(<1). 
    #        
    #      然而, Actor對偏好機率分布的這個調整方向不一定是正確的, 有可能新policy更偏愛這個動作, 但是這個動作
    #      造成的最終累計reward 可能比起採用其他更不偏愛的動作的最終累計reward 還要低.
    #      所以policy loss就得要借助優勢函數的幫忙來決定調整的方向.
    #
    #      優勢函數 (Advantage) = 真實累計得分 - 預期累計得分, 即
    #        "真實環境提供的到終局的累計reward" - "新policy預測的, 目前狀態到終局的預期累計狀態價值"
    #
    #      具體來說, Actor 所使用的 Policy Loss 就是
    #                "這個機率比例 (新policy的"更"偏好程度趨勢)",
    #           乘上 ( "真實環境提供的到終局的累計reward" - "新policy預測的, 目前狀態到終局的預期累計狀態價值" )
    #      再取負值, 因為actor希望"最大化"這個超額reward.
    #
    #      ------ 訓練的原理(跟GAN有點像) ------
    #      Critic 的目標（模擬者）在於「模擬環境」。它想讓自己輸出的 Value 盡可能等於環境回傳的真實 Reward。
    #      它的 Loss 是 MSE，目的是為了讓預測越來越準。
    #      Critic 的理想是：預測準確率 100%，Advantage 永遠為 0。
    #
    #      而 Actor 的目標（決策者）不是要模擬環境，它是要「贏過環境」。
    #      它透過 Critic 提供的「基準線」，不斷尋找能拿到「超額分數（正 Advantage）」的動作。
    #      Actor 的理想是：永遠能做出讓 Critic 驚訝的動作，拿到比 Critic 預期更高的分數。
    #
    #      Critic（師父/模擬者）：不斷看環境的報表，學習「在這種處境下，通常只能拿 50 分」。它致力於模擬環境的難度。
    #
    #      Actor（徒弟/執行者）：它參考師父的評價。當它做了一個動作拿到 60 分（Advantage = +10），它會看 Ratio：
    #      如果新策略對這個動作的偏好增加了，它就覺得「喔！看來我這幾次的修正方向是對的，這動作真的有超額利潤！」
    #      於是，它會繼續把 mu（均值）往那個動作的方向挪。
    #
    #   2. Value Loss : 0.5 * MSELoss
    #      這是訓練 Critic（評論家）讓它預測得分越來越準的損失函數, 類似上面提到的優勢函數(Advantage), 
    #      但是是透過均方差帶進"絕對值"的概念. 它計算了 
    #         "Critic 所預測的到終局的累計狀態價值（target_values）" 與 "實際環境回報的, 到終局所獲得的總reward" 
    #      之間的均方誤差(MSE)。
    #
    #   3. 混亂獎勵 : -0.01 * dist_entropy
    #      這是 PPO 中的一個獎勵項，鼓勵策略保持一定的隨機性（即探索）。
    #      dist_entropy 是動作分佈的熵，而高斯分佈的熵正比於sigma. 熵越大表示動作越多樣化。
    #      而我們希望最大化熵，所以在損失函數中加上負號。
    
    def update(self, storage, timestep):
        # 計算回報 (Rewards-to-go)
        rewards = []
        discounted_reward = 0
        # is_terminal 是一個布林值，表示該步是否結束了這一局。
        # 當 is_terminal 為 True 時，表示這一步是該局的最後一步，後續的回報不應該再累積之前的獎勵，
        # 因此 discounted_reward 被重置為 0。
        #
        # 在強化學習中，我們通常會對未來的獎勵進行「打折」（discounting），這就是為什麼我們在計算回報時會使用 gamma 這個折扣因子。
        # 之所以不直接加總而要「打折」的原因: 數學收斂性、人類直覺、以及對抗風險。
        #
        #   數學上的必要性：防止「無窮大」. 
        #     如果這局永遠不結束且不打折：總回報會變成無限大, 就無法比較 動作A與動作B 哪個更好.
        #     加上折扣後：總回報變成了一個等比級數：1 + gamma + gamma^2 + ...。只要 gamma < 1，這個級數就會收斂到一個確定的數值.
        #
        #   人類直覺：我們通常更關心「眼前的獎勵」，而不是「遙遠未來的獎勵」。打折反映了這種時間上的偏好。
        #
        #   對抗風險：未來的不確定性更高。打折可以讓模型更專注於那些更確定的獎勵，減少對未來不確定事件的過度依賴。
        #
        # gamma 越接近 1：代表 Agent 越「高瞻遠矚」，會為了長遠利益忍受短期的痛苦（例如：為了最後降落準一點，現在多花點燃料）。
        # gamma 越接近 0：代表 Agent 越「短視近利」，只在乎下一秒能拿多少分。
        # 在 LunarLander 中，我們通常設為 0.99。這代表我們希望 Agent 既要關注當下穩定，也要為了最後的成功著陸做長遠打算。
        for reward, is_terminal in zip(reversed(storage.rewards), reversed(storage.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            # 這裡的 discounted_reward 是從最後一步開始往前累積的回報值。
            # 每一步的回報等於當前的 reward 加上之前累計回報的折扣值 (gamma * discounted_reward)。
            # 即在時間點 t 採取動作後，最終能拿到的累積總分是多少（即真實目標值）
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # --- 注意batch_size 匹配 ---
        # 注意 old_states, old_actions, old_logprobs, old_values, target_values, advantages 
        # 的 batch_size 都是 UPDATE_TIMESTEP = 2000

        # 轉換為 Tensor
        # torch.stack 是將列表中的 Tensor 沿著新的維度堆疊起來，形成一個新的 Tensor。
        # 這裡的 storage.states, storage.actions, storage.logprobs, storage.values 都是列表，每個元素都是一個 Tensor。
        # detach() 是將 Tensor 從計算圖中分離出來，這樣在更新過程中就不會對這些 Tensor 進行梯度計算。
        old_states = torch.stack(storage.states).detach()               # 形狀: [batch_size, state_dim]
        old_actions = torch.stack(storage.actions).detach()             # 形狀: [batch_size, action_dim]
        old_logprobs = torch.stack(storage.logprobs).detach()           # 形狀: [batch_size]
        
        # 這裡的 old_values 是 舊的policy所"預測"的, "到終局的累計狀態價值 (V-value)"。
        # 注意 storage.values 是一個列表，每個元素都是一個形狀是 [1] 的 Tensor（因為 critic 輸出的是一個scaler）。
        # squeeze() 是將 Tensor 中的單維度（即大小為 1 的維度）去掉，這樣 old_values 就會變成一維的 Tensor。
        # 也可以用類似的操作像是.view(-1) 來將 Tensor 重新塑形為一維。
        old_values = torch.stack(storage.values).detach().squeeze()     # 形狀: [batch_size]

        # Side info : "-1" 在 reshape() in numpy 與 view() in PyTorch 都代表「自動計算該維度的長度」。
        # reshape() in numpy:
        #    ex. 最常見的用法，將多維矩陣拉直成一行。 (單純展平)
        #        例如：如果你有一個形狀為 (2, 3) 的 Tensor，使用 reshape(-1) 就會得到一個形狀為 (6,) 的一維 Tensor。
        #    ex. 自動計算行數：reshape(-1, n), 或者自動計算列數：reshape(n, -1)
        #        例如：形狀為 (20,) 的 Tensor，使用 reshape(-1, 5) 或 reshape(4, -1) 就會得到一個形狀為 (4, 5) 的二維 Tensor。 
        # view() in torch :
        #    ex. x = torch.randn(2, 3, 4)  # 總共有 2*3*4 = 24 個元素
        #        a = x.view(-1)            # 形狀變為 torch.Size([24])
        #        b = x.view(6, -1)         # 形狀變為 torch.Size([6, 4])，因為 24/6 = 4
        #        c = x.view(-1, 1)         # 形狀變為 torch.Size([24, 1])
        #    view() 的使用限制
        #        數據連續性：.view() 只能用於「連續的」（contiguous）張量。
        #        如果張量經過 transpose() 或 permute() 等操作，必須先呼叫 .contiguous() 才能使用 .view()，否則會報錯。

        # 這裡的 target_values 是 PPO 更新過程中的「真實累計得分」(來自environment的reward, 不是policy預測出來的)，
        target_values = torch.FloatTensor(rewards).to(self.device)      # 形狀: [batch_size]
        
        # 優勢函數 (Advantage). 優勢 (A) = 真實累計得分 - 預期累計得分。
        advantages = target_values - old_values                         # 形狀: [batch_size]
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)   # normalize advantages

        # 用來記錄這一輪 10 個 Epoch 的平均數值
        sum_loss = 0
        sum_policy_loss = 0
        sum_value_loss = 0
        sum_entropy = 0
        sum_advantages = 0
        loss_calc_count = 0

        # 將數據打亂並分成 mini-batches
        batch_size = 64
        indices = np.arange(UPDATE_TIMESTEP)  # UPDATE_TIMESTEP 是這一輪收集的總步數，等於 old_states 的第一維度大小。

        for _ in range(self.K_epochs):      # K_epochs 是每次 PPO 更新的迭代次數，通常是 10。

            np.random.shuffle(indices)
            for start in range(0, UPDATE_TIMESTEP, batch_size):
                end = start + batch_size
                idx = indices[start:end]

                loss_calc_count += 1
                
                # --- 注意batch_size 匹配 ---
                # 注意這邊 mu, sigma, values, dist, logprob, dist_entropy, ratios, surr1, surr2, 
                # policy_loss, value_loss, entropy_loss, loss 的 batch_size 都是 64.

                # 只取一部分樣本更新，能讓梯度更具方向性
                mu, sigma, values = self.policy(old_states[idx])    

                # 在每次更新迭代中，使用正在訓練的 policy 網路 (self.policy) 計算當前狀態下的動作分佈 (mu, sigma) 
                # 和狀態價值 (values)。
                # 注意這邊 dist 的 batch_size = 64.
                dist = Normal(mu, sigma)
                # 我們的動作有兩個維度(主引擎、側引擎). sum(dim=-1) 是把 log機率 加總起來，得到整個「決策」的總 log機率。 
                # 注意這邊不能直接用old_actions, 得用 old_actions[idx], batch_size 才會match = 64.
                logprobs = dist.log_prob(old_actions[idx]).sum(dim=-1)      # 形狀: [batch_size, action_dim] -> [batch_size]
                # 每個維度都有自己的 sigma，也就有自己的熵。把這兩個動作維度的不確定性加總起來，得到這整個「決策」的總不確定性。
                dist_entropy = dist.entropy().sum(dim=-1)                   # 形狀: [batch_size, action_dim] -> [batch_size]
                
                # PPO 核心：計算 Ratio (pi_theta / pi_theta_old), 代表新policy對此action設定值的偏好程度"趨勢".
                # 注意這邊不能直接用old_logprobs, 得用 old_logprobs[idx], batch_size 才會match = 64.
                ratios = torch.exp(logprobs - old_logprobs[idx])            # 形狀: [batch_size]

                # ratios 是 tensor. 
                # torch.isnan(ratios) 的輸出是 同樣形狀的布林 tensor。如果沒有.any()，那麼即使有 NaN 也不會被檢查出來。
                # .any() 是用來檢查 Tensor 中是否存在任何元素為 NaN（Not a Number）。如果 ratios 中有任何一個元素是 NaN，這個條件就會返回 True。
                if torch.isnan(ratios).any():       
                    print(f"發現 ratios 含有 NaN Loss。\n idx: {idx} \n logprobs: {logprobs}\n old_logprobs[idx]: {old_logprobs[idx]}")
                    exit()

                # Clipped Surrogate Objective
                surr1 = ratios * advantages[idx]                            # 形狀: [batch_size]
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages[idx]
                # PPO 的損失函數包含：
                #   Policy Loss = -torch.min(surr1, surr2) ：加負號是因為我們要「最大化」優點。
                #   Value Loss = 0.5 * MSELoss             ：這是訓練 Critic（評論家）讓它預測得分越來越準。
                #   混亂獎勵 = -0.01 * Entropy              ：獎勵那些 sigma 比較大的行為，防止 AI 太快變死板。
                # PPO的訓練目標: 
                #   Advantage 慢慢趨近於 0，但 Reward 卻很高.
                #   那就代表 Critic 已經變成了「預言家」，而 Actor 已經變成了「頂尖飛行員」。
                policy_loss = -torch.min(surr1, surr2).mean()                           # 這邊的mean是在batch維度上.
                value_loss = 0.5 * self.mse_loss(values.squeeze(), target_values[idx])  # MSELoss 會自動在batch維度上取 mean.
                
                # 在entropy_loss 的權重為0.01 時雖然可以讓 Action_Main與Mu_Main絕大部分時間在正值, 但是 Sigma_Main卻卡在1. 
                # 而當權重為0.001時 , 效果就像權重為0的狀況, Sigma_Main有降低的現象, 只是因為有混亂獎勵的原因有一個明顯的上升後再下降.
                # 這才是我們想要的探索（Exploration）轉利用（Exploitation）」 過程：
                # 上升期（探索）：一開始模型對環境不熟，Entropy Loss 權重發揮作用，強迫模型調高 sigma（增加隨機性）。
                # 下降期（收斂）：隨著訓練進行，模型發現了能拿到高分的穩定策略（Advantage 訊號變強）。
                # 當 Policy Loss 的引導力量大於 0.001 的 Entropy 壓力時，模型開始壓低 sigma，進入精確控制階段。
                # 為什麼這樣比 0 權重好？權重為 0 時，模型完全依賴初始隨機性，一旦剛好抓到一個「還不錯」的策略（例如一直往左噴），
                # 它可能就會迅速收斂，而錯失了找到「完美策略」的機會。0.001 提供了一個安全網，防止模型過早陷入局部最佳解（Local Optimum）。
                entropy_loss = -0.001 * dist_entropy.mean()                             # 這邊的mean是在batch維度上.
                #entropy_loss = 0 * dist_entropy.mean()                              # 這邊的mean是在batch維度上.

                # 總損失 (取mean過的)
                loss = policy_loss + value_loss + entropy_loss            

                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪 (Gradient Clipping) 是一種常用的技巧，用於防止梯度爆炸（Gradient Explosion）問題.
                # 0.5 是裁剪的閾值，表示如果梯度的 L2 範數超過 0.5，就會被縮放回 0.5。
                # self.policy.parameters() 是要被裁剪的參數集合，通常是模型的權重。
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)     
                self.optimizer.step()

                # 累加數值
                sum_loss += loss.item()
                sum_policy_loss += policy_loss.item()
                sum_value_loss += value_loss.item()
                sum_entropy += dist_entropy.mean().item()           
                sum_advantages += advantages[idx].mean().item() 

        # 記錄 Actor 的輸出統計
        writer.add_scalar('Stats/Mu_Main', mu[:, 0].mean().item(), timestep)   # 主引擎均值
        writer.add_scalar('Stats/Sigma_Main', sigma[:, 0].mean().item(), timestep) # 主引擎標準差
        # 記錄這批次實際採樣出的動作大小
        writer.add_scalar('Stats/Action_Main_Mean', old_actions[:, 0].mean().item(), timestep)

        # 學習結束後，把新學好的權重同步給 policy_old，作為下一批收集資料時的參考基準。
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 清空 Storage：PPO 是 On-Policy，舊資料學完就不能再用了（因為策略已經變了），必須清空重抓。
        storage.clear()

        # 傳回平均loss.
        return sum_loss/loss_calc_count, sum_policy_loss/loss_calc_count, \
            sum_value_loss/loss_calc_count, sum_entropy/loss_calc_count, sum_advantages/loss_calc_count

class Storage:
    def __init__(self):
        self.states, self.actions, self.logprobs, self.rewards, self.is_terminals, self.values = [], [], [], [], [], []
    def clear(self):
        del self.states[:], self.actions[:], self.logprobs[:], self.rewards[:], self.is_terminals[:], self.values[:]

# --- 訓練主程式 ---
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = PPOAgent(state_dim, action_dim)
storage = Storage()
writer = SummaryWriter(f'runs/PPO_Lander_{datetime.datetime.now().strftime("%H%M%S")}')

# 跑 1000局, 每局最多500步, 每累積4000步就更新一次 PPO 模型.
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

        # x : 水平位置, range : -2.5 ... 2.5
        # height : 垂直高度, range : -2.5 ... 2.5
        # vel_x : 水平速度, range : -10 ... 10
        # vel_y : 垂直速度, range : -10 ... 10
        # angle : 角度, range : -6.2831855 ... 6.2831855 (360度的弧度表示)
        # angular_velocity : 角速度, range : -10 ... 10
        # leg_left_contact : 左腳接觸地面
        # leg_right_contact : 右腳接觸地面
        x, height, vel_x, vel_y, angle, angular_velocity, leg_left_contact, leg_right_contact = state

        episode_reward += reward            # 先記錄實際得分，再進行獎勵塑造 (Reward Shaping)。
        storage.is_terminals.append(done)   # 記錄這一步是否結束了這一局，這對於計算回報 (Rewards-to-go) 非常重要。

        # -------------------------------------------------------------
        # 獎勵塑造：
        if vel_y < 0:
            if vel_y > -0.5:
                reward += 0.05*(1. + vel_y)
            else:
                reward -= 0.2*abs(vel_y)
        if vel_y > 0 and action[0] > 0:
            reward -= 0.01 * action[0]

        dist_from_center = abs(x)
        if height > 0:
            reward -= dist_from_center * 0.001    # x range : -2.5 ... 2.5

        # 一旦腳部接觸地面，我們應該強烈要求它靜止。
        if leg_left_contact > 0 or leg_right_contact > 0:
            if action[0] != 0 or action[1] != 0:
                reward -= 0.02
        # -------------------------------------------------------------

        storage.rewards.append(reward)      # 要提供給訓練的是reward shaping後的獎勵
        
        # 每累積 UPDATE_TIMESTEP (4000) 步就更新一次 PPO 模型。
        if timestep % UPDATE_TIMESTEP == 0:
            total_l, pol_l, val_l, ent, adv = agent.update(storage, timestep)

            # 以 timestep 作為 X 軸，能精確對應訓練進度
            writer.add_scalar('Loss/Total', total_l, timestep)
            writer.add_scalar('Loss/Policy', pol_l, timestep)
            writer.add_scalar('Loss/Value', val_l, timestep)
            writer.add_scalar('Loss/Entropy', ent, timestep)            
            writer.add_histogram('Train/Advantages', adv, timestep)
            
        if done: break
    
    # 每局結束紀錄 reward.
    writer.add_scalar('Reward', episode_reward, episode)
    # 每 10 局輸出一次當前的獎勵，並且將模型權重保存到指定的檔案中。
    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")
        torch.save(agent.policy.state_dict(), checkpoint_path)