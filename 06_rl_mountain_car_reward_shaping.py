# -----------------------------------------------------------------------------------------------------
# 什麼是稀疏獎勵 (Sparse Reward)？
# -----------------------------------------------------------------------------------------------------
#
# 在 CartPole 中，你每撐過 1 秒就能得到 +1 分。這叫 「密集獎勵 (Dense Reward)」——AI 每走一步都知道自己做得好不好。
# 但在 MountainCar 中，環境的設定非常殘酷：
#   規則：小車被困在山谷，動力不足以直接衝上右側山頂。它必須先往左後方退，利用重力位能來衝上山。
#   獎勵：只要沒到山頂，每一步都是 -1 分。只有抵達旗子處，遊戲才結束。
#   稀疏點：AI 在一開始根本不知道「後退」能幫助前進。它會在山谷底下亂晃，拿到一堆 -1 分，直到它「碰巧」晃到山頂。
#          在還沒摸到旗子前，所有的動作看起來都一樣爛。
# 這就是「稀疏獎勵」的問題：AI 在學習初期幾乎沒有任何正向回饋，難以找到正確的行為路徑。
#
# -----------------------------------------------------------------------------------------------------
# 什麼是獎勵塑造 (Reward Shaping)？
# -----------------------------------------------------------------------------------------------------
#
# 既然環境給的獎勵太稀疏，AI 學不動，那我們就「人為地」給它一些提示，引導它走向目標。這就是 Reward Shaping。
# 我們不再只是依賴環境給的 -1，而是根據小車的位置或速度額外給分。
# 常見的 Shaping 策略：
#   高度優先 (Height-based)：小車爬得越高，我們給它越多額外獎勵。
#   速度優先 (Velocity-based)：小車速度越快，我們給它獎勵（鼓勵它衝刺）。
#   能量優先 (Potential-based)：計算小車的動能與位能。

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

# --- Dueling DQN 架構 ---
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

# --- 超參數調整 ---
ENV_NAME = "MountainCar-v0"
GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 20000         # MountainCar 需要較大的記憶池
BATCH_SIZE = 128            # 增加批次大小以穩定訓練
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.998       # 稍微放慢探索衰減速度
TARGET_UPDATE = 10
EPISODES = 2000             # MountainCar 需要更多的訓練回合才能學會爬山
REWARD_SHAPING_METHOD = "exp_potential_velocity_progress"  # 選項 "potential_height_progress", "exp_potential_velocity_progress".

MODEL_SAVE_PATH = "dqn_model"
CHECKPOINT_FILE = "best_mountaincar_model.pth"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)

#---------------------------------------------------------------------------------
# 環境 "MountainCar-v0"
# ref: https://gymnasium.farama.org/environments/classic_control/mountain_car/
#---------------------------------------------------------------------------------
# Observation(state) space: 
#   a ndarray with shape (2,) representing the position (-1.2 ~ 0.6) and velocity (-0.07 ~ 0.07) of the car.
#   position of the car is assigned a uniform random value in [-0.6 , -0.4].
# Action space: 
#   discrete with 3 actions (0: Accelerate to the left, 1: Don’t accelerate, 2: Accelerate to the right)
# Reward:
#   -1 for each step until the goal is reached. 
#   The episode ends when the position of the car reaches 0.5 (the flag) or after 200 steps. 
#   The maximum score is -1 (if it reaches the flag in the first step).
#   The minimum score is -200 (if it never reaches the flag) 
# Transition Dynamics:
#   Given an action, the mountain car follows the following transition dynamics:
#     velocity(t+1) = velocity(t) + (action - 1) * force - cos(3 * position(t)) * gravity
#     position(t+1) = position(t) + velocity(t+1)    
#   where force = 0.001 and gravity = 0.0025. 
#   The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall. 
#   The position is clipped to the range [-1.2, 0.6] and velocity is clipped to the range [-0.07, 0.07].
#
# 這個環境的山谷高度函數 height = sin(3 * position). 3 是為了縮放山谷的波長，讓它在 [-1.2, 0.6] 的範圍內
# 剛好呈現出一個漂亮的「V」字型起伏。
# 重力讓一個物體在坡道上受到的下滑力與坡度的傾斜角有關。斜率是高度函數的微積分導數：d/dx(sin(3x)) = 3 cos(3x). 方向與position方向相反.
# 因為 force (0.001) 比 gravity (0.0025) 還小, 意味著如果直接從山谷底下往上衝，重力永遠會贏過車子的引擎。所以必須學會「利用坡度」來累積動能。

# 初始化環境與模型 "MountainCar-v0"
# env = gym.make(ENV_NAME, render_mode="human")     # 看個效果就好，訓練時不需要渲染不然會超慢.
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DuelingDQN(state_dim, action_dim)
target_net = DuelingDQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(f'runs/DuelingDQN_MountainCar_{current_time}')

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    total_shaped_reward = 0  # 紀錄加權後的獎勵供觀察
    
    for t in range(200): # MountainCar-v1 預設 200 步
        epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** episode))
        
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(torch.FloatTensor(state)).argmax().item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # --- [關鍵] Reward Shaping ---
        # 原生 reward 每步都是 -1
        position, velocity = next_state

        if REWARD_SHAPING_METHOD == "potential_height_progress":
            # 機械能獎勵 + 正向進步獎勵 + 高度獎勵.

            # 機械能獎勵 (Mechanical Energy Shaping)：鼓勵小車擺動起來
            #
            # 總機械能為位置能量 + 動能。位置能量與高度成正比，動能與速度的平方成正比。
            #   動能 (K): 0.5*m*v^2。在程式中我們簡化為 0.5 * velocity^2。這鼓勵小車「跑快一點」.
            #   位能 (P): 地形高度是 sin(3*position)。 所以位能與 sin(3*position) 成正比。這鼓勵小車「爬高一點」。
            # 這種 Shaping 方式告訴 Agent：「不管你有沒有抵達旗子，只要你能讓這台車的總能量增加，你就是對的。」
            # 用這個解決「稀疏獎勵」問題，讓 Agent 在山谷底亂晃時就能得到回饋
            # state[0] 是舊位置，position 是新位置；state[1] 是舊速度，velocity 是新速度。
            # shaping = 新能量 - 舊能量.
            #
            # 優點：非常符合物理定律，Agent 會很快學會透過擺盪來累積能量。
            # 缺點：如果參數沒調好，Agent 可能會在那裡「瘋狂擺盪」而不去撞旗子，因為擺盪就能一直拿到能量變化的獎勵。
            shaping = 10 * ((np.sin(3 * position) * 0.0035 + 0.5 * velocity * velocity) - 
                            (np.sin(3 * state[0]) * 0.0035 + 0.5 * state[1] * state[1]))
            
            # 放大正向進步的權重(新狀態的能量>舊舊狀態的能量)，讓它更明顯地引導學習。
            # 這個也是用來解決「稀疏獎勵」的問題，讓 Agent 在山谷底亂晃時就能得到回饋.
            # 這裡的 10 是一個經過實驗調整的係數，可以根據需要進行微調。
            if shaping > 0: shaping *= 10 
            
            # 高度閾值的額外獎勵：
            # 山谷底部在 position = -0.5. 如果爬過一定高度（如 position = -0.4, 0.1, 0.5）就給額外固定獎勵。  
            # 用這個打破對稱性，告訴 Agent 右邊才是正確的方向.
            bonus = 0
            if position > -0.4: bonus = 0.1
            if position > 0.1: bonus = 0.5
            if position >= 0.5: bonus = 10.0 # 抵達旗子大獎
            
            custom_reward = reward + shaping + bonus

        elif REWARD_SHAPING_METHOD == "exp_potential_velocity_progress":
            # 強化(指數級)位能獎勵 + 速度獎勵 + 進度獎勵.

            # 強化(指數級)位能獎勵：因為山谷底部在 -0.5, 當位置大於 -0.5 時，這個值會隨距離增加而加速變大。
            # 用這個打破對稱性，告訴 Agent 右邊才是正確的方向.
            height_reward = 0
            if position > -0.5:
                height_reward = ((position + 0.5) * 10) ** 2  # 指數級增加高度獎勵

            # 速度獎勵：鼓勵在低處加速
            # 用這個解決「稀疏獎勵」問題，讓 Agent 在山谷底亂晃時就能得到回饋
            vel_reward = 100 * abs(velocity)

            # 進度獎勵: 告訴 AI：「接近旗子了，快衝！」
            # 確保 Agent 在學會晃動後，會全力以赴完成最後的衝刺.
            target_bonus = 0
            if position >= 0.5:
                target_bonus = 500  # 給它一個大甜頭

            custom_reward = reward + height_reward + vel_reward + target_bonus
        # ---------------------------

        memory.append((state, action, custom_reward, next_state, done))
        state = next_state
        total_reward += reward
        total_shaped_reward += custom_reward
        
        # 訓練邏輯 (同前)
        if len(memory) > BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)
            
            curr_q = policy_net(states).gather(1, actions.unsqueeze(1))
            next_q = target_net(next_states).max(1)[0].detach()
            target_q = rewards + (1 - dones) * GAMMA * next_q
            
            loss = nn.MSELoss()(curr_q.squeeze(), target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done: break
        
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    writer.add_scalar('Performance/Real_Reward', total_reward, episode)
    writer.add_scalar('Performance/Shaped_Reward', total_shaped_reward, episode)
    print(f"Episode {episode}, Real Reward: {total_reward}, Shaped: {total_shaped_reward:.2f}")

writer.close()
torch.save(policy_net.state_dict(), checkpoint_path)