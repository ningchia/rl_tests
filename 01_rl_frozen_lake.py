# 需要 pip install gymnasium[toy-text] 和 numpy. 
# 注意是 gymnasium[toy-text] 不是 gymnasium，因為 FrozenLake 是一個「文字遊戲」環境。
# ref: https://gymnasium.farama.org/index.html
#      https://vocus.cc/article/67678732fd897800010a5836

import numpy as np
import gymnasium as gym
import random

# 1. 初始化環境
# is_slippery=False 讓物理規則簡單點：往哪走就真的往哪走
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")

# 2. 初始化 Q 表格 (16 個狀態 x 4 個動作)
# 狀態(observation)：0-15 (4x4 格子), 動作(action)：0:左, 1:下, 2:右, 3:上
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# 3. 設定超參數
learning_rate = 0.1    # 學習率 (alpha)
discount_factor = 0.95 # 折扣因子 (gamma)
epsilon = 0.9          # 探索率 (一開始多嘗試新事物)
epsilon_decay = 0.999  # 隨著時間減少探索
episodes = 500         # 玩幾局

print("正在訓練中...")

# 4. 開始訓練
for i in range(episodes):
    state, _ = env.reset()      # env.reset() 會回傳初始狀態和一些額外資訊，我們只需要狀態，所以用 _ 來忽略它。
    done = False
    
    while not done:
        # Epsilon-Greedy: 決定要「探索」還是「利用」
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # 隨機亂走
        else:
            action = np.argmax(q_table[state, :]) # 走目前這個狀態下所有action中分數最高的action

        # 執行動作，獲得回饋, 並觀察新的狀態(下一個狀態是什麼), 以及是否結束.
        # terminate 是因為掉進洞裡或成功到達終點，truncated 是因為超過最大步數限制。
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # --- Q-learning 核心公式 (貝爾曼方程式) ---
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])   # 下一個狀態的最大 Q 值
        
        # 更新 Q 值.
        # discount_factor * next_max 是未來的期望回報，reward 是當前的回報，兩者加起來就是「總回報」，減去 old_value 就是「誤差」，
        # 乘以 learning_rate 就是「調整幅度」。
        # 之所以要乘上discount_factor，是因為未來的回報通常不如當前的回報重要（尤其是在長期任務中），所以會打折扣。
        #
        # 這樣的更新方式相當於是一種backward update, 用預測的未來回報來修正當前的 Q 值，讓它更接近真實的總回報。
        # 差別是, 監督式學習的backward update是用真實的標籤來修正預測值, 而強化學習的backward update是用預測的未來回報來修正當前的 Q 值。
        # 而且, 監督式學習的backward update是從loss"一次性"的回推更新"所有的"權重與Bias, 
        # 而強化學習的backward update則是逐步地在每一步中, 用"下一步的預測Q值"扮演target值, 來進行"當前這一步"的Q值更新。
        # 這被稱為 時間差分學習 (Temporal Difference Learning, TD)。它的資訊流是從 t+1 時刻（未來）流向 t 時刻（現在）。
        #
        # 也因此, RL 在每一步都在更新 Q 值。但這會導致一個尷尬的情況：正在追逐的「目標」本身就是由 Q 函數算出來的。
        # 像是一邊跑步，一邊手拿著終點線往前跑。
        # 這就是為什麼在深度強化學習（DQN）中，我們需要多加一個 Target Network：我們強行讓終點線「固定」一段時間
        # （例如 1000 步才更新一次 Target），好讓模型能稍微喘口氣，穩定地學會一段路徑。
        new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
        q_table[state, action] = new_value
        
        state = next_state
        
    epsilon *= epsilon_decay # 讓機器人越來越「穩重」

print("訓練完成！")

# 5. 測試訓練結果
print("\n最終 Q 表格 (部分):")
print(q_table[:5]) # 顯示前 5 格的 Q 值

# 關閉環境
env.close()