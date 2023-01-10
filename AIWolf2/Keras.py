# 迷路の読み込み
def readMaze():
  with open("maze.txt") as f:
    maze=f.readlines()
  maze=list(map(lambda x: list(x.strip()),maze))
  return maze

# 迷路を表示
def printMaze(maze):
  for i,r in enumerate(maze):
    for j,c in enumerate(r):
      if i==row and j==col: #  row(行), col(列)を現在の位置とする
        print("*", end="")
      else:
        print(c, end="")
    print()


def get_s_next(s, a , row, col):
    direction = ["up", "right", "down", "left"]
    next_direction = direction[a]  # 行動aの方向

    # 行動から次の状態を決める
    if next_direction == "up":
      if maze[row - 1][col] != '#':
        row -= 1
    if next_direction == "right":
      if maze[row][col + 1] != '#':
        col += 1
    if next_direction == "down":
       if maze[row + 1][col] != '#':
        row += 1
    if next_direction == "left":
       if maze[row][col - 1] != '#':
        col -= 1

    s_next = maze[row][col]

    return s_next, row, col


# namedtupleを生成
from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

# 経験を保存するメモリクラスを定義します
class ReplayMemory:

    def __init__(self, CAPACITY): # 一回だけ読まれる
        self.capacity = CAPACITY  # メモリの最大長さ
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数

    def push(self, state, action, state_next, reward):
        '''transition = (state, action, state_next, reward)をメモリに保存する'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す

        # namedtupleのTransitionを使用し、値とフィールド名をペアにして保存します
        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす

    def sample(self, batch_size):
        '''batch_size分だけ、ランダムに保存内容を取り出す'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)

# エージェントが持つ脳となるクラスです、DQNを実行します
# Q関数をディープラーニングのネットワークをクラスとして定義

import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

BATCH_SIZE = 32
CAPACITY = 10000


class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # 選択しうる行動の数

        # 経験を記憶するメモリオブジェクトを生成
        self.memory = ReplayMemory(CAPACITY)

        # ニューラルネットワークを構築
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 1000))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(1000, 500))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(500, 100))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc4', nn.Linear(100, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc5', nn.Linear(32, num_actions))

        print(self.model)  # ネットワークの形を出力

        # 最適化手法の設定
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):
        '''Experience Replayでネットワークの結合パラメータを学習'''

        # -----------------------------------------
        # 1. メモリサイズの確認
        # -----------------------------------------
        # 1.1 メモリサイズがミニバッチより小さい間は何もしない
        if len(self.memory) < BATCH_SIZE:
            return

        # -----------------------------------------
        # 2. ミニバッチの作成
        # -----------------------------------------
        # 2.1 メモリからミニバッチ分のデータを取り出す
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 各変数をミニバッチに対応する形に変形
        # transitionsは1stepごとの(state, action, state_next, reward)が、BATCH_SIZE分格納されている
        # つまり、(state, action, state_next, reward)×BATCH_SIZE
        # これをミニバッチにしたい。つまり
        # (state×BATCH_SIZE, action×BATCH_SIZE, state_next×BATCH_SIZE, reward×BATCH_SIZE)にする
        batch = Transition(*zip(*transitions))

        # 2.3 各変数の要素をミニバッチに対応する形に変形し、ネットワークで扱えるようVariableにする
        # 例えばstateの場合、[torch.FloatTensor of size 1x4]がBATCH_SIZE分並んでいるのですが、
        # それを torch.FloatTensor of size BATCH_SIZEx4 に変換します
        # 状態、行動、報酬、non_finalの状態のミニバッチのVariableを作成
        # catはConcatenates（結合）のことです。
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                            if s is not None])

        # -----------------------------------------
        # 3. 教師信号となるQ(s_t, a_t)値を求める
        # -----------------------------------------
        # 3.1 ネットワークを推論モードに切り替える
        self.model.eval()

        # 3.2 ネットワークが出力したQ(s_t, a_t)を求める
        # self.model(state_batch)は、右左の両方のQ値を出力しており
        # [torch.FloatTensor of size BATCH_SIZEx2]になっている。
        # ここから実行したアクションa_tに対応するQ値を求めるため、action_batchで行った行動a_tが右か左かのindexを求め
        # それに対応するQ値をgatherでひっぱり出す。
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # 3.3 max{Q(s_t+1, a)}値を求める。ただし次の状態があるかに注意。

        # next_stateがあるかをチェックするインデックスマスクを作成
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)))
        # まずは全部0にしておく
        next_state_values = torch.zeros(BATCH_SIZE)

        # 次の状態があるindexの最大Q値を求める
        # 出力にアクセスし、max(1)で列方向の最大値の[値、index]を求めます
        # そしてそのQ値（index=0）を出力します
        # detachでその値を取り出します
        next_state_values[non_final_mask] = self.model(
            non_final_next_states).max(1)[0].detach()

        # 3.4 教師となるQ(s_t, a_t)値を、Q学習の式から求める
        expected_state_action_values = reward_batch + GAMMA * next_state_values

        # -----------------------------------------
        # 4. 結合パラメータの更新
        # -----------------------------------------
        # 4.1 ネットワークを訓練モードに切り替える
        self.model.train()

        # 4.2 損失関数を計算する（smooth_l1_lossはHuberloss）
        # expected_state_action_valuesは
        # sizeが[minbatch]になっているので、unsqueezeで[minibatch x 1]へ
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # 4.3 結合パラメータを更新する
        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward()  # バックプロパゲーションを計算
        self.optimizer.step()  # 結合パラメータを更新

    def decide_action(self, state, episode):
        '''現在の状態に応じて、行動を決定する'''
        # ε-greedy法で徐々に最適行動のみを採用する
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()  # ネットワークを推論モードに切り替える
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
            # ネットワークの出力の最大値のindexを取り出します = max(1)[1]
            # .view(1,1)は[torch.LongTensor of size 1]　を size 1x1 に変換します

        else:
            # 0,1, 2, 3の行動をランダムに返す
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])  # 0, 1, 2, 3の行動をランダムに返す
            # actionは[torch.LongTensor of size 1x1]の形になります

        return action



maze = readMaze()
turn = 1

step_history = [] # Goalするのにかかったstep数を記録するリスト

# Q学習で迷路を解く
eta = 0.1  # 学習率
GAMMA = 0.9  # 時間割引率
episode = 1

MAX_STEPS = 1000  # 1試行のstep数
NUM_EPISODES = 100  # 最大試行回数

num_states = 2
num_actions = 4

brain = Brain(num_states, num_actions)

for episode in range(NUM_EPISODES):  
    print("エピソード:" + str(episode))

    # row, colをスタート地点に戻す
    row = start_row 
    col = start_col
    s = maze[row][col] # スタート地点 sは, " ", "#", "S", "G"の値をとる
    s_a_history = [['S', np.nan]]  # エージェントの移動を記録するリスト
    x_y_history = [[col, row]]  # エージェントの移動を記録するリスト
    for step in range(MAX_STEPS):  # MAX_STEPSまでループ

        s_array = np.array([row, col]) # 現在いる位置の座標を配列で表示(次にテンソルに変換するため)
        state = torch.from_numpy(s_array).type(torch.FloatTensor) # s_arrayをテンソルに変換
        state = torch.unsqueeze(state, 0)

        a = a_next = brain.decide_action(state, episode) # 行動更新

        s_a_history[-1][1] = a
        # 現在の状態（つまり一番最後なのでindex=-1）に行動を代入

        # row, colが更新されて次の状態(" ", "#", "S", "G")がs_nextに入る
        s_next, next_row, next_col = get_s_next(s, a, row, col)

        s_array_next = np.array([next_row, next_col])
        state_next = torch.from_numpy(s_array_next).type(torch.FloatTensor)
        state_next = torch.unsqueeze(state_next, 0)

          # 報酬を与え,　次の行動を求めます
        if s_next == 'G':
            a_next = np.nan
            state_next = None
            reward = torch.FloatTensor([1.0]) # 報酬を与える
        else:
            reward = torch.FloatTensor([0.0]) # 普段は報酬0


        # メモリに経験を追加
        brain.memory.push(state, a, state_next, reward)

            
        s_a_history.append([s_next, np.nan])
        # 次の状態を代入。行動はまだ分からないのでnanにしておく


        x_y_history.append([next_col, next_row])


        #print("step = " , step) # step数を表示
        #printMaze(maze) # 迷路を表示

        # Experience ReplayでQ関数を更新する
        brain.replay()

        # 終了判定
        if s_next == 'G':  # ゴール地点なら終了
            break
        else:
            s = s_next
            state = state_next
            s_array = s_array_next
            row = next_row
            col = next_col

    # print(s_a_history)
    print("迷路を解くのにかかったステップ数は" + str(len(s_a_history) - 1) + "です")
    step_history.append(len(s_a_history) - 1)
    # print(step_history)
    print(x_y_history)


# 最短経路を表示
for i, r in enumerate(x_y_history):
  for j ,c in enumerate(r):
    if j==0:
      x=c
    if j==1:
      y=c
    if maze[x][y] != 'S' and maze[x][y] != 'G' and maze[x][y] != '#':
      maze[x][y] = '>'

for i,r in enumerate(maze):
    for j,c in enumerate(r):
        print(c, end="")
    print()


# グラフを描画
plt.title("learningprocess")
plt.xlabel("num_episode")
plt.ylabel("num_step")
x = range(NUM_EPISODES)
y = step_history

plt.plot(x, y)
plt.show