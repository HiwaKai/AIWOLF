#!/usr/bin/env python

# This sample script connects to the AIWolf server, but
# does not do anything else. It will choose itself as the
# target for any actions requested by the server, (voting,
# attacking ,etc) forcing the server to choose a random target.
import logging
from logging import getLogger, StreamHandler, Formatter, FileHandler

from numpy.lib.function_base import diff
import aiwolfpy
import argparse

import numpy as np
import random
import csv

# name
myname = 'miura'

# content factory
cf = aiwolfpy.ContentFactory()

# logger
logger = getLogger("aiwolfpy")
logger.setLevel(logging.NOTSET)
# handler
stream_handler = StreamHandler()
stream_handler.setLevel(logging.NOTSET)
handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(handler_format)


logger.addHandler(stream_handler)

# file_handler = FileHandler('aiwolf_game.log')
# file_handler.setLevel(logging.WARNING)
# file_handler.setFormatter(handler_format)
# logger.addHandler(file_handler)

#----------------------------- DQN -------------------------------------#

# namedtupleを生成
from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, CAPACITY): # 一回だけ読まれる
        self.capacity = CAPACITY  # メモリの最大長さ
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数

    def push(self, state, action, next_state, reward):
        '''transition = (state, action, next_state, reward)をメモリに保存する'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す

        # namedtupleのTransitionを使用し、値とフィールド名をペアにして保存します
        self.memory[self.index] = Transition(state, action, next_state, reward)

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

BATCH_SIZE = 100
CAPACITY = 450000 # メモリの最大の長さ

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # 行動を取得
        self.GAMMA = 0.99  # 時間割引率

        # 経験を記憶するメモリオブジェクトを生成
        self.memory = ReplayMemory(CAPACITY)

        # ニューラルネットワークを構築
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 128))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(128, 64))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(64, 32))
        self.model.add_module('relu3', nn.ReLU())
        self.model.add_module('fc4', nn.Linear(32, num_actions))
        self.model.add_module('relu4', nn.ReLU())

        # print(self.model)  # ネットワークの形を出力

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
        state_action_values = self.model(state_batch).gather(0, action_batch)

        # 3.3 max{Q(s_t+1, a)}値を求める。ただし次の状態があるかに注意。

        # cartpoleがdoneになっておらず、next_stateがあるかをチェックするインデックスマスクを作成
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
        expected_state_action_values = reward_batch + self.GAMMA * next_state_values

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
        # epsilon = 0.9 * (1 / (episode + 1)) # epislonの初期値は0.9
        epsilon = -(episode / 100000) + 1

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()  # ネットワークを推論モードに切り替える
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
            # ネットワークの出力の最大値のindexを取り出します = max(1)[1]
            # .view(1,1)は[torch.LongTensor of size 1]　を size 1x1 に変換します

        else:
            # 0,1の行動をランダムに返す
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])  # 0,1の行動をランダムに返す
            # actionは[torch.LongTensor of size 1x1]の形になります

        return action


#----------------------------- DQN -------------------------------------#


class SampleAgent(object):
    
    def __init__(self):
        # my name
        self.base_info = dict()
        self.game_setting = dict()
        self.episode = 1 # タイムステップ
        self.num_win = 0 # 勝った数を記録する
        self.win_rate_list = [] # 勝率を記録していく
        # Q学習の定数の定義
        self.eta = 0.001  # 学習率
        self.MAX_EPISODES = 100000  # 最大試行回数
        self.num_states = 5
        self.num_actions = 4

        self.villager_vote = Brain(self.num_states, self.num_actions)
        # self.villager_vote = torch.load("villager_vote.pth")
        self.s_list=[]
        self.a_list = []


    def getName(self):
        return self.my_name
    
    # new game (no return)
    def initialize(self, base_info, diff_data, game_setting):
        self.base_info = base_info
        self.game_setting = game_setting

        self.myRole = self.base_info['myRole']
        self.myIdx = self.base_info['agentIdx']
        print(base_info)

        self.reward = torch.FloatTensor([0.0])
        self.villager_vote_s_a_list = [] # 投票の経験を保存しておくリスト
        self.fact_list = np.zeros((5), dtype=int) # 事実にある特徴
        # print("initialize: \n", diff_data)
        print(self.episode)
        
    # new information (no return)
    def update(self, base_info, diff_data, request):
        self.base_info = base_info
        # print("update: \n", diff_data)
        def _get_fact_list(diff_data, fact_list):
            for i in range(diff_data.shape[0]):
                # どれだけ投票されたか
                if diff_data['type'][i] == 'vote':
                    fact_list[diff_data['agent'][i]-1] += 1

            return fact_list
        self.fact_list = _get_fact_list(diff_data, self.fact_list)
        # print(self.fact_list)

        if request == 'DAILY_FINISH':
         
            self.state = torch.from_numpy(self.fact_list).type(torch.FloatTensor)  # numpy変数をPytorchのテンソルに変換
            self.state = torch.unsqueeze(self.state, 0) # sizeをsize X 1 に変換 ミニバッチとして扱いやすくするため
            self.a = self.villager_vote.decide_action(self.state, self.episode) + 1
            # print("action:" ,self.a)

            # 経験をリストに追加
            self.villager_vote_s_a_list.append([self.state, self.a])  # 経験をリストに追加
            self.s_list.append(self.state.tolist())
            self.a_list.append(self.a.item())
            # print("s_a_list: ", self.villager_vote_s_a_list)
        

        if request == 'FINISH':

            self.state = torch.from_numpy(self.fact_list).type(torch.FloatTensor)  # numpy変数をPytorchのテンソルに変換
            self.state = torch.unsqueeze(self.state, 0) # sizeをsize X 1 に変換 ミニバッチとして扱いやすくするため

            # 最後の状態をリストに追加
            self.villager_vote_s_a_list.append([self.state, 0])

            # 報酬を作る
            self.Winner = 0
            self.episode += 1
            for i in range(len(diff_data)):
                if 'WEREWOLF' in diff_data['text'][i]:
                    if base_info['statusMap'][str(i+1)] == 'ALIVE':
                        self.Winner = 1
                        break

            if self.Winner == 0 and (self.myRole == 'SEER' or self.myRole == 'VILLAGER'):
                self.reward = torch.FloatTensor([1.0]) # 報酬を与える
                self.num_win += 1
            elif self.Winner == 1 and (self.myRole == 'WEREWOLF' or self.myRole == 'POSSESSED'):
                self.reward = torch.FloatTensor([1.0]) # 報酬を与える
                self.num_win += 1
            else:
                self.reward = torch.FloatTensor([0.0]) # 負けたら-1.0
            # print(self.Winner)
            # print(self.reward)

            # メモリに経験を追加
            # print(self.villager_vote_s_a_list)
            for i in range(len(self.villager_vote_s_a_list)-1):
              self.villager_vote.memory.push(self.villager_vote_s_a_list[i][0],
                                             self.villager_vote_s_a_list[i][1],
                                             self.villager_vote_s_a_list[i+1][0],
                                             self.reward)
            # Experience ReplayでQ関数を更新する
            self.villager_vote.replay()

            # 100回ごとの勝率を記録する
            if self.episode % 100 == 0:
                self.win_rate_list.append(self.num_win / 100)
                self.num_win = 0

            if self.episode == self.MAX_EPISODES: # 全てのエピソードが終了したら
                with open('win_rate/villager_win_rate.csv', 'a') as f:
                    writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
                    writer.writerow(self.win_rate_list)     # list（1次元配列）の場合
                with open('state/s_list.csv', 'a') as f:
                    writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
                    writer.writerow(self.s_list)
                with open('action/a_list.csv', 'a') as f:
                    writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
                    writer.writerow(self.a_list)     # list（1次元配列）の場合
                torch.save(self.villager_vote, "villager_vote.pth") # DeskTopに保存


    # Start of the day (no return)
    def dayStart(self):
        return None

    # conversation actions: require a properly formatted
    # protocol string as the return.
    def talk(self):
        
        return cf.over()
    
    def whisper(self):
        return cf.over()
        
    # targetted actions: Require the id of the target
    # agent as the return
    def vote(self):
        return self.a

    def attack(self):
        return self.base_info['agentIdx']

    def divine(self):
        return self.base_info['agentIdx']

    def guard(self):
        return self.base_info['agentIdx']

    # Finish (no return)
    def finish(self):
        return None
    

agent = SampleAgent()

# read args
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-p', type=int, action='store', dest='port')
parser.add_argument('-h', type=str, action='store', dest='hostname')
parser.add_argument('-r', type=str, action='store', dest='role', default='none')
parser.add_argument('-n', type=str, action='store', dest='name', default=myname)
input_args = parser.parse_args()


client_agent = aiwolfpy.AgentProxy(
    agent, input_args.name, input_args.hostname, input_args.port, input_args.role, logger, "pandas"
)

# run
if __name__ == '__main__':
    client_agent.connect_server()