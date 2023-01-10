#!/usr/bin/env python
from __future__ import print_function, division

# this is main script
# simple version

import aiwolfpy
import aiwolfpy.contentbuilder as cb
import random
import numpy as np
import keras
import os
from keras.models import model_from_json
import csv


myname = 'dqn_villager'


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

        # 経験を記憶するメモリオブジェクトを生成
        self.memory = ReplayMemory(CAPACITY)

        # ニューラルネットワークを構築
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 128))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(128, 64))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(64, 64))
        self.model.add_module('relu3', nn.ReLU())
        self.model.add_module('fc4', nn.Linear(64, 32))
        self.model.add_module('relu4', nn.ReLU())
        self.model.add_module('fc5', nn.Linear(32, num_actions))
        self.model.add_module('relu5', nn.ReLU())

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
        state_action_values = self.model(state_batch).gather(1, action_batch)

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
        # epsilon = 0.5 * (1 / (episode + 1))
        epsilon = (-0.7/450000) * episode + 0.9
        if epsilon < 0.2:
          epsilon = 0.2
        

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()  # ネットワークを推論モードに切り替える
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
            # ネットワークの出力の最大値のindexを取り出します = max(1)[1]
            # .view(1,1)は[torch.LongTensor of size 1]　を size 1x1 に変換します

        else:
            # 0,1,2,3 の行動をランダムに返す
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])  # 0,1,2,3 の行動をランダムに返す
            # actionは[torch.LongTensor of size 1x1]の形になります

        return action



class Agent(object):

    def __init__(self, agent_name):

        """
        DQNに使う変数
        """
        self.pre_state = np.zeros((5,5,4), dtype=int)  # 状態表現をいれる三次元配列
        self.episode = 0 # タイムステップ
        self.num_win = 0 # 勝った数を記録する
        self.win_rate_list = [] # 勝率を記録していく


    def initialize(self, base_info, diff_data, game_setting):
        self.base_info = base_info
        self.game_setting = game_setting

        print('episode:', self.episode)

        """
        DQNに使う変数
        """
        self.myRole = self.base_info['myRole']
        self.myIdx = self.base_info['agentIdx']
        self.declare_list = np.zeros((4), dtype=int) # 宣言にある特徴
        self.comingout = ''
        self.comingout_list = ['', '', '', '', ''] # nがカミングアウトしていればその役職名
        self.comingout_list_int = np.zeros((5), dtype=int)  # 0:カミングアウトしていない 1:VILLAGERとしてカミングアウト 2:SEER 3:POSSESSED 4:WEREWOLF
        self.alive_list = np.ones((5), dtype=int)  # nが生存 1 : 死亡 0
        self.myself_list = np.zeros((5), dtype=int)  # nがエージェントであれば 1
        self.attitude_list = np.zeros((4), dtype=int) # 態度にある特徴
        self.say_ability_to_list = np.zeros((5,5), dtype=int) # mがnに特殊能力を使うと言ったか  言った 1: 言っていない 0
        self.say_vote_to_list = np.zeros((5,5), dtype=int) # mがnに投票すると言ったかどうか
        self.suspect_list = np.zeros((5,5), dtype=int) # mがnを疑うと言ったかどうか
        self.say_voted_to_list = np.zeros((5,5), dtype=int) # mがnに投票したと言ったかどうか
        self.fact_list = np.zeros((4), dtype=int) # 事実にある特徴
        self.divined_list = ['', '', '', '', ''] # nの役職名 (占い師の場合、占い結果を入れる)
        self.divined_list_int = np.zeros((5), dtype=int) # 0:不明 5:HUMAN 4:WEREWOLF
        self.voted_to_list = np.zeros((5), dtype=int) # エージェントがnに投票したかどうか
        self.abilitied_list = np.zeros((5), dtype=int) # エージェントがnに特殊能力を使ったかどうか

        self.reward = torch.FloatTensor([0.0])

        self.villager_vote_s_a_list = [] # 投票の経験を保存しておくリスト

        self.wolf_predict = 0 # 人狼推定の結果を入れる変数
        self.seer_predict = 0
        self.poss_predict = 0


        # Q学習の定数の定義
        self.eta = 0.001  # 学習率
        self.GAMMA = 0.99  # 時間割引率


        self.MAX_EPISODES = 20000  # 最大試行回数
        self.plot = 1000  # 何回ごとの勝率を測定するか

        self.num_states = 102
        self.num_actions = 4

        self.villager_vote = Brain(self.num_states, self.num_actions)
        # self.villager_vote = torch.load("villager_vote.pth")


    def update(self, base_info, diff_data, request):

        # print(request)

        if request == "DAILY_INITIALIZE" or request == "DIVINE" or request == "ATTACK":
           

        # 生存者をAliveリストに格納する
        self.AliveList = [
            int(id) for id, status in self.base_info['statusMap'].items() if (status == 'ALIVE') and (int(id) != self.myIdx)]
        # print(self.AliveList)

        # 状態表現を取得
        for i in range(5):
            for j in range(5):
                if i == j:
                    self.pre_state[i, j] = self.get_declare_list(base_info, diff_data, j + 1)
                elif i + 1 == self.base_info['agentIdx'] and i != j:
                    self.pre_state[i, j] = self.get_fact_list(base_info, diff_data, j + 1)
                else:
                    self.pre_state[i, j] = self.get_attitude_list(base_info, diff_data, i + 1, j + 1)


        if request == 'DAILY_FINISH': # 1日が終わったら

            self.state = self.pre_state.reshape(-1) #三次元配列を１次元配列に変換
            self.state = np.append(self.state, self.base_info["day"])  # 日付を追加
            # print('state', self.state)
            # self.state_next = None
            # print('self.state1: ',self.state)
            # print('self.state_next: ', self.state_next)
            self.state = torch.from_numpy(self.state).type(torch.FloatTensor)  # numpy変数をPytorchのテンソルに変換
            self.state = torch.unsqueeze(self.state, 0) # sizeをsize X 1 に変換 ミニバッチとして扱いやすくするため

            self.a = self.villager_vote.decide_action(self.state, self.episode) + 1

            while(self.a not in self.AliveList): # self.a がAlivelistに含まれるまでself.aを選び直す
                self.a = self.villager_vote.decide_action(self.state, self.episode) + 1

            # 経験をリストに追加
            self.villager_vote_s_a_list.append([self.state, self.a])  # 経験をリストに追加

        if request == 'FINISH':

            # print(self.pre_state)
            self.state = self.pre_state.reshape(-1)
            self.state = np.append(self.state, self.base_info["day"])  # 日付を追加
            self.state = np.append(self.state, self.wolf_predict)


            self.state = torch.from_numpy(self.state).type(torch.FloatTensor)  # numpy変数をPytorchのテンソルに変換
            self.state = torch.unsqueeze(self.state, 0) # sizeをsize X 1 に変換 ミニバッチとして扱いやすくするため

            # self.a = self.villager_vote.decide_action(self.state, self.episode) + 1 # 最後なので行動はできない

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

            if self.Winner == 0 and (self.myRole == 'SEER' or self.myRole == 'VILLAGER'): # 自分が村人陣営で村人陣営が勝ったら
                self.reward = torch.FloatTensor([1.0]) # 報酬を与える
                self.num_win += 1
            elif self.Winner == 1 and (self.myRole == 'WEREWOLF' or self.myRole == 'POSSESSED'): # 自分が狼陣営で狼陣営が勝ったら
                self.reward = torch.FloatTensor([1.0]) # 報酬を与える
                self.num_win += 1
            else:
                self.reward = torch.FloatTensor([-1.0]) # 負けたら
            # print(self.Winner)
            # print(self.reward)

            # メモリに経験を追加
            # print(self.villager_vote_s_a_list)
            n=len(self.villager_vote_s_a_list)
            for i in range(n-1):
              r=self.reward if i==n-2 else torch.FloatTensor([0.0])
              self.villager_vote.memory.push(self.villager_vote_s_a_list[i][0],
                                             self.villager_vote_s_a_list[i][1],
                                             self.villager_vote_s_a_list[i+1][0],
                                             r)
            # Experience ReplayでQ関数を更新する
            self.villager_vote.replay()

            # 1000回ごとの勝率を記録する
            if self.episode % self.plot == 0:
                self.win_rate_list.append(self.num_win / self.plot)
                self.num_win = 0

            if self.episode == self.MAX_EPISODES: # 全てのエピソードが終了したら
                with open('villager_win_rate.csv', 'a') as f:
                    writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
                    writer.writerow(self.win_rate_list)     # list（1次元配列）の場合
                torch.save(self.villager_vote, "villager_vote.pth") # 保存



    def dayStart(self):
        return None

    def talk(self):
        if self.base_info['myRole'] == 'SEER' and self.comingout == '':
            self.comingout = 'SEER'
            return cb.comingout(self.base_info['agentIdx'], self.comingout)
        elif self.base_info['myRole'] == 'VILLAGER' and 1 in self.suspect_list[:, self.myIdx-1]: # 村人かつ自分が疑われていたら
            self.comingout = 'VILLAGER'
            return cb.comingout(self.base_info['agentIdx'], self.comingout)
        elif self.base_info['myRole'] == 'POSSESSED' and 1 in self.suspect_list[:, self.myIdx-1]:
            self.comingout = 'SEER'
            return cb.comingout(self.base_info['agentIdx'], self.comingout)
        elif self.base_info['myRole'] == 'WEREWOLF' and 1 in self.suspect_list[:, self.myIdx-1]:
            self.comingout = 'VILLAGER'
            return cb.comingout(self.base_info['agentIdx'], self.comingout)
        return cb.over()

    def whisper(self):
        return cb.over()

    def vote(self):
        return self.a

    def attack(self):
        return None

    def divine(self):
        return None

    def guard(self):
        return None

    def finish(self):
        return None



    """
    DQNの状態表現を抽出メソッド
    """
    def role_to_int(self,x):
        if x == '': y = 0
        elif x == 'VILLAGER': y = 1
        elif x == 'SEER': y = 2
        elif x == 'POSSESSED': y = 3
        elif x == 'WEREWOLF': y = 4
        elif x == 'HUMAN': y = 5
        return y

    def get_declare_list(self, base_info, diff_data, n):
        for i in range(diff_data.shape[0]):
            # CO状況
            if diff_data['type'][i] == 'talk' or diff_data['type'][i] == 'finish' or diff_data['type'][i] == 'initialize':
                if diff_data['agent'][i] == n and 'COMINGOUT' in diff_data['text'][i] and 'BECAUSE' not in diff_data['text'][i]:
                    remark = diff_data['text'][i].split() # remark = ['COMINGOUT','AGENT[01]','SEER']
                    self.comingout_list_int[n-1] = self.role_to_int(remark[2])
            # 生存しているかどうか
            if diff_data['type'][i] == 'execute' or diff_data['type'][i] == 'dead': # 処刑または襲撃されたら
                if diff_data['agent'][i] == n:
                    self.alive_list[n-1] = 0

            if base_info['statusMap'][str(n)] == 'DEAD':
                    self.alive_list[n-1] = 0


            # 自己がエージェントかどうか
            if self.myIdx == n:
                self.myself_list[n-1] = 1

        # declare_listに特徴を追加
        declare_list = [self.comingout_list_int[n-1], self.alive_list[n-1], self.myself_list[n-1], 0]

        return declare_list

    def get_attitude_list(self, base_info, diff_data, m , n):

        for i in range(diff_data.shape[0]):
            if n == 1:
                remark_agent = 'Agent[01]'
            if n == 2:
                remark_agent = 'Agent[02]'
            if n == 3:
                remark_agent = 'Agent[03]'
            if n == 4:
                remark_agent = 'Agent[04]'
            if n == 5:
                remark_agent = 'Agent[05]'
            # mがnに特殊能力を使うと言ったかどうか
            if diff_data['type'][i] == 'talk':
                if 'DIVINATION' in diff_data['text'][i] or 'DIVINED' in diff_data['text'][i]:
                    remark = diff_data['text'][i].split() # DIVINED Agent[04] WEREWOLF
                    if diff_data['agent'][i] == m and remark[1] == remark_agent:
                        self.say_ability_to_list[m-1, n-1] = 1

            # mがnに投票するといったかどうか
            if diff_data['type'][i] == 'talk':
                if 'VOTE' in diff_data['text'][i] and 'REQUEST' not in diff_data['text'][i]:
                    remark = diff_data['text'][i].split() # VOTE Agent[02]
                    if diff_data['agent'][i] == m and remark[1] == remark_agent:
                        self.say_vote_to_list[m-1, n-1] = 1


                if 'VOTE' in diff_data['text'][i] and 'REQUEST' in diff_data['text'][i]:
                    remark = diff_data['text'][i].split() # REQUEST ANY (VOTE AGENT[02])
                    if diff_data['agent'][i] == m and remark[3][:-1] == remark_agent:  # remark[3][:-1]は AGENT[02]) から一番後ろの文字をとった値
                        self.say_vote_to_list[m-1, n-1] = 1

            # mがnを疑うかどうか
            if diff_data['type'][i] == 'talk':
                if 'ESTIMATE' in diff_data['text'][i]:
                    remark = diff_data['text'][i].split() # ESTIMATE [target] [role]
                    if remark[3] == 'WEREWOLF' or remark[3] == 'POSSESSED':
                        if diff_data['agent'][i] == m and remark[1] == remark_agent:
                            self.suspect_list[m-1, n-1] = 1
                if '(ESTIMATE' in diff_data['text'][i]:
                    remark = diff_data['text'][i].split() # [subject] ESTIMATE [target] [role]
                    if remark[remark.index('(ESTIMATE') + 2][:-1] == 'WEREWOLF' or remark[remark.index('(ESTIMATE') + 2][:-1] == 'POSSESSED':
                        if diff_data['agent'][i] == m and remark[remark.index('(ESTIMATE') + 1] == remark_agent:
                            self.suspect_list[m-1, n-1] = 1


                if 'DIVINED' in diff_data['text'][i]:
                    remark = diff_data['text'][i].split() # DIVINED Agent[04] WEREWOLF
                    if remark[2] == 'WEREWOLF' or remark[2] == 'POSSESSED':
                        if diff_data['agent'][i] == m and remark[1] == remark_agent:
                            self.suspect_list[m-1][n-1] = 1
                if '(DIVINED' in diff_data['text'][i]:
                    remark = diff_data['text'][i].split() # DIVINED Agent[04] WEREWOLF
                    if remark[remark.index('(DIVINED') + 2][:-1] == 'WEREWOLF' or remark[remark.index('(DIVINED') + 2][:-1] == 'POSSESSED':
                        if diff_data['agent'][i] == m and remark[remark.index('(DIVINED') + 1] == remark_agent:
                            self.suspect_list[m-1, n-1] = 1

            # mがnに投票したといったかどうか
            if diff_data['type'][i] == 'talk':
                if 'VOTED' in diff_data['text'][i]:
                    remark = diff_data['text'][i].split()
                    if diff_data['agent'][i] == m and remark[2] == n:
                        self.say_voted_to_list[m-1, n-1] = 1


        # attitude_listに特徴を追加
        attitude_list = [self.say_ability_to_list[m-1, n-1], self.say_vote_to_list[m-1, n-1], self.suspect_list[m-1, n-1], self.say_voted_to_list[m-1, n-1]]

        return attitude_list

    def get_fact_list(self, base_info, diff_data, n):

        for i in range(diff_data.shape[0]):
            # nの役職(占い師の場合、占い結果)
            if self.myRole == 'SEER':
                if diff_data['type'][i] == 'divine' and diff_data['agent'][i] == n:
                    remark = diff_data['text'][i].split()
                    self.divined_list_int[n-1] = self.role_to_int(remark[2]) # HUMAN:5 WEREWOLF:4


            # エージェントがnに投票したかどうか
            if diff_data['type'][i] == 'vote':
                if diff_data['agent'][i] == n and diff_data['idx'][i] == self.myIdx: # nが投票対象者の時 かつ 自分が投票者
                    self.voted_to_list[n-1] = 1

            # エージェントnに特殊能力を使ったかどうか
            if diff_data['type'][i] == 'divine':
                if diff_data['agent'][i] == n and diff_data['idx'][i] == self.myIdx: # nが能力対象者の時 かつ 自分が能力使用者
                    self.abilitied_list[n-1] = 1

        # fact_listに特徴を追加
        fact_list = [self.divined_list_int[n-1], self.voted_to_list[n-1], self.abilitied_list[n-1], 0]

        return fact_list



agent = Agent(myname)


# run
if __name__ == '__main__':
    aiwolfpy.connect_parse(agent)
