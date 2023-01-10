#!/usr/bin/env python
from __future__ import print_function, division

# this is main script
# simple version

import logging
from logging import getLogger, StreamHandler, Formatter, FileHandler
import aiwolfpy
import aiwolfpy.contentbuilder as cb
from collections import deque
import numpy as np
import argparse
import csv
import sys
import os
import time
import pickle
import random
import time

myname = 'test'

# logger
logger = getLogger("aiwolfpy")
logger.setLevel(logging.NOTSET)
# handler
stream_handler = StreamHandler()
stream_handler.setLevel(logging.NOTSET)
handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(handler_format)


logger.addHandler(stream_handler)


import aiwolfpy.ql
code = {"WEREWOLF":1,"VILLAGER":2,"SEER":3,"POSSESSED":4,"BODYGUARD":5,"MEDIUM":6,"ANY":7,"HUMAN":8,"SELF":9}
thisRole = "WEREWOLF"
rounds = 1000000
class SampleAgent(object):

    def __init__(self, agent_name):
        # myname
        self.myname = agent_name
        self.dqn_talk = aiwolfpy.ql.DeepQNetwork_v2(thisRole,"talk")
        if (os.path.exists("kill_rate_gamecount")):
            self.gamecount = pickle.load(open("kill_rate_gamecount","rb+"))
        else:
            self.gamecount = 0
        self.win_count = 0
        self.reward_count = 0
        self.time_sta = time.time()


    def getName(self):
        return self.myname

    def initialize(self, base_info, diff_data, game_setting):
        #print("Initialize")
        self.base_info = base_info
        # game_setting
        self.gamecount += 1
    
        self.playerNum = 5
        # print("--------------" + str(self.gamecount) + "----------------")
        # self.game_setting = game_setting
        # print("--------------" + "base_info" + "----------------")
        # print(base_info)
        # print("-------------------------------------------------")
        # print("--------------" + "game_setting" + "----------------")
        # print(game_setting)
        # print("-------------------------------------------------")
        # self.playerNum = len(base_info["statusMap"])
        #state=特殊能力:役職:CO, 投票:投票:生存, 疑う:特殊能力:自己, 投票済み

        self.stateCo = np.zeros((self.playerNum,self.playerNum))
        self.stateEs = np.zeros((self.playerNum,self.playerNum))
        self.stateVote = np.zeros((self.playerNum,self.playerNum))
        self.stateTalk = np.zeros((self.playerNum,self.playerNum))
        self.stateTalkNum = np.zeros(self.playerNum)
        self.stateIsAlive = np.zeros(self.playerNum)
        self.day = -1

        self.index = base_info['agentIdx']
        self.islearn = False
        self.role = base_info["myRole"]
        self.finishCount = 0
        self.possiblewolf = []
        self.lastatack = None
        self.atackaction = None

        self.reward = 0

        # for i in range(15):
        for i in range(5):
            self.possiblewolf.append(True)
        #print(thisRole,self.role)
        if "VILLAGER" == self.role or "WEREWOLF" == self.role:
            #print("equal")
            if os.path.exists("kill_rate_count"):
                self.count = pickle.load(open("kill_rate_count","rb+"))
            else:
                self.count = 0
            self.count += 1
            if self.count <= rounds:
                self.islearn = True
                pickle.dump(self.count,open("kill_rate_count","wb+"))
                self.stateLog = []
                self.actionLog = []

    def update(self, base_info, diff_data, request):
        #print(self.day,"update")
        self.base_info = base_info
        self.diff_data = diff_data
        self.statusMap = base_info["statusMap"]
        if len(diff_data) == 0 or not self.islearn:
            #print(self.day,"return")
            return
        #print(diff_data,request)
        if diff_data["type"][0] == "whisper":
            #print("whisper")
            whisperlist = np.array([diff_data["agent"],diff_data["text"]])
            #print(whisperlist)
            for i in range(len(whisperlist)):
                content = self.splitText(whisperlist[1][i])
                if len(content) == 0:
                    continue
                for j in range(len(whisperlist[0])):
                    self.possiblewolf[int(whisperlist[0][j])-1] = False
                if content[0] == "COMINGOUT":
                    if content[2] == "WEREWOLF":
                        # print("#####", content[0],content[1],content[2])
                        pass
                        #self.possiblewolf[int(content[1])-1] = False

        if diff_data["type"][0] == "talk":
            #self.day = diff_data["day"][0] + 1
            talklist = np.array([diff_data["agent"],diff_data["text"]])
            print("--- talklist ---")
            print(talklist)
            for i in range(len(talklist[0])):
                content = self.splitText(talklist[1][i])
         
                if len(content) == 0:
                    continue
                print("#####", content[0],content[1])
                if content[0] == "COMINGOUT":
                    self.stateCo[talklist[0][i]-1][int(content[1])-1] = code[content[2]]
                    self.stateTalk[talklist[0][i]-1][int(content[1])-1] += 1
                    self.stateTalkNum[talklist[0][i]-1] += 1
                elif content[0] == "ESTIMATE":
                    self.stateEs[talklist[0][i]-1][int(content[1])-1] = code[content[2]]
                    self.stateTalk[talklist[0][i]-1][int(content[1])-1] += 1
                    self.stateTalkNum[talklist[0][i]-1] += 1
                elif content[0] == "DIVINED":
                    self.stateTalk[talklist[0][i]-1][int(content[1])-1] += 1
                    self.stateTalkNum[talklist[0][i]-1] += 1
                elif content[0] == "DIVINATION":
                    self.stateTalk[talklist[0][i]-1][int(content[1])-1] += 1
                    self.stateTalkNum[talklist[0][i]-1] += 1
                elif content[0] == "GUARD":
                    self.stateTalk[talklist[0][i]-1][int(content[1])-1] += 1
                    self.stateTalkNum[talklist[0][i]-1] += 1
                elif content[0] == "VOTE":
                    self.stateTalk[talklist[0][i]-1][int(content[1])-1] += 1
                    self.stateTalkNum[talklist[0][i]-1] += 1
                elif content[0] == "ATTACK":
                    self.stateTalk[talklist[0][i]-1][int(content[1])-1] += 1
                    self.stateTalkNum[talklist[0][i]-1] += 1
                elif content[0] == "DIVINED":
                    self.stateTalk[talklist[0][i]-1][int(content[1])-1] += 1
                    self.stateTalkNum[talklist[0][i]-1] += 1
                elif content[0] == "IDENTIFIED":
                    self.stateTalk[talklist[0][i]-1][int(content[1])-1] += 1
                    self.stateTalkNum[talklist[0][i]-1] += 1
                elif content[0] == "GUARDED":
                    self.stateTalk[talklist[0][i]-1][int(content[1])-1] += 1
                    self.stateTalkNum[talklist[0][i]-1] += 1
                elif content[0] == "VOTED":
                    self.stateTalk[talklist[0][i]-1][int(content[1])-1] += 1
                    self.stateTalkNum[talklist[0][i]-1] += 1
                elif content[0] == "ATTACKED":
                    self.stateTalk[talklist[0][i]-1][int(content[1])-1] += 1
                    self.stateTalkNum[talklist[0][i]-1] += 1
                elif content[0] == "AGREE":
                    self.stateTalkNum[talklist[0][i]-1] += 1
                elif content[0] == "DISAGREE":
                    self.stateTalkNum[talklist[0][i]-1] += 1

        #stateVote
        if diff_data["type"][0] == "vote":
            #idxが投票者,agentが投票対象
            votelist = np.array(diff_data["idx"],diff_data["agent"])
            for i in range(len(votelist)):
                self.stateVote[votelist[0]-1][votelist[1]-1] += 1

        #stateIsAlive
        for i in range(len(diff_data)):
            if diff_data["type"][i] == "execute":
                if self.stateIsAlive[diff_data["agent"][i]-1] == 0:
                    self.stateIsAlive[diff_data["agent"][i]-1] = 1 + self.day
            elif diff_data["type"][i] == "dead":
                if self.stateIsAlive[diff_data["agent"][i]-1] == 0:
                    self.stateIsAlive[diff_data["agent"][i]-1] = 20 + self.day

    def whisper(self):
        #print(self.day,"whisperchat")
        if self.whisper_count == 0:
            self.whisper_count += 1
            return cb.attack(self.atackaction)
        elif self.whisper_count == 1:
            self.whisper_count += 1
            if self.day == 0:
                return cb.comingout(self.index,"WEREWOLF")
            else:
                return cb.skip()
        elif self.whisper_count == 2:
            self.whisper_count += 1
            if self.lastatack == None:
                return cb.over()
            else:
                return 'ATTACKED Agent[' + "{0:02d}".format(self.lastatack) + ']'
        return cb.skip()

    def attack(self):
        #print("attack")
        f = open('attack_action.csv', 'a')
        f.write(str(self.gamecount))
        f.write(",")
        f.write(str(self.atackaction))
        f.write("\n")
        return self.atackaction

    def divine(self):
        #print("divine")
        while True:
            action = random.randint(1,self.playerNum)
            if action!= self.base_info["agentIdx"] and self.statusMap[str(action)] == "ALIVE":
                return action

    def guard(self):
        #print("guard")
        while True:
            action = random.randint(1,self.playerNum)
            if action!= self.base_info["agentIdx"] and self.statusMap[str(action)] == "ALIVE":
                return action

    def vote(self):
        #print("vote")
        if self.islearn:
            return self.voteaction

        else:
            while True:
                action = random.randint(1,self.playerNum)
                if action!= self.base_info["agentIdx"] and self.statusMap[str(action)] == "ALIVE":
                    return action
        

    def dayStart(self):
        #print("dayStart")
        #print("dayStart")
        self.start = time.time()
        self.talk_count = 0
        self.whisper_count = 0
        self.action_flag = False
        self.action = None
        self.voteaction = None
        self.day += 1
        if self.islearn:
            possibleActions = self.possible()
            state = np.concatenate((self.stateEs.reshape([self.playerNum*self.playerNum]),self.stateCo.reshape([self.playerNum*self.playerNum])))
            state = np.concatenate((self.stateVote.reshape([self.playerNum*self.playerNum]),state))
            state = np.concatenate((self.stateTalk.reshape([self.playerNum*self.playerNum]),state))
            state = np.concatenate((self.stateTalkNum.reshape([self.playerNum]),state))
            state = np.concatenate((self.stateIsAlive.reshape([self.playerNum]),state))
            state = np.concatenate(([self.day],state))
            state = np.concatenate(([code[self.role]],state))
            f = open("state.csv","a")
            f.write(str(self.gamecount))
            f.write(",")
            f.write(str(state))
            f.write("\n")
            print("-------------- state -------------------")
            print(state)

            if len(self.stateLog) != 0:
                print("daystart")
                print("--------------" + str(self.gamecount) + "----------------")
                print("--------------" + str(self.day) + "----------------")
                # print(self.stateLog[-1])
                # print("----------------------------------------------")
                self.dqn_talk.update_DQN(self.stateLog[-1], state, self.actionLog[-1], 0, False,self.gamecount)
            self.action,log_action = self.dqn_talk.get_action(possibleActions,self.possiblewolf,state,self.role)
            self.action_flag = True
            #print(self.day,possibleActions,self.possiblewolf)
            #print(self.action)
            if self.role == "WEREWOLF":
                i = 1
                while True:
                    actionIdx = self.action[-i]
                    if actionIdx != self.base_info["agentIdx"] and self.statusMap[str(actionIdx)] == "ALIVE":
                        self.lastatack = self.atackaction
                        self.atackaction = actionIdx
                        i += 1
                        break
                i = 1
                while True:
                    actionIdx = self.action[-i]
                    if actionIdx != self.base_info["agentIdx"] and self.statusMap[str(actionIdx)] == "ALIVE":
                        i += 1
                        self.voteaction = actionIdx
                        break
            elif self.role == "VILLAGER":
                i = 0
                while True:
                    actionIdx = self.action[i]
                    if actionIdx != self.base_info["agentIdx"] and self.statusMap[str(actionIdx)] == "ALIVE":
                        i += 1
                        self.voteaction = actionIdx
                        break
            self.stateLog.append(state)
            self.actionLog.append(log_action)
        return None

    def talk(self):
        #print("talk")
        #print(self.start,time.time())
        #if (time.time() - self.start) >= 0.8:
            #print("time out")
            #return cb.over()
        if not self.islearn:
            return cb.over()

            #print(self.actionLog)
        if self.action_flag:
            if self.role == "VILLAGER":
                if self.talk_count == 0:
                    self.talk_count+=1
                    return cb.estimate(self.action[0],"WEREWOLF")
                    #return 'ESTIMATET Agent[' + "{0:02d}".format(self.action[0]) + '] ' + "WEREWOLF"
                elif self.talk_count == 1:
                    self.talk_count+=1
                    return cb.vote(self.action[0])
                    #return 'VOTE Agent[' + "{0:02d}".format(self.action[0]) + ']'
                elif self.talk_count == 2:
                    self.talk_count+=1
                    if self.day <= 3:
                        return cb.skip()
                    return cb.comingout(self.action[0],"WEREWOLF")
                    #return 'COMMINGOUT Agent[' + "{0:02d}".format(self.action[0]) + '] ' + "WEREWOLF"
                elif self.talk_count == 3:
                    self.talk_count+=1
                    if len(self.action) >= 2:
                        return cb.estimate(self.action[1],"WEREWOLF")
                    #return 'ESTIMATET Agent[' + "{0:02d}".format(self.action[1]) + '] ' + "WEREWOLF"
                elif self.talk_count == 4:
                    self.talk_count+=1
                    if len(self.action) >= 3:
                        return cb.comingout(self.action[-1],"VILLAGER")
                    #return 'COMMINGOUT Agent[' + "{0:02d}".format(self.action[2]) + '] ' + "HUMAN"
                elif self.talk_count == 5:
                    self.talk_count+=1
                    if len(self.action) >= 4:
                        return cb.estimate(self.action[-2],"VILLAGER")
                    #return 'ESTIMATET Agent[' + "{0:02d}".format(self.action[3]) + '] ' + "HUMAN"
                elif self.talk_count == 6:
                    self.talk_count+=1
                    self.action_flag = False
                    return cb.over()

            elif self.role == "WEREWOLF":
                if self.talk_count == 0:
                    self.talk_count+=1
                    if len(self.action) >= 4:
                        return cb.estimate(self.action[-1],"WEREWOLF")
                    #return 'ESTIMATET Agent[' + "{0:02d}".format(self.action[0]) + '] ' + "WEREWOLF"
                elif self.talk_count == 1:
                    self.talk_count+=1
                    if len(self.action) >= 4:
                        return cb.vote(self.action[-1])
                    #return 'VOTE Agent[' + "{0:02d}".format(self.action[0]) + ']'
                elif self.talk_count == 2:
                    self.talk_count+=1
                    if self.day <= 3:
                        return cb.skip()
                    if len(self.action) >= 4:
                        return cb.comingout(self.action[-1],"WEREWOLF")
                    #return 'COMMINGOUT Agent[' + "{0:02d}".format(self.action[0]) + '] ' + "WEREWOLF"
                elif self.talk_count == 3:
                    self.talk_count+=1
                    if len(self.action) >= 3:
                        return cb.estimate(self.action[-2],"WEREWOLF")
                    #return 'ESTIMATET Agent[' + "{0:02d}".format(self.action[1]) + '] ' + "WEREWOLF"
                elif self.talk_count == 4:
                    self.talk_count+=1
                    if self.day <= 3:
                        return cb.skip()
                    return cb.comingout(self.action[1],"VILLAGER")
                    #return 'COMMINGOUT Agent[' + "{0:02d}".format(self.action[2]) + '] ' + "HUMAN"
                elif self.talk_count == 5:
                    self.talk_count+=1
                    if len(self.action) >= 3:
                        return cb.estimate(self.action[2],"VILLAGER")
                    #return 'ESTIMATET Agent[' + "{0:02d}".format(self.action[3]) + '] ' + "HUMAN"
                elif self.talk_count == 6:
                    self.talk_count+=1
                    self.action_flag = False
                    return cb.over()
            else:
                cb.over()
        return cb.skip()

    def calc_reward(self):
        if self.statusMap[str(self.index)] == "ALIVE":
            self.reward = 1
            self.reward_count += 1
        else:
            self.reward = 0

    def Win(self,base_info,diff_data):
        wolf_count, village_count = 0, 0
      
        for i in range(diff_data.shape[0]):
            if "WEREWOLF" in diff_data["text"][i] and self.statusMap[str(i+1)] == "ALIVE":
                wolf_count+=1
            elif "POSSESSED" in diff_data["text"][i] and self.statusMap[str(i+1)] == "ALIVE":
                village_count+=1
            elif "VILLAGER" in diff_data["text"][i] and self.statusMap[str(i+1)] == "ALIVE":
                village_count+=1
            elif "SEER" in diff_data["text"][i] and self.statusMap[str(i+1)]== "ALIVE":
                village_count+=1
        if ((base_info["myRole"] == "WEREWOLF" or base_info["myRole"] == "POSSESSED") and wolf_count >= village_count) or ((base_info["myRole"] == "VILLAGER" or base_info["myRole"] == "SEER") and wolf_count == 0):
            return True
        else:
            return False

    def finish(self):
        self.calc_reward()
                
        if self.Win(self.base_info, self.diff_data):
                self.win_count += 1
        if(self.gamecount % 100 == 0):
            self.time_end = time.time()
            self.spended_time = self.time_end - self.time_sta
            f = open('time_log.csv', 'a')
            f.write(str(self.gamecount))
            f.write(",")
            f.write(str(self.spended_time))
            f.write("\n")
            self.time_sta = time.time()
        if(self.gamecount % 100 == 0):
            f = open('win_rate.csv', 'a')
            f.write(str(self.gamecount))
            f.write(",")
            f.write(str(self.win_count/100))
            f.write("\n")
            self.win_count = 0
        f = open('reward.csv', 'a')
        f.write(str(self.gamecount))
        f.write(",")
        f.write(str(self.reward))
        f.write("\n")
                
        if self.islearn and self.finishCount == 0:
            # self.calc_reward()
            #print(self.reward)
            #print("finish")
            state = np.concatenate((self.stateEs.reshape([self.playerNum*self.playerNum]),self.stateCo.reshape([self.playerNum*self.playerNum])))
            state = np.concatenate((self.stateVote.reshape([self.playerNum*self.playerNum]),state))
            state = np.concatenate((self.stateTalk.reshape([self.playerNum*self.playerNum]),state))
            state = np.concatenate((self.stateTalkNum.reshape([self.playerNum]),state))
            state = np.concatenate((self.stateIsAlive.reshape([self.playerNum]),state))
            state = np.concatenate(([self.day],state))
            state = np.concatenate(([code[self.role]],state))
            print("finish")
            # print(len(self.stateLog[-1]))
            # print(self.stateLog[-1])
            self.dqn_talk.update_DQN(self.stateLog[-1],state,self.actionLog[-1],self.reward,True,self.gamecount)
            pickle.dump(self.gamecount,open("kill_rate_gamecount","wb+"))
            self.finishCount += 1

            self.dqn_talk.finish()

        return None

    def possible(self):
        possibleActions = []
        for i in range(self.playerNum):
            if self.statusMap[str(i+1)] == "ALIVE" and i+1 != self.index:
                possibleActions.append(True)
            else:
                possibleActions.append(False)

        return possibleActions

    def splitText(self, text):
        # print("splitText() start with text", text)
        temp = text.split()
        topic = temp[0]
        if topic == "ESTIMATE":
            target = int(temp[1][6:8])
            role = temp[2]
            return [topic,target,role]
        elif topic == "COMINGOUT":
            target = int(temp[1][6:8])
            role = temp[2]
            return [topic,target,role]
        elif topic == "DIVINATION":
            target = int(temp[1][6:8])
            return [topic,target]
        elif topic == "VOTE":
            target = int(temp[1][6:8])
            return [topic,target]
        elif topic == "ATTACK":
            target = int(temp[1][6:8])
            return [topic,target]
        elif topic == "DIVINED":
            target = int(temp[1][6:8])
            species = temp[2]
            return [topic,target,species]
        elif topic == "IDENTIFIED":
            target = int(temp[1][6:8])
            species = temp[2]
            return [topic,target,species]
        elif topic == "GUARDED":
            target = int(temp[1][6:8])
            return [topic,target]
        elif topic == "VOTED":
            target = int(temp[1][6:8])
            species = temp[2]
            return [topic,target]
        elif topic == "ATTACKED":
            target = int(temp[1][6:8])
            return [topic,target]
        elif topic == "AGREE":
            return [topic]
        elif topic == "DISAGREE":
            return [topic]
        else:
            return []

agent = SampleAgent(myname)

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
