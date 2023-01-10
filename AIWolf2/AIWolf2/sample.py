#!/usr/bin/env python

# This sample script connects to the AIWolf server, but
# does not do anything else. It will choose itself as the
# target for any actions requested by the server, (voting,
# attacking ,etc) forcing the server to choose a random target.
import logging
from logging import getLogger, StreamHandler, Formatter, FileHandler
import aiwolfpy
import argparse

# name
myname = 'sample_python'

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


class SampleAgent(object):
    
    def __init__(self):
        # my name
        self.base_info = dict()
        self.game_setting = dict()
        self.gamecount = 0
        self.win_count = 0

    def getName(self):
        return self.my_name
    
    # new game (no return)
    def initialize(self, base_info, diff_data, game_setting):
        self.base_info = base_info
        self.diff_data = diff_data
        self.game_setting = game_setting
        self.gamecount += 1
        
    # new information (no return)
    def update(self, base_info, diff_data, request):
        self.base_info = base_info
        self.diff_data = diff_data
        self.statusMap = base_info["statusMap"]
        
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
        return self.base_info['agentIdx']

    def attack(self):
        return self.base_info['agentIdx']

    def divine(self):
        return self.base_info['agentIdx']

    def guard(self):
        return self.base_info['agentIdx']

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
    # Finish (no return)
    def finish(self):
        if self.Win(self.base_info, self.diff_data):
                self.win_count += 1
        if(self.gamecount % 100 == 0):
            f = open('win_rate_sample.csv', 'a')
            f.write(str(self.gamecount))
            f.write(",")
            f.write(str(self.win_count/100))
            f.write("\n")
            self.win_count = 0
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