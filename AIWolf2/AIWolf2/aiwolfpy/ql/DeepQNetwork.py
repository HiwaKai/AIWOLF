import numpy as np
import tensorflow as tf
import os
import pickle
import collections
import random

MAXSIZE = 4500000
INITIAL_REPLAY = 200
GAMMA = 0.99
EXPLORE = 180000
BATCH_SIZE = 128
UPDATE_STEP = 8
FINALE_EPSILON = 0.2
EPSILON = 0.9
class DeepQNetwork:
    def __init__(self,network_type_role,network_type_act,learning_rate = 0.001,state_size = 101,hidden1_size = 128,hidden2_size = 64,hidden3_size = 64,hidden4_size = 32):
        self.actions = 15
        self.input_size = 932
        self.role = network_type_role
        self.act = network_type_act
        self.epsilon = EPSILON
        if (os.path.exists("kill_rate_replayMemory")):
            self.buffer = pickle.load(open("kill_rate_replayMemory","rb+"))
        else:
            self.buffer = collections.deque(maxlen = MAXSIZE)
        if (os.path.exists("kil_rate_step")):
            self.step = pickle.load(open("kil_rate_step","rb+"))
        else:
            self.step = 0
        if (os.path.exists("kill_rate_states")):
            self.states = pickle.load(open("kill_rate_states","rb+"))
            print("ok")
        else:
            self.states = set()
        if(os.path.exists("kill_rate_loss")):
            self.lossValue = pickle.load(open("kill_rate_loss", 'rb+'))
        else:
            self.lossValue = float('inf')
        """
        if (os.path.exists(self.role + "_" + self.act + "_replayMemory")):
            self.buffer = pickle.load(open(self.role + "_" + self.act + "_replayMemory","rb+"))
        else:
            self.buffer = collections.deque(maxlen = MAXSIZE)
        if (os.path.exists(self.role + "_" + self.act + "_step")):
            self.step = pickle.load(open(self.role + "_" + self.act + "_step","rb+"))
        else:
            self.step = 0
        if (os.path.exists(self.role + "_" + self.act + "_states")):
            self.states = pickle.load(open(self.role + "_" + self.act + "_states","rb+"))
            print("ok")
        else:
            self.states = set()
        if(os.path.exists(self.role + '_' + self.act + '_loss')):
            self.lossValue = pickle.load(open(self.role + '_' + self.act + '_loss', 'rb+'))
        else:
            self.lossValue = float('inf')
        """
        if network_type_act == "talk":


            self.x = tf.placeholder(tf.float32, [None, self.input_size])

            self.x_flat = tf.reshape(self.x,[-1,self.input_size])

            fc1 = tf.contrib.layers.fully_connected(inputs=self.x_flat, num_outputs=128, activation_fn=tf.nn.relu)
            # fc1 = tf.nn.dropout(fc1, self.dropout)

            fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=64, activation_fn=tf.nn.relu)
            # fc2 = tf.nn.dropout(fc2, self.dropout)

            fc3 = tf.contrib.layers.fully_connected(inputs=fc2,num_outputs=32,activation_fn=tf.nn.relu)
            #932,128,64,32,15のDNN
            self.Qnn = tf.contrib.layers.fully_connected(inputs=fc3, num_outputs=self.actions, activation_fn=tf.nn.sigmoid)

            self.actionInput = tf.placeholder(tf.float32, [None, self.actions])
            self.yInput = tf.placeholder(tf.float32, [None,])
            #qnnにself.actionInputとの積を入れ，その総和

            QOfAction = tf.reduce_sum(tf.multiply(self.Qnn, self.actionInput), reduction_indices = 1)
            #総和と実際の行動との誤差の平均
            self.loss = tf.reduce_mean(tf.square(self.yInput - QOfAction))
            #lossはslef.QnnにactionInput()
            self.optimizer = tf.train.RMSPropOptimizer(0.0001)
            self.training = self.optimizer.minimize(self.loss)

            self.saver = tf.train.Saver()

            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

            checkpoint = tf.train.latest_checkpoint("saved_networks_kill_rate/")

            if checkpoint:
                self.saver.restore(self.session, checkpoint)
                print("Model was loaded successfully: ", checkpoint)
            else:
                print("Network weights could not be found!")
                # print("createQNetwork() end")


    def sample(self,batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),size=batch_size,replace=False)
        return [self.buffer[ii] for ii in idx]


    def update_DQN(self,state,nextobservationaction,action,reward,terminal,gamecount):
        self.buffer.append((state,action,reward,nextobservationaction,terminal))
        self.states.add(str(state))
        self.states.add(str(nextobservationaction))
        if len(self.buffer) > MAXSIZE:
            self.buffer.popleft()
        if self.step > INITIAL_REPLAY and (self.step - INITIAL_REPLAY) % UPDATE_STEP == 0:
            self.replay()
        self.step += 1
        if self.step % 100 == 0:
            self.saver.save(self.session,"saved_networks_kill_rate/network_dqn",global_step=self.step)
            #self.saver.save(self.session,"saved_networks_" + str(self.role) + "_" + str(self.act) + "/network_dqn", global_step = self.step)

        f = open("kill_rate_log.txt","a")

        if self.step <= INITIAL_REPLAY:
            print("step :",self.step," gamecount :",gamecount," states:",len(self.states))
            f.write(str("step :"))
            f.write(str(self.step))
            f.write(" gamecount :")
            f.write(str(gamecount))
            f.write(" states:")
            f.write(str(len(self.states)))
            f.write("\n")
            #f.write(str("step :",self.step," gamecount :",gamecount," states :",len(self.states)))
        else:
            print("step :",self.step," gamecount :",gamecount," states:",len(self.states),"loss",self.lossValue,"epsilon",self.epsilon)
            f.write(str("step :"))
            f.write(str(self.step))
            f.write(" gamecount :")
            f.write(str(gamecount))
            f.write(" states:")
            f.write(str(len(self.states)))
            f.write(" loss :")
            f.write(str(self.lossValue))
            f.write("\n")
            #f.write(str("step :",self.step," gamecount :",gamecount," states :",len(self.states),"loss :",self.lossValue))
        f.close()
        #epsilon

    def replay(self):
        minibatch = random.sample(self.buffer,BATCH_SIZE)
        stateBatch = [data[0] for data in minibatch]
        actionBatch = [data[1] for data in minibatch]
        rewardBatch = [data[2] for data in minibatch]
        nextStateBatch = [data[3] for data in minibatch]

        yBatch = []
        QValueBatch = self.Qnn.eval(session=self.session,feed_dict = {self.x:nextStateBatch})
        for i in range(0,BATCH_SIZE):
            terminal = minibatch[i][4]
            #finishの時はelse,talkの時はif
            if terminal:
                yBatch.append(rewardBatch[i])
            else:
                yBatch.append(rewardBatch[i] + GAMMA * np.max(QValueBatch[i]))

        self.training.run(session=self.session,feed_dict={self.yInput:yBatch,self.actionInput:actionBatch,self.x:stateBatch})
        self.lossValue = self.loss.eval(session=self.session,feed_dict={self.yInput:yBatch,self.actionInput:actionBatch,self.x:stateBatch})

    def possible(self,action,possibleActions):
        for i in list(reversed(range(15))):
            if not possibleActions[int(action[i]) -1]:
                action = np.delete(action,i)
        return action

    def possible2(self,action,possibleActions,possibleAction_wolf):
        for i in list(reversed(range(15))):
            if not possibleActions[int(action[i])-1] or not possibleAction_wolf[int(action[i])-1]:
                action = np.delete(action,i)
        return action

    def get_action(self,possibleActions,possibleAction_wolf,state,role):
        if self.epsilon > FINALE_EPSILON and self.step > INITIAL_REPLAY:
            self.epsilon = EPSILON - self.step * (EPSILON - FINALE_EPSILON) / EXPLORE

        if self.epsilon >= np.random.uniform(0,1):
            retTargetQs = self.Qnn.eval(session=self.session,feed_dict = {self.x:[state]})[0]
            action = np.array(retTargetQs).argsort()[::-1]
        else:
            action = np.random.uniform(0,1,15)
            action = np.array(action).argsort()[::-1]
        action += 1
        if role == "VILLAGER":
            action_index = self.possible(action,possibleActions)
        elif role == "WEREWOLF":
            action_index = self.possible2(action,possibleActions,possibleAction_wolf)
        return action_index, action

    def finish(self):
        #if self.step % 1000 == 0:
        pickle.dump(self.buffer,open("kill_rate_replayMemory","wb+"))
        pickle.dump(self.states,open("kill_rate_states","wb+"))
        pickle.dump(self.step,open("kill_rate_step","wb+"))
        pickle.dump(self.lossValue,open("kill_rate_loss","wb+"))
        #pickle.dump(self.buffer,open(self.role + "_" + self.act + "_replayMemory","wb+"))
        #pickle.dump(self.states,open(self.role + "_" + self.act + "_states","wb+"))
        #pickle.dump(self.step,open(self.role + "_" + self.act + "_step","wb+"))
        #pickle.dump(self.lossValue,open(self.role + "_" + self.act + "_loss","wb+"))
