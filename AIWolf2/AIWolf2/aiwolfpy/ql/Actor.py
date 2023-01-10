import numpy as np

class Actor:
    def get_action(self,state,episode,mainQN):
        epsilon = 0.2 + 0.9 / (1 + episode)

        if epsilon <= np.random.uniform(0,1):
            retTargetQs = mainQN.model.predict(state)[0]
            action = np.argmax(retTargetQs)
        else:
            # action = np.random.randint(1,16)
            action = np.random.randint(1,6)

        return action

    def get_action_talk(self,state,episode,mainQN):
        epsilon = 0.2 + 0.9 / (1 + episode)

        if epsilon <= np.random.unifornm(0,1):
            retTargetQs = mainQN.modelTalk.predict(state)[0]
            action = np.argmax(retTargetQs)

        else:
            action = np.random.randint(1,150)

        return action
