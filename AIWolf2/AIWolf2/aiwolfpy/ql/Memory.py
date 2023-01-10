class Memory:
    def __init__(self,max_size=4500000):
        self.buffer = deque(maxlen=max_size)

    def add(self,experience):
        self.buffer.append(experience)

    def sample(self,batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),size=batch_size,replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)
