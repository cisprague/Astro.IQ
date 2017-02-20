class TabularQAgent(object):
    def __init__(self, environment):
        self.env       = environment
        self.obs_space = self.env.obs_space
        self.act_space = self.env.act_space
        self.act_n     = self.env.act_space.n
        self.config = {"init_mean"     : 0.0,
                       "init_std"      : 0.0,
                       "learning_rate" : 0.1,
                       "eps"           : 0.05,
                       "discount"      : 0.95,
                       "n_iter"        : 10000}
        self.q = defaultdict(
        lambda: self.config["init_std"]*randn(self.action_n) \
        + self.config["init_mean"])

    def act(self, observation):
        if random() > self.config['eps']:
            action = argmax(self.q[observation])
        else:
            action = self.act_space.sample()
        return action

    def learn(self):
        for t in range(config["n_iter"]):
            action            = self.act(obs)
            obs, reward, done = env.step(action)
            future            = 0.0
            if not done:
                future = max(q[obs2])
            q[obs][action] -=                            \
            self.config["learning_rate"]*(q[obs][action] \
            - reward                                     \
            - config["discount"]*future)
