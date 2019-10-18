import numpy as np
from dynamics import Dynamics 




# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id = None):
    env = Dynamics(dataset = 'mnist', cls = 'CLS_mnist')
    env = NormalizedEnv(env)
    return env


class NormalizedEnv(Dynamics):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(dataset = 'mnist', cls = 'CLS_mnist')
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)
