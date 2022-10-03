import numpy as np


class batch_sampler:
    def __init__(self):
        self.gamma = 0.99
        self.lamba_1 = 0.95
        self.observation_batch_traj = []
        self.action_batch_traj = []
        self.reward_batch_traj = []
        self.value_batch_traj = []

        self.observation_batch_total = []
        self.action_batch_total = []
        self.rtg_batch_total = []
        self.gae_batch_total = []
        self.value_next_batch_total = []
        self.reward_batch_total = []

    def sampler_traj(self, state, action, reward, value):
        self.observation_batch_traj.append(state)
        self.action_batch_traj.append(action)
        self.reward_batch_traj.append(reward)
        self.value_batch_traj.append(value)

    def sampler_total(self, last_value):
        if self.reward_batch_traj:
            self.observation_batch_total.extend(self.observation_batch_traj)
            self.action_batch_total.extend(self.action_batch_traj)
            value_next_batch_traj = self.value_batch_traj[1:]
            value_next_batch_traj.append(last_value)


            rtg = discounted_rewards(self.reward_batch_traj, last_value, self.gamma)
            gae = GAE(self.reward_batch_traj, self.value_batch_traj, last_value, self.gamma, self.lamba_1)


            self.rtg_batch_total.extend(rtg)
            self.gae_batch_total.extend(gae)
            self.value_next_batch_total.extend(value_next_batch_traj)
            self.reward_batch_total.extend(self.reward_batch_traj)

            self.observation_batch_traj = []
            self.action_batch_traj = []
            self.reward_batch_traj = []
            self.value_batch_traj = []

    def sampler_reset(self):
        self.observation_batch_traj = []
        self.action_batch_traj = []
        self.reward_batch_traj = []
        self.value_batch_traj = []

        self.observation_batch_total = []
        self.action_batch_total = []
        self.rtg_batch_total = []
        self.gae_batch_total = []
        self.value_next_batch_total = []
        self.reward_batch_total = []

    def sampler_get(self):
        normalized_gae = (self.gae_batch_total - np.mean(self.gae_batch_total)) / (np.std(self.gae_batch_total) + 1e-10)
        return self.observation_batch_total, self.action_batch_total, self.rtg_batch_total, normalized_gae, \
               self.value_next_batch_total, self.reward_batch_total

    def sampler_get_parallel(self):
        #normalized_gae = (self.gae_batch_total - np.mean(self.gae_batch_total)) / (np.std(self.gae_batch_total) + 1e-10)
        return self.observation_batch_total, self.action_batch_total, self.rtg_batch_total, self.gae_batch_total, \
               self.value_next_batch_total, self.reward_batch_total


def GAE(rews, v, v_last, gamma=0.99, lam=0.95):
    '''
    Generalized Advantage Estimation
    '''
    assert len(rews) == len(v)
    vs = np.append(v, v_last)
    delta = np.array(rews) + gamma * vs[1:] - vs[:-1]
    gae_advantage = discounted_rewards(delta, 0, gamma * lam)
    return gae_advantage


def discounted_rewards(rews, last_sv, gamma):
    '''
    Discounted reward to go

    Parameters:
    ----------
    rews: list of rewards
    last_sv: value of the last state
    gamma: discount value
    '''
    rtg = np.zeros_like(rews, dtype=np.float32)
    rtg[-1] = rews[-1] + gamma * last_sv
    for i in reversed(range(len(rews) - 1)):
        rtg[i] = rews[i] + gamma * rtg[i + 1]
    return rtg