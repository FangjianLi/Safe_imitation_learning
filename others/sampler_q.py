import numpy as np


# here we just need to record the observation, action,  next observation,  next action, done or not
# also the


class batch_sampler_Q:
    def __init__(self):
        self.observation_batch_traj = []
        self.action_batch_traj = []
        self.reward_batch_traj = []
        self.risk_batch_traj = []
        self.done_batch_traj = []

        self.observation_batch_total = []
        self.observation_next_batch_total = []
        self.action_batch_total = []
        self.action_next_batch_total = []
        self.reward_batch_total = []
        self.risk_batch_total = []
        self.done_batch_total = []

    def sampler_traj(self, state, action, reward, risk, done):
        self.observation_batch_traj.append(state)
        self.action_batch_traj.append(action)
        self.reward_batch_traj.append(reward)
        self.risk_batch_traj.append(risk)
        self.done_batch_traj.append(done)

    def sampler_total(self, next_state, next_action):

        if self.reward_batch_traj:

            self.observation_batch_total.extend(self.observation_batch_traj)  # we should not extend it, instead we need to append it

            observation_next_batch_traj = self.observation_batch_traj[1:]
            observation_next_batch_traj.append(next_state)
            self.observation_next_batch_total.extend(observation_next_batch_traj)

            self.action_batch_total.extend(self.action_batch_traj)

            action_next_batch_traj = self.action_batch_traj[1:]
            action_next_batch_traj.append(next_action)
            self.action_next_batch_total.extend(action_next_batch_traj)
            self.reward_batch_total.extend(self.reward_batch_traj)

            self.risk_batch_total.extend(self.risk_batch_traj)
            self.done_batch_total.extend(self.done_batch_traj)

            self.observation_batch_traj = []
            self.action_batch_traj = []
            self.reward_batch_traj = []
            self.risk_batch_traj = []
            self.done_batch_traj = []

    def sampler_reset(self):
        self.observation_batch_traj = []
        self.action_batch_traj = []
        self.reward_batch_traj = []
        self.risk_batch_traj = []
        self.done_batch_traj = []

        self.observation_batch_total = []
        self.observation_next_batch_total = []
        self.action_batch_total = []
        self.action_next_batch_total = []
        self.reward_batch_total = []
        self.risk_batch_total = []
        self.done_batch_total = []

    def sampler_get(self):
        return self.observation_batch_total, self.observation_next_batch_total,  self.action_batch_total, self.action_next_batch_total, self.reward_batch_total, self.risk_batch_total,  self.done_batch_total

    def sampler_get_parallel(self):
        # normalized_gae = (self.gae_batch_total - np.mean(self.gae_batch_total)) / (np.std(self.gae_batch_total) + 1e-10)
        return self.observation_batch_total, self.observation_next_batch_total,  self.action_batch_total, self.action_next_batch_total, self.reward_batch_total, self.risk_batch_total,  self.done_batch_total


