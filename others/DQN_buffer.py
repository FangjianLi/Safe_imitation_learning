import numpy as np
from collections import deque




class DQN_buffer():
    '''
    Experience Replay Buffer
    '''

    def __init__(self, buffer_size):
        self.obs_buf = deque(maxlen=buffer_size)
        self.rew_buf = deque(maxlen=buffer_size)
        self.act_buf = deque(maxlen=buffer_size)
        self.obs2_buf = deque(maxlen=buffer_size)
        self.done_buf = deque(maxlen=buffer_size)

    def add(self, obs, rew, act, obs2, done):
        # Add a new transition to the buffers
        self.obs_buf.append(obs)
        self.rew_buf.append(rew)
        self.act_buf.append(act)
        self.obs2_buf.append(obs2)
        self.done_buf.append(done)

    def sample_minibatch(self, batch_size):
        # Sample a minibatch of size batch_size
        mb_indices = np.random.randint(len(self.obs_buf), size=batch_size)

        mb_obs = [self.obs_buf[i] for i in mb_indices]
        mb_rew = [self.rew_buf[i] for i in mb_indices]
        mb_act = [self.act_buf[i] for i in mb_indices]
        mb_obs2 = [self.obs2_buf[i] for i in mb_indices]
        mb_done = [self.done_buf[i] for i in mb_indices]

        return mb_obs, mb_rew, mb_act, mb_obs2, mb_done

    def __len__(self):
        return len(self.obs_buf)



class DQN_batch():
    '''
    Experience Replay Buffer
    '''


    def __init__(self, traj_dir):
        self.obs_batch = np.load(traj_dir+'observations.npy')
        self.obs_next_batch = np.load(traj_dir+'observations_next.npy')
        self.act_batch = np.load(traj_dir+'actions.npy')
        self.act_next_batch = np.load(traj_dir+'actions_next.npy')
        self.risk_batch= np.load(traj_dir+'risks.npy')
        self.done_batch = np.load(traj_dir+'done.npy')
        print(len(self.obs_batch))


    def sample_minibatch(self, batch_size):
        # Sample a minibatch of size batch_size
        mb_indices = np.random.randint(len(self.obs_batch), size=batch_size)

        mb_obs = [self.obs_batch[i] for i in mb_indices]
        mb_obs_next = [self.obs_next_batch[i] for i in mb_indices]
        mb_act = [self.act_batch[i] for i in mb_indices]
        mb_act_next = [self.act_next_batch[i] for i in mb_indices]
        mb_risk = [self.risk_batch[i] for i in mb_indices]



        return mb_obs, mb_obs_next, mb_act, mb_act_next, mb_risk

    def __len__(self):
        return len(self.obs_batch)