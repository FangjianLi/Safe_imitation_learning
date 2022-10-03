import numpy as np
from collections import deque



# we can let this buffer directly generate the panlty item 
class Extra_penalty_buffer():
    '''
    Experience Replay Buffer
    '''

    def __init__(self, buffer_size, threshold_high, threshold_low):
        self.obs_buf_safe = deque(maxlen=buffer_size)
        self.act_buf_safe = deque(maxlen=buffer_size)
        self.safety_q_buf_safe = deque(maxlen=buffer_size)
        self.obs_buf_unsafe = deque(maxlen=buffer_size)
        self.act_buf_unsafe = deque(maxlen=buffer_size)
        self.safety_q_buf_unsafe = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low

    def add(self, obs_batch_total, act_batch_total, safety_q_batch):
        obs_batch_total = np.squeeze(obs_batch_total)
        act_batch_total = np.squeeze(act_batch_total)
        safety_batch_total = np.squeeze(safety_q_batch)
        # Add a new transition to the buffers
        for obs, act, safety_q in zip(obs_batch_total, act_batch_total, safety_batch_total):
            if safety_q > self.threshold_high:
                self.obs_buf_unsafe.append(obs)
                self.act_buf_unsafe.append(act)
                self.safety_q_buf_unsafe.append(safety_q)
            elif safety_q < self.threshold_low:
                self.obs_buf_safe.append(obs)
                self.act_buf_safe.append(act)
                self.safety_q_buf_safe.append(safety_q)


    def sample_minibatch(self, batch_size):
        # Sample a minibatch of size batch_size
        mb_indices = np.random.randint(len(self.obs_buf_safe), size=int(batch_size/2))

        mb_obs_safe = [self.obs_buf_safe[i] for i in mb_indices]
        mb_act_safe = [self.act_buf_safe[i] for i in mb_indices]
        mb_safety_q_safe = [self.safety_q_buf_safe[i] for i in mb_indices]
        
        mb_indices = np.random.randint(len(self.obs_buf_unsafe), size=int(batch_size/2))
        
        mb_obs_unsafe = [self.obs_buf_unsafe[i] for i in mb_indices]
        mb_act_unsafe = [self.act_buf_unsafe[i] for i in mb_indices]
        mb_safety_q_unsafe = [self.safety_q_buf_unsafe[i] for i in mb_indices]

        return np.concatenate((mb_obs_unsafe, mb_obs_safe)), np.concatenate((mb_act_unsafe, mb_act_safe)), np.concatenate((mb_safety_q_unsafe, mb_safety_q_safe))
    
    
    def generate_the_penalty_item(self, batch_size, Discriminator_1):
        if len(self.obs_buf_safe) == len(self.obs_buf_unsafe) == self.buffer_size:
            # print("yes")
            mb_obs, mb_act, mb_safety_q = self.sample_minibatch(batch_size)
            mb_safety_q = np.reshape(mb_safety_q, (-1,1))
            reward_1 = np.squeeze(Discriminator_1.get_rewards_ep(mb_obs, mb_act, mb_safety_q))
            sort_index = np.argsort(reward_1)
            # print(reward_1)
            # print(sort_index)
            wrong_sort_count = 0
            for index in sort_index[:int(batch_size/2)]:
                if index>=batch_size/2:
                    wrong_sort_count += 1

            return 2*wrong_sort_count/batch_size
        else:
            return 0
        
        
        

    def __len__(self):
        return len(self.obs_buf_safe) + len(self.obs_buf_unsafe)