import tensorflow as tf
import numpy as np

class DQNTrain:
    def __init__(self, Q_net, Target_Q_net,  lr=1e-4, discounted_value=0.99):

        self.Q_net = Q_net
        self.Target_Q_net = Target_Q_net

        Q_net_trainable = self.Q_net.get_trainable_variables()

        Target_Q_net_trainable = self.Target_Q_net.get_trainable_variables()

        # assign_operations for Q_net parameter values to Target_Q_net
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_target, v in zip(Target_Q_net_trainable, Q_net_trainable):
                self.assign_ops.append(tf.assign(v_target, v))

        # inputs for train_op
        with tf.variable_scope('train_inp'):
            self.risks = tf.placeholder(dtype=tf.float32, shape=[None], name='risks_batch')



        q_values = self.Q_net.q_values
        target_q_values = self.Target_Q_net.q_values #we should feed the target_Q_net the next states, we are interested in the list of q value for each state
        # instead of the q value for the batch

        y_values = q_target_values(self.risks, target_q_values, discounted_value)

        self.q_loss = tf.reduce_mean((y_values - q_values) ** 2)
        self.train_q_net = tf.train.AdamOptimizer(lr).minimize(self.q_loss, var_list=Q_net_trainable)








    def train_policy(self, obs, next_obs, actions, next_acts, risks):
        loss_1, _ = tf.get_default_session().run([self.q_loss, self.train_q_net], feed_dict={self.Q_net.obs: obs,
                                                               self.Target_Q_net.obs: next_obs,
                                                               self.Q_net.acts: actions,
                                                               self.Target_Q_net.acts: next_acts,
                                                               self.risks: risks,
                                                                  })
        return loss_1


    def update_target_q_network(self):
        # assign policy parameter values to old policy parameters
        return tf.get_default_session().run(self.assign_ops)



def q_target_values(mini_batch_risks, mini_batch_target_Q, discounted_value):

    ys = mini_batch_risks + (1-mini_batch_risks)*discounted_value*mini_batch_target_Q

    return ys
