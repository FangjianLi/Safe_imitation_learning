import tensorflow as tf
import numpy as np
import gym


class Q_net:
    def __init__(self, name: str, env, units_p, activation_p=tf.nn.relu, activation_p_last_d=None):
        """
        :param name: string
        :param env: gym env
        """

        try:
            ob_space_shape = env.observation_space_shape
        except AttributeError:
            ob_space_shape = env.observation_space.shape


        act_dim = env.action_space.n



        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space_shape), name='obs')


            self.acts = tf.placeholder(dtype=tf.int32, shape=[None], name='acts')



            initializer = None




            with tf.variable_scope('q_net'):
                layer_p = self.obs
                for l_p in units_p:
                    layer_p = tf.layers.dense(inputs=layer_p, units=l_p, activation=activation_p,
                                              kernel_initializer=initializer)

                self.act_q_value = tf.layers.dense(inputs=layer_p, units=act_dim, activation=activation_p_last_d,
                                                 kernel_initializer=initializer)







            self.q_values = self.act_q_value * tf.one_hot(indices=self.acts, depth=act_dim)

            self.q_values = tf.reduce_sum(self.q_values, axis=1) # it can output the the q_values of the batch. still, it is a vector




            self.scope = tf.get_variable_scope().name






    def get_act_q_values(self, obs):
        return tf.get_default_session().run(self.act_q_value, feed_dict={self.obs: obs})

    def get_q_values(self, obs, acts):
        return tf.get_default_session().run(self.q_values, feed_dict={self.obs: obs, self.acts: acts})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


