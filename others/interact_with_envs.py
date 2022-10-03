import numpy as np
from others.sampler_q import batch_sampler_Q
from others.sampler_v_CBF import batch_sampler_CBF
from others.sampler_v2 import batch_sampler
from others.utils import StructEnv_Highway, StructEnv_Highway_Q, CBF_sample_action
from others.utils import StructEnv_AIRL_Highway
from network_models.policy_net_continuous_discrete import Policy_net
from network_models.AIRL_net_discriminator_blend import Discriminator
from network_models.AIRL_net_discriminator_blend_CBF import Discriminator_CBF
from network_models.AIRL_net_discriminator_blend_CBF_total import Discriminator_CBF_total
from network_models.q_net import Q_net
import ray
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import gym
import customized_highway_env

ray.init()


@ray.remote
def test_function_Highway(args_envs, network_values, discrete_env_check, EPISODE_LENGTH, i, units_p_i, units_v_i,
                          arg_ovc_i, arg_vc_i, arg_s_fre, arg_p_fre):
    import customized_highway_env

    tf.autograph.set_verbosity(
        0, alsologtostdout=False
    )
    tf.compat.v1.logging.set_verbosity(
        tf.compat.v1.logging.ERROR
    )
    tf.reset_default_graph()
    units_p = units_p_i
    units_v = units_v_i

    sampler = batch_sampler()

    env = StructEnv_Highway(gym.make(args_envs))
    env.config["observation"]["vehicles_count"] = arg_ovc_i
    env.config['vehicles_count'] = arg_vc_i
    env.config["duration"] = 100
    env.config["simulation_frequency"] = arg_s_fre
    env.config["policy_frequency"] = arg_p_fre
    env.reset_0()

    env.seed(10*i)

    Policy_a = Policy_net('Policy_a_{}'.format(i), env, units_p, units_v)
    net_param = Policy_a.get_trainable_variables()
    net_para_values = network_values

    net_operation = []
    for i in range(len(net_param)):
        net_operation.append(tf.assign(net_param[i], net_para_values[i]))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(net_operation)

        x1 = sess.run(net_param)
        episode_length = 0

        render = False
        sampler.sampler_reset()
        reward_episode_counter = []

        while True:

            episode_length += 1
            act, v_pred = Policy_a.act(obs=[env.obs_a])
            # print("iteration: {}, episode: {}".format(iteration, episode_length))

            if discrete_env_check:
                act = np.asscalar(act)
            else:
                act = np.reshape(act, env.action_space.shape)

            v_pred = np.asscalar(v_pred)

            next_obs, reward, done, info = env.step(act)
            # env.render()

            # reward_episode += reward
            sampler.sampler_traj(env.obs_a.copy(), act, reward, v_pred)

            if render:
                env.render()

            if done:
                # next state of terminate state has 0 state value
                sampler.sampler_total(0)
                reward_episode_counter.append(env.get_episode_reward())
                env.reset()
                # reward_episode = 0
            else:
                env.obs_a = next_obs.copy()

            if episode_length >= EPISODE_LENGTH:
                last_value = np.asscalar(Policy_a.get_value([next_obs]))
                sampler.sampler_total(last_value)
                env.reset()
                break

    return sampler.sampler_get_parallel(), np.mean(reward_episode_counter)




@ray.remote
def test_function_Highway_CBF(args_envs, EPISODE_LENGTH, i):
    import customized_highway_env

    def check_risk(masked_state):
        vehicle_length = 5
        longi_margin = 1.5  # 1.2
        vehicle_width = 2
        lateral_margin = 0.5  # 0.2

        # separate it
        states = np.reshape(masked_state.copy(), (-1, 5))
        # counter-normalized
        states[:, 1] *= 150
        states[:, 2] *= 16

        risk_indicators = np.zeros(len(states) - 1)

        for i in range(1, len(states)):
            if abs(states[i, 1]) < vehicle_length + longi_margin and abs(states[i, 2]) < vehicle_width + lateral_margin:
                risk_indicators[i - 1] = 1

        presence = np.array(states[1:, 0])
        risk = max(presence * risk_indicators)

        return risk

    sampler = batch_sampler_Q()

    env = StructEnv_Highway_Q(gym.make(args_envs))
    env.config["observation"]["vehicles_count"] = 7
    env.config['vehicles_count'] = 20
    env.config["duration"] = 100
    env.config["policy_frequency"] = 2

    _, info = env.reset_0()

    env.seed(10*i)

    episode_length = 0

    render = False
    sampler.sampler_reset()
    reward_episode_counter = []

    while True:

        episode_length += 1

        act = CBF_sample_action(info, 0.05)

        next_obs, reward, done, info = env.step(act)

        risk_o = check_risk(env.obs_a.copy())

        sampler.sampler_traj(env.obs_a.copy(), act, reward, risk_o, float(done))

        if render:
            env.render()

        if done or risk_o > 0:
            next_act = CBF_sample_action(info, 0.05)
            sampler.sampler_total(next_obs, next_act)
            reward_episode_counter.append(env.get_episode_reward())
            _, info = env.reset()
        else:
            env.obs_a = next_obs.copy()

        if episode_length >= EPISODE_LENGTH:
            next_act = CBF_sample_action(info, 0.05)
            sampler.sampler_total(next_obs, next_act)
            env.reset()
            break

    return sampler.sampler_get_parallel(), np.mean(reward_episode_counter)


@ray.remote
def AIRL_test_function_Highway(args_envs, network_policy_values, network_discrim_values, discrete_env_check,
                               EPISODE_LENGTH, i, units_p_i, units_v_i, lr_d, num_batches):
    import customized_highway_env

    tf.autograph.set_verbosity(
        0, alsologtostdout=False
    )
    tf.compat.v1.logging.set_verbosity(
        tf.compat.v1.logging.ERROR
    )
    tf.reset_default_graph()
    units_p = units_p_i
    units_v = units_v_i

    sampler = batch_sampler_CBF()

    env = StructEnv_AIRL_Highway(gym.make(args_envs))
    env.config["observation"]["vehicles_count"] = 7
    env.config['vehicles_count'] = 20
    env.config["duration"] = 100
    env.config["policy_frequency"] = 2
    env.reset_0()

    env.seed(10 * i)

    Policy_a = Policy_net('Policy_a_{}'.format(i), env, units_p, units_v)
    Discrim_a = Discriminator('Discriminator_a_{}'.format(i), env, lr_d, num_batches)

    network_policy_param = Policy_a.get_trainable_variables()
    network_discrim_param = Discrim_a.get_trainable_variables()

    network_policy_operation = []
    for i in range(len(network_policy_param)):
        network_policy_operation.append(tf.assign(network_policy_param[i], network_policy_values[i]))

    network_discrim_operation = []
    for i in range(len(network_discrim_param)):
        network_discrim_operation.append(tf.assign(network_discrim_param[i], network_discrim_values[i]))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(network_policy_operation)
        sess.run(network_discrim_operation)

        # x1 = sess.run(net_param)
        episode_length = 0

        render = False
        sampler.sampler_reset()
        reward_episode_counter = []
        reward_episode_counter_airl = []

        while True:

            episode_length += 1
            act, v_pred = Policy_a.act(obs=[env.obs_a])

            if discrete_env_check:
                act = np.asscalar(act)
            else:
                act = np.reshape(act, env.action_space.shape)

            v_pred = np.asscalar(v_pred)

            next_obs, reward, done, info = env.step(act)

            agent_sa_ph = sess.run(Policy_a.act_probs, feed_dict={Policy_a.obs: [env.obs_a.copy()],
                                                                  Policy_a.acts: [act]})
            reward_a = np.asscalar(Discrim_a.get_rewards([env.obs_a.copy()], [act], agent_sa_ph))
            env.step_airl(reward_a)

            sampler.sampler_traj(env.obs_a.copy(), act, reward_a, v_pred, float(info["crashed"]))

            if render:
                env.render()

            if done:
                # next state of terminate state has 0 state value
                sampler.sampler_total(0)
                reward_episode_counter.append(env.get_episode_reward())
                reward_episode_counter_airl.append(env.get_episode_reward_airl())
                env.reset()
                # reward_episode = 0
            else:
                env.obs_a = next_obs.copy()

            if episode_length >= EPISODE_LENGTH:
                last_value = np.asscalar(Policy_a.get_value([next_obs]))
                sampler.sampler_total(last_value)
                env.reset()
                break

    return sampler.sampler_get_parallel(), np.mean(reward_episode_counter), np.mean(reward_episode_counter_airl)


@ray.remote
def AIRL_Sampling_Highway_CBF(args_envs, network_policy_values, network_discrim_values, discrete_env_check,
                              EPISODE_LENGTH, i, units_p_i, units_v_i, lr_d, num_batches, arg_q_units):
    import customized_highway_env

    tf.autograph.set_verbosity(
        0, alsologtostdout=False
    )
    tf.compat.v1.logging.set_verbosity(
        tf.compat.v1.logging.ERROR
    )
    tf.reset_default_graph()
    units_p = units_p_i
    units_v = units_v_i

    sampler = batch_sampler_CBF()

    env = StructEnv_AIRL_Highway(gym.make(args_envs))
    env.config["observation"]["vehicles_count"] = 7
    env.config['vehicles_count'] = 20
    env.config["duration"] = 100
    env.config["policy_frequency"] = 2
    env.reset_0()

    env.seed(10 * i)

    Policy_a = Policy_net('Policy_a_{}'.format(i), env, units_p, units_v)
    Discrim_a = Discriminator_CBF('Discriminator_a_{}'.format(i), env, lr_d, num_batches)
    # we need also to build the
    Q_net_a = Q_net('safety_critic_a_{}'.format(i), env, arg_q_units, activation_p=tf.nn.relu,
                    activation_p_last_d=tf.nn.sigmoid)

    network_policy_param = Policy_a.get_trainable_variables()
    network_discrim_param = Discrim_a.get_trainable_variables()
    network_Q_param = Q_net_a.get_trainable_variables()

    network_Q_values = np.load("trained_models/CBF_guided_safety_critic/Q_para.npy", allow_pickle=True)

    network_policy_operation = []
    for i in range(len(network_policy_param)):
        network_policy_operation.append(tf.assign(network_policy_param[i], network_policy_values[i]))

    network_discrim_operation = []
    for i in range(len(network_discrim_param)):
        network_discrim_operation.append(tf.assign(network_discrim_param[i], network_discrim_values[i]))

    network_Q_operation = []
    for i in range(len(network_Q_param)):
        network_Q_operation.append(tf.assign(network_Q_param[i], network_Q_values[i]))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(network_policy_operation)
        sess.run(network_discrim_operation)
        sess.run(network_Q_operation)

        episode_length = 0

        render = False
        sampler.sampler_reset()
        reward_episode_counter = []
        reward_episode_counter_airl = []

        while True:

            episode_length += 1
            act, v_pred = Policy_a.act(obs=[env.obs_a])

            if discrete_env_check:
                act = np.asscalar(act)
            else:
                act = np.reshape(act, env.action_space.shape)

            v_pred = np.asscalar(v_pred)

            next_obs, reward, done, info = env.step(act)

            agent_sa_ph = sess.run(Policy_a.act_probs, feed_dict={Policy_a.obs: [env.obs_a.copy()],
                                                                  Policy_a.acts: [act]})
            agent_safety_q = Q_net_a.get_q_values([env.obs_a.copy()], [act])

            reward_a = np.asscalar(Discrim_a.get_rewards([env.obs_a.copy()], [act], agent_sa_ph, [agent_safety_q]))
            env.step_airl(reward_a)

            # reward_episode += reward
            sampler.sampler_traj(env.obs_a.copy(), act, reward_a, v_pred, float(info["crashed"]))

            if render:
                env.render()

            if done:
                # next state of terminate state has 0 state value
                sampler.sampler_total(0)
                reward_episode_counter.append(env.get_episode_reward())
                reward_episode_counter_airl.append(env.get_episode_reward_airl())
                env.reset()
            else:
                env.obs_a = next_obs.copy()

            if episode_length >= EPISODE_LENGTH:
                last_value = np.asscalar(Policy_a.get_value([next_obs]))
                sampler.sampler_total(last_value)
                env.reset()
                break

    return sampler.sampler_get_parallel(), np.mean(reward_episode_counter), np.mean(reward_episode_counter_airl)


@ray.remote
def AIRL_Sampling_Highway_CBF_total(args_envs, network_policy_values, network_discrim_values, discrete_env_check,
                                    EPISODE_LENGTH, i, units_p_i, units_v_i, lr_d, num_batches, arg_q_units):
    import customized_highway_env

    tf.autograph.set_verbosity(
        0, alsologtostdout=False
    )
    tf.compat.v1.logging.set_verbosity(
        tf.compat.v1.logging.ERROR
    )
    tf.reset_default_graph()
    units_p = units_p_i
    units_v = units_v_i

    sampler = batch_sampler_CBF()

    env = StructEnv_AIRL_Highway(gym.make(args_envs))
    env.config["observation"]["vehicles_count"] = 7
    env.config['vehicles_count'] = 20
    env.config["duration"] = 100
    env.config["policy_frequency"] = 2
    env.reset_0()

    env.seed(10 * i)

    Policy_a = Policy_net('Policy_a_{}'.format(i), env, units_p, units_v)
    Discrim_a = Discriminator_CBF_total('Discriminator_a_{}'.format(i), env, lr_d, num_batches)
    # we need also to build the
    Q_net_a = Q_net('safety_critic_a_{}'.format(i), env, arg_q_units, activation_p=tf.nn.relu,
                    activation_p_last_d=tf.nn.sigmoid)

    network_policy_param = Policy_a.get_trainable_variables()
    network_discrim_param = Discrim_a.get_trainable_variables()
    network_Q_param = Q_net_a.get_trainable_variables()

    network_Q_values = np.load("trained_models/CBF_guided_safety_critic/Q_para.npy", allow_pickle=True)

    network_policy_operation = []
    for i in range(len(network_policy_param)):
        network_policy_operation.append(tf.assign(network_policy_param[i], network_policy_values[i]))

    network_discrim_operation = []
    for i in range(len(network_discrim_param)):
        network_discrim_operation.append(tf.assign(network_discrim_param[i], network_discrim_values[i]))

    network_Q_operation = []
    for i in range(len(network_Q_param)):
        network_Q_operation.append(tf.assign(network_Q_param[i], network_Q_values[i]))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(network_policy_operation)
        sess.run(network_discrim_operation)
        sess.run(network_Q_operation)

        # x1 = sess.run(net_param)
        episode_length = 0

        render = False
        sampler.sampler_reset()
        reward_episode_counter = []
        reward_episode_counter_airl = []

        while True:

            episode_length += 1
            act, v_pred = Policy_a.act(obs=[env.obs_a])
            # print("iteration: {}, episode: {}".format(iteration, episode_length))

            if discrete_env_check:
                act = np.asscalar(act)
            else:
                act = np.reshape(act, env.action_space.shape)

            v_pred = np.asscalar(v_pred)

            next_obs, reward, done, info = env.step(act)

            agent_sa_ph = sess.run(Policy_a.act_probs, feed_dict={Policy_a.obs: [env.obs_a.copy()],
                                                                  Policy_a.acts: [act]})
            agent_safety_q = Q_net_a.get_q_values([env.obs_a.copy()], [act])

            reward_a = np.asscalar(Discrim_a.get_rewards([env.obs_a.copy()], [act], agent_sa_ph, [agent_safety_q]))
            env.step_airl(reward_a)
            # env.render()

            # reward_episode += reward
            sampler.sampler_traj(env.obs_a.copy(), act, reward_a, v_pred, float(info["crashed"]))

            if render:
                env.render()

            if done:
                # next state of terminate state has 0 state value
                sampler.sampler_total(0)
                reward_episode_counter.append(env.get_episode_reward())
                reward_episode_counter_airl.append(env.get_episode_reward_airl())
                env.reset()
                # reward_episode = 0
            else:
                env.obs_a = next_obs.copy()

            if episode_length >= EPISODE_LENGTH:
                last_value = np.asscalar(Policy_a.get_value([next_obs]))
                sampler.sampler_total(last_value)
                env.reset()
                break

    return sampler.sampler_get_parallel(), np.mean(reward_episode_counter), np.mean(reward_episode_counter_airl)
