# this is the newer one


import argparse
import gym
import numpy as np
import tensorflow as tf
from network_models.policy_net_continuous_discrete import Policy_net
from algo.ppo_combo import PPOTrain
from network_models.q_net import Q_net  # import a safety critic
from others.interact_with_envs import AIRL_Sampling_Highway_CBF_total
from others.utils import StructEnv_AIRL_Highway
from network_models.AIRL_net_discriminator_blend_CBF_total import Discriminator_CBF_total
from others.extra_penalty_buffer import Extra_penalty_buffer
import ray
import os
import warnings
import customized_highway_env

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.reset_default_graph()
tf.autograph.set_verbosity(
    0, alsologtostdout=False
)

tf.compat.v1.logging.set_verbosity(
    tf.compat.v1.logging.ERROR
)


def check_and_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', help='save directory', default='trained_models/SAIRL/')
    parser.add_argument('--model_save', help='save model name', default='model.ckpt')
    parser.add_argument('--safety_critic_directory', help='save model name',
                        default='trained_models/CBF_guided_safety_critic/Q_para.npy')
    parser.add_argument('--index', default='model_1')
    parser.add_argument('--reward_savedir', help="reward save directory", default='rewards_record/SAIRL/')

    # expert data
    parser.add_argument('--expert_traj_dir', help="expert data directory",
                        default='trajectory/expert_demonstrations_original/')

    # The environment
    parser.add_argument("--envs_1", default="highway_original-v0")

    # The hyperparameter of PPO_training
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lambda_1', default=0.95, type=float)
    parser.add_argument('--lr_policy', default=5e-5, type=float)
    parser.add_argument('--ep_policy', default=1e-9, type=float)
    parser.add_argument('--lr_value', default=5e-5, type=float)
    parser.add_argument('--ep_value', default=1e-9, type=float)
    parser.add_argument('--clip_value', default=0.2, type=float)
    parser.add_argument('--alter_value', default=False, type=bool)

    # The hyperparameter of the safety critic Q network
    parser.add_argument('--Q_units_p', default=[128, 128], type=int)

    # The hyperparameter of the policy network
    parser.add_argument('--units_p', default=[96, 96, 96], type=int)
    parser.add_argument('--units_v', default=[128, 128, 128], type=int)

    # The hyperparameter of the policy trainingAIRL
    parser.add_argument('--iteration', default=200, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_epoch_policy', default=10, type=int)  # 175 is good
    parser.add_argument('--num_epoch_value', default=20, type=int)
    parser.add_argument('--min_length', default=80000, type=int)  # 80000
    parser.add_argument('--num_parallel_sampler', default=40, type=int)  # 100

    # The hyperparameter of the discriminator network
    parser.add_argument('--lr_discrim', default=1e-4, type=float)

    # The hyperparameter of the discriminator training
    parser.add_argument('--num_expert_dimension', default=80000, type=int)  # 80000
    parser.add_argument('--num_epoch_discrim', default=5, type=int)
    parser.add_argument('--batch_size_discrim', default=512, type=int)

    # The hyperparameter of restoring the model
    parser.add_argument('--model_restore', help='filename of model to recover', default='model.ckpt')
    parser.add_argument('--continue_s', default=False, type=bool)
    parser.add_argument('--log_file', help='file to record the continuation of the training', default='continue_C1.txt')

    # The hyperparameter of extra penalty buffer
    parser.add_argument('--extra_buffer_size', default=256, type=int)
    parser.add_argument('--extra_buffer_threshold_h', default=0.6, type=float)  # 0.6
    parser.add_argument('--extra_buffer_threshold_l', default=0.05, type=float)
    
    parser.add_argument('--seed', default=0, type=float)

    return parser.parse_args()


def main(args):
    model_save_dir = args.savedir + args.index + '/'
    expert_traj_dir = args.expert_traj_dir
    reward_save_dir = args.reward_savedir + args.index + '/'
    check_and_create_dir(model_save_dir)
    check_and_create_dir(reward_save_dir)
    
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)

    if args.continue_s:
        args = np.load(model_save_dir + "setup.npy", allow_pickle=True).item()

    env = StructEnv_AIRL_Highway(gym.make(args.envs_1))
    env.reset_0()

    discrete_env_check = isinstance(env.action_space, gym.spaces.discrete.Discrete)
    env.seed(args.seed)
    

    Policy = Policy_net('policy', env, args.units_p, args.units_v)
    Old_Policy = Policy_net('old_policy', env, args.units_p, args.units_v)
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma, lambda_1=args.lambda_1, lr_policy=args.lr_policy,
                   lr_value=args.lr_value, clip_value=args.clip_value)
    saver = tf.train.Saver(max_to_keep=50)

    expert_observations = np.load(expert_traj_dir + 'observations.npy')
    expert_actions = np.load(expert_traj_dir + 'actions.npy')

    if not discrete_env_check:
        act_dim = env.action_space.shape[0]
        expert_actions = np.reshape(expert_actions, [-1, act_dim])
    else:
        expert_actions = expert_actions.astype(np.int32)

    discrim_ratio = args.num_expert_dimension / args.min_length
    discrim_batch_number = args.num_expert_dimension / args.batch_size_discrim

    D = Discriminator_CBF_total('AIRL_discriminator', env, args.lr_discrim, discrim_batch_number)

    # add the safety critic Q network:
    Q_net_1 = Q_net('Safety_critic_net', env, args.Q_units_p, activation_p=tf.nn.relu,
                    activation_p_last_d=tf.nn.sigmoid)

    # get the network recovery function:
    Q_net_param = Q_net_1.get_trainable_variables()
    Q_net_param_values = np.load(args.safety_critic_directory, allow_pickle=True)
    net_operation = []
    for i in range(len(Q_net_param)):
        net_operation.append(tf.assign(Q_net_param[i], Q_net_param_values[i]))

    # initialize the sampler buffer
    buffer_extra_penalty = Extra_penalty_buffer(args.extra_buffer_size, args.extra_buffer_threshold_h,
                                                args.extra_buffer_threshold_l)

    origin_reward_recorder = []
    AIRL_reward_recorder = []
    collision_rate_recorder = []
    counter_d = 0

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # recover the Q_network
        sess.run(net_operation)

        if args.continue_s:
            saver.restore(sess, model_save_dir + args.model_restore)
            origin_reward_recorder = np.load(reward_save_dir + "origin_reward.npy").tolist()
            AIRL_reward_recorder = np.load(reward_save_dir + "airl_reward.npy").tolist()
            collision_rate_recorder = np.load(reward_save_dir + "collision_rate.npy").tolist()

            with open(model_save_dir + args.log_file, 'a+') as r_file:
                r_file.write(
                    "the continue point: {}, the lr_policy: {}, the lr_value: {}, the lr_discrim: {} \n".format(
                        len(origin_reward_recorder), args.lr_policy, args.lr_value, args.lr_discrim))

        else:
            np.save(model_save_dir + "setup.npy", args)

        expert_sa_ph = sess.run(Policy.act_probs, feed_dict={Policy.obs: expert_observations,
                                                             Policy.acts: expert_actions})
        expert_safety_q_ph = np.reshape(Q_net_1.get_q_values(expert_observations, expert_actions), (-1, 1))

        for iteration in range(args.iteration):

            policy_value = sess.run(Policy.get_trainable_variables())
            discriminator_value = sess.run(D.get_trainable_variables())

            environment_sampling = []

            for i in range(args.num_parallel_sampler):
                x1 = AIRL_Sampling_Highway_CBF_total.remote(args.envs_1, policy_value, discriminator_value,
                                                            discrete_env_check,
                                                            np.ceil(args.min_length / args.num_parallel_sampler),
                                                            i, args.units_p, args.units_v, args.lr_discrim,
                                                            discrim_batch_number, args.Q_units_p)
                environment_sampling.append(x1)

            results = ray.get(environment_sampling)

            sampling_unpack = np.concatenate([result[0] for result in results], axis=1)
            evaluation_1 = np.mean([result[1] for result in results])
            evaluation_AIRL = np.mean([result[2] for result in results])

            observation_batch_total, action_batch_total, rtg_batch_total, gaes_batch_total, \
            value_next_batch_total, reward_batch_total, collision_batch_total = sampling_unpack

            observation_batch_total = np.array([observation_batch for observation_batch in observation_batch_total])
            action_batch_total = np.array([action_batch for action_batch in action_batch_total])
            rtg_batch_total = np.array([rtg_batch for rtg_batch in rtg_batch_total])
            gaes_batch_total = np.array([gaes_batch for gaes_batch in gaes_batch_total])
            value_next_batch_total = np.array([value_next_batch for value_next_batch in value_next_batch_total])
            reward_batch_total = np.array([reward_batch for reward_batch in reward_batch_total])
            collision_batch_total = np.array([collision_batch for collision_batch in collision_batch_total])

            gaes_batch_total = (gaes_batch_total - np.mean(gaes_batch_total)) / (
                    np.std(gaes_batch_total) + 1e-10)

            counter_d += 1
            # The AIRL training process, SGD might be used here
            if counter_d >= 5 + (iteration / 500) * 50 or iteration == 0:
                print("discriminator updated")

                counter_d = 0
                agent_sa_ph = sess.run(Policy.act_probs, feed_dict={Policy.obs: observation_batch_total,
                                                                    Policy.acts: action_batch_total})
                agent_safety_q_ph = np.reshape(Q_net_1.get_q_values(observation_batch_total, action_batch_total),
                                               (-1, 1))
                discrim_batch_expert = [expert_observations, expert_actions, expert_sa_ph, expert_safety_q_ph]
                discrim_batch_agent = [observation_batch_total, action_batch_total, agent_sa_ph, agent_safety_q_ph]

                # sample the extra penalty buffer
                buffer_extra_penalty.add(observation_batch_total, action_batch_total, agent_safety_q_ph)

                for epoch_discrim in range(args.num_epoch_discrim):

                    total_index_agent = np.arange(args.min_length)
                    total_index_expert = np.arange(int(args.min_length * discrim_ratio))

                    np.random.shuffle(total_index_agent)
                    np.random.shuffle(total_index_expert)

                    for i in range(0, args.min_length, args.batch_size_discrim):
                        sample_indices_agent = total_index_agent[i:min(i + args.batch_size_discrim, args.min_length)]
                        sample_indices_expert = total_index_expert[int(i * discrim_ratio):min(
                            int(i * discrim_ratio + args.batch_size_discrim * discrim_ratio),
                            int(args.min_length * discrim_ratio))]

                        sampled_batch_agent = [np.take(a=a, indices=sample_indices_agent, axis=0) for a in
                                               discrim_batch_agent]
                        sampled_batch_expert = [np.take(a=a, indices=sample_indices_expert, axis=0) for a in
                                                discrim_batch_expert]

                        extra_penalty_1 = buffer_extra_penalty.generate_the_penalty_item(args.batch_size_discrim, D)

                        D_loss, _ = D.train(expert_s=sampled_batch_expert[0],
                                            expert_a=sampled_batch_expert[1],
                                            agent_s=sampled_batch_agent[0],
                                            agent_a=sampled_batch_agent[1],
                                            expert_sa_p=sampled_batch_expert[2],
                                            agent_sa_p=sampled_batch_agent[2],
                                            expert_safety_q=sampled_batch_expert[3],
                                            agent_safety_q=sampled_batch_agent[3],
                                            extra_penalty=extra_penalty_1
                                            )
                print("The discriminator loss is {}".format(D_loss))
                print("The extra penalty is {}".format(extra_penalty_1 * 0.5))

            print("at {}, the average episode reward is: {}".format(iteration, evaluation_1))
            print("at {}, the average episode AIRL reward is: {}".format(iteration, evaluation_AIRL))
            print("at {}, the average collision rate is: {}".format(iteration, np.mean(collision_batch_total)))
            origin_reward_recorder.append(evaluation_1)
            AIRL_reward_recorder.append(evaluation_AIRL)
            collision_rate_recorder.append(np.mean(collision_batch_total))

            if iteration % 5 == 0 and iteration > 0:
                np.save(reward_save_dir + "origin_reward.npy", origin_reward_recorder)
                np.save(reward_save_dir + "airl_reward.npy", AIRL_reward_recorder)
                np.save(reward_save_dir + "collision_rate.npy", collision_rate_recorder)
                saver.save(sess, model_save_dir + '{}'.format(iteration) + args.model_save)

            inp_batch = [observation_batch_total, action_batch_total, gaes_batch_total, rtg_batch_total,
                         value_next_batch_total, reward_batch_total]

            PPO.assign_policy_parameters()

            # train
            if args.alter_value:
                for epoch in range(args.num_epoch_value):
                    total_index = np.arange(args.min_length)
                    np.random.shuffle(total_index)
                    for i in range(0, args.min_length, args.batch_size):
                        sample_indices = total_index[i:min(i + args.batch_size, args.min_length)]
                        sampled_inp_batch = [np.take(a=a, indices=sample_indices, axis=0) for a in inp_batch]
                        PPO.train_value_v(obs=sampled_inp_batch[0], v_preds_next=sampled_inp_batch[4],
                                          rewards=sampled_inp_batch[5])
            else:
                for epoch in range(args.num_epoch_value):
                    total_index = np.arange(args.min_length)
                    np.random.shuffle(total_index)
                    for i in range(0, args.min_length, args.batch_size):
                        sample_indices = total_index[i:min(i + args.batch_size, args.min_length)]

                        sampled_inp_batch = [np.take(a=a, indices=sample_indices, axis=0) for a in inp_batch]

                        PPO.train_value(obs=sampled_inp_batch[0], rtg=sampled_inp_batch[3])

            for epoch in range(args.num_epoch_policy):
                total_index = np.arange(args.min_length)
                np.random.shuffle(total_index)
                for i in range(0, args.min_length, args.batch_size):
                    sample_indices = total_index[i:min(i + args.batch_size, args.min_length)]
                    sampled_inp_batch = [np.take(a=a, indices=sample_indices, axis=0) for a in inp_batch]
                    PPO.train_policy(obs=sampled_inp_batch[0], actions=sampled_inp_batch[1], gaes=sampled_inp_batch[2])


if __name__ == '__main__':
    args = argparser()
    warnings.filterwarnings("ignore")
    main(args)
