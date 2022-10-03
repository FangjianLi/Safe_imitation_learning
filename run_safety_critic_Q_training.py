import argparse
import gym
import numpy as np
import tensorflow as tf
from network_models.q_net import Q_net
from algo.safety_critic_Q import DQNTrain
from others.DQN_buffer import DQN_batch
from others.utils import StructEnv_Highway
import warnings
import time
import os
import customized_highway_env

tf.reset_default_graph()
tf.autograph.set_verbosity(
    0, alsologtostdout=False
)


def check_and_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', help='save directory', default='trained_models/CBF_guided_safety_critic/')
    parser.add_argument('--model_save', help='save model name', default='model.ckpt')
    parser.add_argument('--reward_save', help="reward save directory",
                        default='rewards_record/CBF_guided_safety_critic/')

    # The hyperparameter of DQN_training
    parser.add_argument('--gamma', default=0.75, type=float)
    parser.add_argument('--lr', default=1e-5, type=float)  # 1e-5
    parser.add_argument('--buffer_size', default=int(2e4), type=int)
    parser.add_argument('--min_buffer_size', default=1e4, type=int)

    # The sampled trajectories
    parser.add_argument('--trajectory_dir', help='trajectory directory',
                        default='trajectory/CBF_sampled_safety_critic_training/data_1/')

    # The environment, we still want to keep it
    parser.add_argument("--envs_1", default="highway_original-v0")

    # The hyperparameter of the Q network
    parser.add_argument('--units_p', default=[128, 128], type=int)

    # The hyperparameter of the training
    parser.add_argument('--num_epoch', default=100000, type=int)
    parser.add_argument('--test_frequency', default=50, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--update_target_interval', default=20, type=int)
    parser.add_argument('--num_update_Q_network', default=6, type=int)

    # The hyperparameter of restoring the model
    parser.add_argument('--model_restore', help='filename of model to recover', default='model.ckpt')
    parser.add_argument('--continue_s', default=False, type=bool)
    parser.add_argument('--log_file', help='file to record the continuation of the training',
                        default='safety_critic_Q_policy.txt')

    return parser.parse_args()


def main(args):
    check_and_create_dir(args.savedir)
    check_and_create_dir(args.reward_save)

    if args.continue_s:
        args = np.load(args.savedir + "setup.npy", allow_pickle=True).item()

    env = StructEnv_Highway(gym.make(args.envs_1))
    env.seed(0)
    env.reset_0()

    Q_net_1 = Q_net('policy', env, args.units_p, activation_p=tf.nn.relu, activation_p_last_d=tf.nn.sigmoid)
    Target_Q_net = Q_net('old_policy', env, args.units_p, activation_p=tf.nn.relu, activation_p_last_d=tf.nn.sigmoid)
    DQN = DQNTrain(Q_net_1, Target_Q_net, lr=args.lr, discounted_value=args.gamma)
    saver = tf.train.Saver()

    loss_recorder = []

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        if args.continue_s:
            saver.restore(sess, args.savedir + '/' + args.model_restore)
            loss_recorder = np.load(args.loss_save).tolist()
            with open(args.savedir + args.log_file, 'a+') as r_file:
                r_file.write("the continue point: {}, the lr_policy: {} \n".format(len(loss_recorder), args.lr_policy))

        else:
            np.save(args.savedir + "setup.npy", args)

        update_loss = []
        buffer = DQN_batch(args.trajectory_dir)
        DQN.update_target_q_network()
        start_time = time.time()

        for ep in range(args.num_epoch):

            for _ in range(10):
                mb_obs, mb_obs_next, mb_act, mb_act_next, mb_risk = buffer.sample_minibatch(args.batch_size)
                train_loss = DQN.train_policy(mb_obs, mb_obs_next, mb_act, mb_act_next, mb_risk)

            update_loss.append(train_loss)
            loss_recorder.append(train_loss)

            if ep % args.update_target_interval == 0:
                DQN.update_target_q_network()

            # every test_frequency episodes, test the agent and write some stats in TensorBoard
            if ep % args.test_frequency == 0:
                print(
                    "Episode: {},  Train_loss:{}, Takes: {}".format(ep, np.mean(update_loss), time.time() - start_time))
                Q_network_parameters = sess.run(Q_net_1.get_trainable_variables())
                np.save(args.savedir + 'Q_para.npy', Q_network_parameters)
                np.save(args.reward_save + 'loss.npy', loss_recorder)
                saver.save(sess, args.savedir + '{}'.format(ep) + args.model_save)

                update_loss = []
                start_time = time.time()

        env.close()


if __name__ == '__main__':
    args = argparser()
    warnings.filterwarnings("ignore")
    main(args)
