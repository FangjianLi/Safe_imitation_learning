import argparse
import gym
import numpy as np
from others.interact_with_envs import test_function_Highway_CBF
from others.utils import StructEnv_Highway_Q
import ray
import os
import customized_highway_env
import time



def check_and_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_savedir', default='trajectory/CBF_sampled_safety_critic_training/')
    parser.add_argument('--index', default='data_1')
    parser.add_argument("--envs_1", default="highway_cbf-v0")
    parser.add_argument('--min_length', default=200000, type=int)  # 50000
    parser.add_argument('--num_parallel_sampler', default=100, type=int)  # 100

    return parser.parse_args()


def main(args):
    start_timer = time.time()

    traj_save_dir = args.traj_savedir + args.index +'/'
    check_and_create_dir(traj_save_dir)


    env = StructEnv_Highway_Q(gym.make(args.envs_1))
    env.config["observation"]["vehicles_count"] = 7
    env.config['vehicles_count'] = 20
    env.config["duration"] = 100
    env.config["simulation_frequency"] = 10
    env.config["policy_frequency"] = 2
    env.reset_0()

    env.seed(0)


    environment_sampling = []

    # here, we need the we need to output observation_batch_total, observation_next_batch_total,  action_batch_total, action_next_batch_total, reward_batch_total, risk_batch_total,  done_batch_total

    for i in range(args.num_parallel_sampler):
        x1 = test_function_Highway_CBF.remote(args.envs_1, np.ceil(args.min_length / args.num_parallel_sampler), i)
        environment_sampling.append(x1)

    results = ray.get(environment_sampling)

    print(np.shape(results[0][0]))

    sampling_unpack = np.concatenate([result[0] for result in results], axis=1)
    evaluation_1 = np.mean([result[1] for result in results])

    observation_batch_total, observation_next_batch_total, action_batch_total, action_next_batch_total, reward_batch_total, risk_batch_total, done_batch_total = sampling_unpack

    observation_batch_total = np.array([observation_batch for observation_batch in observation_batch_total])
    observation_next_batch_total = np.array(
        [observation_next_batch for observation_next_batch in observation_next_batch_total])
    action_batch_total = np.array([action_batch for action_batch in action_batch_total])
    action_next_batch_total = np.array([action_next_batch for action_next_batch in action_next_batch_total])
    reward_batch_total = np.array([reward_batch for reward_batch in reward_batch_total])
    risk_batch_total = np.array([risk_batch for risk_batch in risk_batch_total])
    done_batch_total = np.array([done_batch for done_batch in done_batch_total])

    print("The average episode reward is: {}".format(evaluation_1))
    print("It takes: {}".format(time.time() - start_timer))
    print(np.shape(observation_batch_total))
    print(np.shape(observation_next_batch_total))
    print(np.shape(action_batch_total))
    print(np.shape(action_next_batch_total))
    print(np.shape(reward_batch_total))
    print(np.shape(risk_batch_total))
    print(np.shape(done_batch_total))

    print(f"The number of sampled risk episodes are: {np.sum(risk_batch_total)}")

    np.save(traj_save_dir + 'observations.npy', observation_batch_total)
    np.save(traj_save_dir + 'observations_next.npy', observation_next_batch_total)
    np.save(traj_save_dir + 'actions.npy', action_batch_total)
    np.save(traj_save_dir + 'actions_next.npy', action_next_batch_total)
    np.save(traj_save_dir + 'rewards.npy', reward_batch_total)
    np.save(traj_save_dir + 'risks.npy', risk_batch_total)
    np.save(traj_save_dir + 'done.npy', done_batch_total)


if __name__ == '__main__':
    args = argparser()
    main(args)
