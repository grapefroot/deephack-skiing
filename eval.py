# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import logging
import os
import pickle
import random
import time

import gym
from gym import wrappers
import numpy as np
import pandas as pd
import pytz
import tqdm

import agents

gym.undo_logger_setup()
log = logging.getLogger(name=__name__)

AGENTS = {
  'agent':agents.Agent
}

def load_env():
    """
    Loads the skiing environment

    Returns
    -------
    open ai gym environment
    """
    log.debug('Loading Skiing environment')
    env = gym.make('Skiing-v0')
    env.reset()
    return env


def main(agent_name,
         render=False,
         upload=False,
         monitor=False,
         slow=0.0,
         n_episodes=None,
         seed=None,
         agent_args='{}',
         **kwargs):
    """
    Run an evaluation of an agent

    Parameters
    ----------
    agent_name : str
        The name of an agent to use
    render : bool, optional
        Whether to draw the game on screen
        Default: False
    upload : bool, optional
        Whether to upload the results
        Default: False
    monitor : bool, optional
        If True, record video and stats about this evaluation.
        Useful for debugging.
        Default: False
    slow : float, optional
        How long to wait (in secs) between frames
        Default: 0.0
    n_episodes : int, optional
        How many episodes to run. If None, will use env default setting.
        Default: None
    seed: int, optional
        The random seed to set
        Default: None
    agent_args: str, optional
        A dict (as JSON) of additional arguments to pass to the agent
        initialization
        Default: '{}'
    **kwargs : keyword arguments
        Unused

    Raises
    ------
    ValueError
        Incompatible options
    """
    if not monitor and upload:
        raise ValueError('Cannot upload without monitoring!')

    # load the gym
    env = load_env()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        env._seed(seed)

    # get a timestamp for the results
    eastern = pytz.timezone('US/Eastern')
    timestamp = datetime.datetime.now(eastern).strftime(
        '%Y-%m-%d__%H-%M-%S')

    # instantiate a specific agent
    agent = AGENTS[agent_name](**json.loads(agent_args))

    # set up a location for the reuslts
    local_dir = os.path.dirname(__file__)
    results_dir = os.path.join(local_dir, 'results', agent_name, timestamp)
    os.makedirs(results_dir)

    if monitor:
        # don't let the monitor log at anything below info
        monitor_level = max(logging.INFO, log.getEffectiveLevel())
        logging.getLogger(
            'gym.monitoring.video_recorder').setLevel(monitor_level)

        # start monitoring results
        # env.monitor.start(results_dir, seed=0)
        env = wrappers.Monitor(env, results_dir)

    episode_count = env.spec.trials if n_episodes is None else n_episodes
    # max_steps = env.spec.timestep_limit
    max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    # track total rewards
    total_rewards = []
    reward_log = []

    try:
        # use a progress bar unless debug logging
        with tqdm.tqdm(total=episode_count * max_steps,
                       disable=log.getEffectiveLevel() < logging.INFO) as pbar:
            # for each episode
            for episode in range(episode_count):
                if hasattr(agent, 'episode_number'):
                    episode = agent.episode_number

                total_reward = 0.0

                image = agents.features.green(env.reset())

                reward = 0

                for iteration in range(max_steps):
                    # update progress bar
                    pbar.update(n=1)

                    if render:
                        env.render()

                    # ask the agent what to do next
                    action = agent.act(image, centiseconds=-reward)

                    # take the action and get the new state and reward
                    new_image, reward, done, _ = env.step(action)
                    new_image = agents.features.green(new_image)
                    total_reward += reward

                    # feed back to the agent
                    agent.react(
                        image,
                        action,
                        reward,
                        done,
                        new_image,
                        centiseconds=((-reward) % 10) + 1
                    )

                    if done:
                        # calculate components of reward
                        pos_reward = int(-reward)
                        goal_reward = pos_reward - (pos_reward % 500)
                        slaloms_missed = goal_reward / 500
                        if slaloms_missed == 0 and total_reward == -30000:
                            slaloms_missed = 20

                        pbar.update(max_steps - iteration - 1)
                        break
                    else:
                        # update the old state
                        image = new_image

                    # slow down the simulation if desired
                    if slow > 0.0:
                        time.sleep(slow)

                # timeout the sim
                if iteration == max_steps:
                    msg = 'Episode {} timed out after {} steps'.format(
                        episode, max_steps)
                    log.debug(msg)

                msg = (
                    'Episode {} ({} steps): '
                    '{}/{} (Sloth: {}, Slaloms Missed: {})'
                )
                msg = msg.format(
                    episode,
                    iteration,
                    int(total_reward),
                    int(total_reward + 15000),
                    int(total_reward + goal_reward),
                    slaloms_missed
                )
                log.debug(msg)

                total_rewards.append(total_reward)

                reward_log.append({
                    'episode': episode,
                    'reward': total_reward,
                    'sloth': int(total_reward + goal_reward),
                    'missed': slaloms_missed
                })

                if episode % 100 == 0 and episode != 0:
                    log.debug('100 episode average reward was {}'.format(
                        np.mean(total_rewards[-100:])))
                    # save the model
                    agent_path = os.path.join(
                        results_dir, 'agent_{}.pkl'.format(episode)
                    )
                    with open(agent_path, 'wb') as fout:
                        pickle.dump(agent, fout)

        log.debug('Last 100 episode average reward was {}'.format(
            np.mean(total_rewards[-100:])))
        log.debug('Best {}-episode average reward was {}'.format(
            episode_count, np.mean(total_rewards)))

    finally:
        if monitor:
            # Dump result info to disk
            # env.monitor.close()
            env.close()

        # debugging output
        if hasattr(agent, 'data') and agent.data is not None:
            df = pd.DataFrame(agent.data)
            df.to_csv(os.path.join(results_dir, 'data.csv'))

        # rewards output
        df = pd.DataFrame(reward_log)
        df.to_csv(os.path.join(results_dir, 'rewards.csv'))

        log.info('Average reward of last 100 episodes: {}'.format(
            df.reward.values[-100:].mean())
        )
        log.info(
            'Average cost of elapsed time over last 100 episodes: {}'.format(
                df.sloth.values[-100:].mean()
            )
        )
        log.info(
            'Average number of slaloms missed over last 100 episodes: '
            '{}'.format(df.missed.values[-100:].mean())
        )

        with open(os.path.join(results_dir, 'agent_args.json'), 'w') as fout:
            fout.write(agent_args)

    if upload:
        # Upload to the scoreboard.
        log.info('Uploading results from {}'.format(results_dir))
        gym.upload(results_dir)


def parse_args():
    """
    Parses the arguments from the command line

    Returns
    -------
    argparse.Namespace
    """
    desc = 'Run and upload an evaluation for the skiing project'
    parser = argparse.ArgumentParser(description=desc)

    agent_name_help = 'Which agent to use'
    parser.add_argument('agent_name',
                        type=str,
                        help=agent_name_help)

    render_help = 'Whether to render the screen'
    parser.add_argument('-r',
                        '--render',
                        action='store_true',
                        help=render_help)

    upload_help = 'Whether to upload'
    parser.add_argument('-u',
                        '--upload',
                        action='store_true',
                        help=upload_help)

    monitor_help = 'Record video and stats'
    parser.add_argument('--monitor',
                        action='store_true',
                        help=monitor_help)

    slow_help = 'How long (in seconds) to wait between frames'
    parser.add_argument('-s',
                        '--slow',
                        type=float,
                        default=0.0,
                        help=slow_help)

    n_episodes_help = ('How many episodes to run'
                       '(if None, will use env default setting)')
    parser.add_argument('-e',
                        '--n-episodes',
                        type=int,
                        default=None,
                        help=n_episodes_help)

    seed_help = ('Set the random seed')
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help=seed_help)

    agent_args_help = ('Additional arguments to pass to the agent on '
                       'initialization')
    parser.add_argument('--agent-args',
                        type=str,
                        default='{}',
                        help=agent_args_help)

    verbosity_help = 'Verbosity level (default: %(default)s)'
    choices = [logging.getLevelName(logging.DEBUG),
               logging.getLevelName(logging.INFO),
               logging.getLevelName(logging.WARN),
               logging.getLevelName(logging.ERROR)]

    parser.add_argument('-v',
                        '--verbosity',
                        choices=choices,
                        help=verbosity_help,
                        default=logging.getLevelName(logging.INFO))

    # Parse the command line arguments
    args = parser.parse_args()

    # Set the logging to console level
    logging.basicConfig(level=args.verbosity)

    return args

if __name__ == '__main__':
    main(**parse_args().__dict__)
