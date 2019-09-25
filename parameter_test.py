from pommerman.agents import BaseAgent,SimpleAgent,PlayerAgent,SimpleNoBombAgent
import pommerman.envs
from gym import spaces
from pommerman.configs import *
import numpy as np
import tensorflow as tf
import joblib
import random
from pommerman.utility import join_json_state
from util import featurize, old_featurize
from baselines.a2c.a2c import *

def act_ex_communication(agent,obs):
    '''Handles agent's move without communication'''
    if agent.is_alive:
        return agent.act(obs, action_space=spaces.Discrete(6))
    else:
        return constants.Action.Stop.value

def t():
    # agent_list = [BaseAgent(),PlayerAgent(),BaseAgent(),SimpleAgent()]
    agent_list = [SimpleAgent(),SimpleAgent(),SimpleAgent(),SimpleAgent()]
    # agent_list = [BaseAgent(),BaseAgent(),BaseAgent(),PlayerAgent()]
    env = pommerman.make('PommeFFACompetitionFast-v0',agent_list)

    # env._agent[0].is_alive = False
    # env._agent[1].is_alive = False
    # env._agent[2].is_alive = False
    # env._agent[3].is_alive = False
    env._agents[0].is_alive = False
    env._agents[2].is_alive = False

    env._agents[0].restart = False
    env._agents[1].restart = True
    env._agents[2].restart = False
    env._agents[3].restart = True

    env.observation_space.shape = (8,8,19)
    env.num_envs=1
    network_kwargs = {}
    network_kwargs['one_dim_bias'] = True
    policy=build_policy(env,'cnn',**network_kwargs)
    sess = tf.InteractiveSession()
    # mod = PPOPolicy(sess, 5, env.action_space, 1, 1, 1)

    model = Model(policy=policy,
                  env=env,
                  nsteps=16)
    # model.load('64')
    # model.load('initial_parameter/26000_kfac')
    # model.load('initial_parameter/480000')
    model.load('parameter/parameter78/updates1')
    # env.seed(123)
    # random.seed(0)
    train_agent_rewards = 0
    opponent_rewards = 0
    nGame = 100
    reward = [0,0,0,0]
    seed = []
    time_steps = []
    for i in range(nGame):
        done = False
        # random.seed(4)
        state = env.reset()
        time_step = 0
        while not done:
            # env.render()
            # env.save_json('jsonjson')
            act3 = act_ex_communication(agent_list[3],state[3])
            act0 = act_ex_communication(agent_list[0],state[0])
            act2 = act_ex_communication(agent_list[2], state[2])
            # act0 = 0
            # act2 = 0
            # act3= 0
            act1,_,_,_ = model.act2(featurize(state[1],1).reshape(-1,8,8,19),keep_probs=1.0)
            # action = [5,act1,5,act3]
            action = [act0,act1,act2,act3]
            state, reward, done, info = env.step(action)
            time_step += 1
        print(reward,time_step)
        time_steps.append(time_step)
        # join_json_state('jsonjson', ["StopAgent", "StopAgent", "StopAgent", "SimpleAgent"], "00",
        #                 "PommeFFACompetitionFast-v0", info)
        if reward[1] > 0:
            train_agent_rewards += 1
        elif reward[3] > 0:
            opponent_rewards += 1
    print("平均值为：%f" %np.mean(time_steps))
    print("方差为：%f" %np.var(time_steps))
    win_rate = train_agent_rewards/nGame
    lose_rate = opponent_rewards/nGame
    tie_rate = 1 - win_rate - lose_rate
    return win_rate,lose_rate,tie_rate


def LSTM(model):
    agent_list = [BaseAgent(),BaseAgent(),BaseAgent(),SimpleAgent()]
    env = pommerman.make('PommeFFACompetitionFast-v0',agent_list)

    train_agent_rewards = 0
    opponent_rewards = 0
    nGame = 50
    reward = [0,0,0,0]
    for i in range(nGame):
        done = False
        # random.seed(2368878)
        random.seed(4)
        state = env.reset()
        time_step = 0
        mbstates = model.initial_state
        while not done:
            env.render()
            # env.save_json('json_file')
            act3 = act_ex_communication(agent_list[3],state[3])
            act3 = 0
            # act1 = model.act(featurize(state[1]).reshape(-1,11,11,18))
            act1,_,mbstates,_ = model.step(featurize(state[1]).reshape(-1,11,11,18),S=mbstates, M=[done])
            action = [5,act1,5,act3]
            # action = [0,0,0,act3]
            state, reward, done, info = env.step(action)
            time_step += 1

        if reward[1] == 1:
            train_agent_rewards += 1
        elif reward[3] == 1:
            opponent_rewards += 1

    # join_json_state('json_file', ["StopAgent", "StopAgent", "StopAgent", "SimpleAgent"], "00",
    #                 "PommeFFACompetitionFast-v0", info)
    win_rate = train_agent_rewards/nGame
    lose_rate = opponent_rewards/nGame
    tie_rate = 1 - win_rate - lose_rate
    return win_rate,lose_rate,tie_rate

if __name__ == '__main__':
    a,b,c = t()
    print(a,b,c)


