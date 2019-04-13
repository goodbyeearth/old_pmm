from pommerman.agents import BaseAgent,SimpleAgent,SimpleNoBombAgent
import pommerman.envs
from gym import spaces
from pommerman.configs import *
import numpy as np
import tensorflow as tf
import joblib
import random
from pommerman.utility import join_json_state
from util import featurize

def act_ex_communication(agent,obs):
    '''Handles agent's move without communication'''
    if agent.is_alive:
        return agent.act(obs, action_space=spaces.Discrete(6))
    else:
        return constants.Action.Stop.value

def test_CNN(model):
    agent_list = [BaseAgent(),BaseAgent(),BaseAgent(),SimpleNoBombAgent()]
    env = pommerman.make('PommeFFACompetitionFast-v0',agent_list)
    # env.seed(123)
    # random.seed(0)
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
        while not done:
            # env.render()
            # env.save_json('json_file')
            act3 = act_ex_communication(agent_list[3],state[3])
            # act3 = 0
            act1,_,_,_ = model.act(featurize(state[1],1).reshape(-1, 8, 8, 19))
            # act1,_,_,_ = model.eval_step(featurize(state[1]).reshape(-1,11,11,18))
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



def test_LSTM(model):
    agent_list = [BaseAgent(),BaseAgent(),BaseAgent(),SimpleAgent()]
    env = pommerman.make('PommeFFACompetitionFast-v0',agent_list)
    # env.seed(123)
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
        mbstates = model.eval_model.initial_state
        while not done:
            # env.render()
            # env.save_json('json_file')
            act3 = act_ex_communication(agent_list[3],state[3])
            act3 = 0
            # act1 = model.act(featurize(state[1]).reshape(-1,11,11,18))
            act1,_,mbstates,_ = model.eval_step(featurize(state[1]).reshape(-1,11,11,18),S=mbstates, M=[done])
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
