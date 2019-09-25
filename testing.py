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
    agent_list = [BaseAgent(),BaseAgent(),BaseAgent(),SimpleAgent()]
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
        # random.seed(4)
        state = env.reset()
        time_step = 0
        while not done:
            # env.render()
            # env.save_json('json_file')
            act3 = act_ex_communication(agent_list[3],state[3])
            act0 = 0
            act1,_,_,_,_ = model.act(featurize(state[1],1).reshape(-1, 8, 8, 19),keep_probs=1.0)
            # act1,_,_,_ = model.eval_step(featurize(state[1]).reshape(-1,11,11,18))
            action = [act0,act1,5,act3]
            # action = [0,0,0,act3]
            state, reward, done, info = env.step(action)
            time_step += 1
        print(reward,time_step)
        if reward[1] == 1:
            train_agent_rewards += 1
        elif reward[0] == 1:
            opponent_rewards += 1

    # join_json_state('json_file', ["StopAgent", "StopAgent", "StopAgent", "SimpleAgent"], "00",
    #                 "PommeFFACompetitionFast-v0", info)
    win_rate = train_agent_rewards/nGame
    lose_rate = opponent_rewards/nGame
    tie_rate = 1 - win_rate - lose_rate
    return win_rate,lose_rate,tie_rate

def test_TwoAgent(model):
    # agent_list = [BaseAgent(),BaseAgent(),BaseAgent(),SimpleNoBombAgent()]
    agent_list = [SimpleAgent(),BaseAgent(),SimpleAgent(),SimpleAgent()]
    env = pommerman.make('PommeFFACompetitionFast-v0',agent_list)
    # env.seed(123)
    # random.seed(0)
    win_rate_simple = 0
    lose_rate_simple = 0
    tie_rate_simple = 0
    win_rate_static = 0
    lose_rate_static = 0
    tie_rate_static = 0
    for j in range(2):
        train_agent_rewards = 0
        opponent_rewards = 0
        nGame = 50
        reward = [0,0,0,0]
        for _ in range(nGame):
            done = False
            # random.seed(2368878)
            state = env.reset()
            time_step = 0
            while not done:
                # env.render()
                # act0 = act_ex_communication(agent_list[0],state[0])
                act0 = 5
                # act2 = act_ex_communication(agent_list[2],state[2])
                act2 = 5
                if j == 0:
                    act3 = act_ex_communication(agent_list[3],state[3])
                    # act1, _, _, _ = model.act2(featurize(state[1], 1).reshape(-1, 8, 8, 19), keep_probs=1.0)
                elif j == 1:
                    act3 = 0
                act1,_,_,_ ,_= model.act(featurize(state[1],1).reshape(-1, 8, 8, 19),keep_probs=1.0)
                # act1,_,_,_ = model.eval_step(featurize(state[1]).reshape(-1,11,11,18))
                # action = [5,act1,5,act3]
                action = [act0,act1,act2,act3]
                state, reward, done, info = env.step(action)
                time_step += 1
            if reward[1] == 1:
                train_agent_rewards += 1
            # elif reward[3] == reward[0] == reward[1] == reward[2]:
            #     pass
            # else:
            elif reward[3] ==1:
                opponent_rewards += 1
        if j == 0:
            win_rate_simple = train_agent_rewards/nGame
            lose_rate_simple = opponent_rewards/nGame
            tie_rate_simple = 1 - win_rate_simple - lose_rate_simple
        else:
            win_rate_static = train_agent_rewards/nGame
            lose_rate_static = opponent_rewards/nGame
            tie_rate_static = 1 - win_rate_static - lose_rate_static

    return win_rate_simple,lose_rate_simple,tie_rate_simple,win_rate_static,lose_rate_static,tie_rate_static


def test_static(model):
    agent_list = [SimpleAgent(),BaseAgent(),SimpleAgent(),SimpleAgent()]
    env = pommerman.make('PommeFFACompetitionFast-v0',agent_list)
    env._agents[0].is_alive = False
    env._agents[2].is_alive = False

    env._agents[0].restart = False
    env._agents[1].restart = True
    env._agents[2].restart = False
    env._agents[3].restart = True

    train_agent_rewards = 0
    opponent_rewards = 0
    nGame = 50
    reward = [0,0,0,0]
    for i in range(nGame):
        done = False
        state = env.reset()
        time_step = 0
        while not done:
            act3 = 0
            act1,_,_,_ ,_= model.act(featurize(state[1],1).reshape(-1, 8, 8, 19),keep_probs=1.0)
            action = [5,act1,5,act3]
            state, reward, done, info = env.step(action)
            time_step += 1
        if reward[1] == 1:
            train_agent_rewards += 1
        elif reward[3] == 1:
            opponent_rewards += 1

    win_rate = train_agent_rewards/nGame

    return win_rate

def test_static_0(model):
    agent_list = [SimpleAgent(),BaseAgent(),SimpleAgent(),SimpleAgent()]
    env = pommerman.make('PommeFFACompetitionFast-v0',agent_list)
    env._agents[3].is_alive = False
    env._agents[2].is_alive = False

    env._agents[3].restart = False
    env._agents[1].restart = True
    env._agents[2].restart = False
    env._agents[0].restart = True


    train_agent_rewards = 0
    opponent_rewards = 0
    nGame = 50
    reward = [0,0,0,0]
    for i in range(nGame):
        done = False
        random.seed(4)
        state = env.reset()
        time_step = 0
        while not done:
            act0 = 0
            act1,_,_,_ ,_= model.act(featurize(state[1],1).reshape(-1, 8, 8, 19),keep_probs=1.0)
            action = [act0,act1,5,5]
            state, reward, done, info = env.step(action)
            time_step += 1
        if reward[1] == 1:
            train_agent_rewards += 1
        elif reward[0] == 1:
            opponent_rewards += 1

    win_rate = train_agent_rewards/nGame

    return win_rate


def test_nobomb(model):
    agent_list = [SimpleAgent(), BaseAgent(), SimpleAgent(), SimpleNoBombAgent()]
    env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)
    env._agents[0].is_alive = False
    env._agents[2].is_alive = False
    env._agents[0].restart = False
    env._agents[1].restart = True
    env._agents[2].restart = False
    env._agents[3].restart = True

    train_agent_rewards = 0
    opponent_rewards = 0
    nGame = 50
    reward = [0, 0, 0, 0]
    for i in range(nGame):
        done = False
        state = env.reset()
        time_step = 0
        while not done:
            act3 = act_ex_communication(agent_list[3], state[3])
            act1, _, _, _ = model.act3(featurize(state[1], 1).reshape(-1, 8, 8, 19),keep_probs=1.0)
            action = [5, act1, 5, act3]
            state, reward, done, info = env.step(action)
            time_step += 1
        if reward[1] == 1:
            train_agent_rewards += 1
        elif reward[3] == 1:
            opponent_rewards += 1
        # print(reward)
    win_rate = train_agent_rewards / nGame

    return win_rate

def test_3ffa(model):
    agent_list = [SimpleAgent(),BaseAgent(),SimpleAgent(),SimpleAgent()]
    env = pommerman.make('PommeFFACompetitionFast-v0',agent_list)
    env._agents[0].restart = True
    env._agents[1].restart = True
    env._agents[2].restart = True
    env._agents[3].restart = True

    train_agent_rewards = 0
    opponent_rewards = 0
    nGame = 50
    reward = [0,0,0,0]
    for i in range(nGame):
        done = False
        state = env.reset()
        time_step = 0
        while not done and env._agents[1].is_alive == True:
            act0 = act_ex_communication(agent_list[0], state[0])
            act2 = act_ex_communication(agent_list[2], state[2])
            act3 = act_ex_communication(agent_list[3], state[3])
            act1,_,_,_ ,_= model.act(featurize(state[1],1).reshape(-1, 8, 8, 19),keep_probs=1.0)
            action = [act0,act1,act2,act3]
            state, reward, done, info = env.step(action)
            time_step += 1
        if reward[1] == 1:
            train_agent_rewards += 1
        elif reward[3] == 1:
            opponent_rewards += 1

    win_rate = train_agent_rewards/nGame

    return win_rate

# if __name__ == '__main__':
#     # win_static = test_static(1)
#     # win_nobomb = test_nobomb()
#     # win_3ffa = test_3ffa()

def test_simple(model):
    agent_list = [SimpleAgent(),BaseAgent(),SimpleAgent(),SimpleAgent()]
    env = pommerman.make('PommeFFACompetitionFast-v0',agent_list)
    env._agents[0].is_alive = False
    env._agents[2].is_alive = False

    env._agents[0].restart = False
    env._agents[1].restart = True
    env._agents[2].restart = False
    env._agents[3].restart = True

    train_agent_rewards = 0
    opponent_rewards = 0
    nGame = 50
    reward = [0,0,0,0]
    for i in range(nGame):
        done = False
        # random.seed(4)
        state = env.reset()
        time_step = 0
        while not done:
            act3 = act_ex_communication(agent_list[3],state[3])
            act1,_,_,_ = model.act2(featurize(state[1],1).reshape(-1, 8, 8, 19),keep_probs=1.0)
            action = [5,act1,5,act3]
            state, reward, done, info = env.step(action)
            time_step += 1
        if reward[1] == 1:
            train_agent_rewards += 1
        elif reward[3] == 1:
            opponent_rewards += 1

    win_rate = train_agent_rewards/nGame

    return win_rate
