import multiprocessing as mp
from pommerman.agents import BaseAgent,SimpleAgent     #,SimpleNoBombAgent
import numpy as np
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
from gym import spaces
from pommerman.configs import *
import random
from util import featurize

def act_ex_communication(agent,obs):
    '''Handles agent's move without communication'''
    if agent.is_alive:
        return agent.act(obs, action_space=spaces.Discrete(6))
    else:
        return constants.Action.Stop.value


def worker(remote, parent_remote, env_fn_wrapper):
    """
    这个work函数运行在 Process 里，把Process里的parent_remote关掉，通信对象是外面的remote。
    remote就是work_remote
    """
    parent_remote.close()

    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                act0 = act_ex_communication(env._agents[0], env.get_observations()[0])
                act2 = act_ex_communication(env._agents[2], env.get_observations()[2])
                act3 = act_ex_communication(env._agents[3], env.get_observations()[3])
                whole_action = [act0, data, act2, act3]
                ob, reward, done, info = env.step(whole_action)

                #通过距离来设计reward reshape
                # train_agent_x, train_agent_y = ob[1]['position']
                # opponent_agent_x, opponent_agent_y = ob[3]['position']
                # distance = abs(train_agent_x - opponent_agent_x) + abs(train_agent_y - opponent_agent_y)
                # r = 1 - (distance / 20) ** 0.4

                #通过剪道具来获得reward
                # reward_kick = (ob[1]['can_kick'] - 0) * 0.1
                # reward_ammo = (ob[1]['ammo'] - 1) * 0.1
                # reward_strength = (ob[1]['blast_strength'] - 2) * 0.1
                # r = reward_kick + reward_ammo + reward_strength

                # print("ob[1]：", ob[1])

                """ob[1]就是原始的字典，featurize之后变成三百多维的向量"""
                ob_1 = featurize(ob[1], 1)
                # print("len of ob:%d, type of ob:%s" % (len(ob), type(ob)))   # 4, list
                # print("all key of ob[1]:", ob[1].keys())
                # print("after featurize, ob[1] shape:", ob_1.shape)     # 11*11*19

                r = 0
                reward_1 = reward[1] + r

                #在多人条件下，如果我训练的智能体死了，那么就需要提前结束游戏
                if not done and not env._agents[1].is_alive:
                    reward_1 = -1
                    done = True

                if done:
                    # random.seed(4)
                    ob = env.reset()
                    ob_1 = featurize(ob[1], 1)
                    # if reward[1] == -1 and reward[3] == -1:
                    #     reward_1 = reward[1]
                    # elif reward[1] == -1:
                    #     reward_1 = reward[1]
                    # else:
                    #     reward_1 = reward[1] * 10
                    reward_1 = reward[1]

                remote.send((ob_1, reward_1, done, info))

            elif cmd == 'reset':
                # random.seed(4)
                ob = env.reset()
                ob_1 = featurize(ob[1], 1)
                remote.send(ob_1)

            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))

            elif cmd == 'close':
                remote.close()
                break

            elif cmd == 'get_spaces_spec':
                remote.send((env.observation_space, env.action_space, env.spec))

            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None, context='spawn'):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        self.nenvs=nenvs = len(env_fns)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(nenvs)])
        """
        ps 是一个长度为 nenvs 的进程列表，每一项都是一个新建的Process。
        每个Process里都在处理worker这个函数，每个worker函数都有两个remote，两个remote之间有管道连接
        """
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]

        """
        进程开关设置
        """

        """
        打开进程列表 ps 里的每个 Process
        """
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            # with clear_mpi_env_vars():
            p.start()

        """
        关掉当前进程的所有 work_remote，
        （由于worker函数中关掉了外面的remote）
        即让 work_remote 在 Process 里跑，remote在当前进程里跑
        """
        for remote in self.work_remotes:
            remote.close()   # 关work_remotes

        """remotes[0]即为在当前进程里跑的remote,向进程Process里的worker_remote查询space信息"""
        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv()

        """自定义部分,对observation重新定shape"""
        print("从worker_remote里得到的space信息")
        print("observation_space.shape:", observation_space.shape)
        print("action_space.shape:", action_space.shape)
        observation_space.shape = (11, 11, 19)

        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()

def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)
