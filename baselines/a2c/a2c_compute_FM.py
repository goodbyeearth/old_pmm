
import time
import functools
import tensorflow as tf

from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from baselines.common.policies import build_policy


from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.runner import Runner

from tensorflow import losses
from testing import *
import numpy as np
import joblib
from copy import deepcopy
import matplotlib.pyplot as plt
from functools import reduce

class Model(object):

    """
    """
    def __init__(self, policy, env, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):
        sess = tf_util.get_session()
        nenvs = env.num_envs
        nbatch = nenvs*nsteps

        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess)

            # train_model is used to train our network
            train_model = policy(nbatch, nsteps, sess)
            eval_model = policy(1,1,sess)

        self.params = find_trainable_variables("a2c_model")

        #compute softmax of output
        y = tf.nn.softmax(eval_model.pi)
        def compute_exact_fisher(obs,plot_diffs=False,disp_freq=10):
            var_list = []
            F_accum = []
            l = len(params)
            for i in range(l-2):
                #不考虑value function对应的两部分权值
                var_list.append(params[i])
                F_accum.append(np.zeros(var_list[i].get_shape().as_list()))
            num_sample = obs.shape[0]
            if(plot_diffs):
                # track differences in mean Fisher info
                F_prev = deepcopy(F_accum)
                mean_diffs = np.zeros(0)
            print(num_sample)
            for v in range(num_sample):
                for ind in range(6):
                    #输出6个动作
                    grad = tf.gradients(y[0][ind], var_list)
                    ders,output = sess.run([grad,y], feed_dict={eval_model.X:obs[v].reshape(1,8,8,19)})
                    for j in range(len(var_list)):
                        F_accum[j] += np.square(ders[j])/output[0][ind]
                if (plot_diffs):
                    if v % disp_freq == 0 and v > 0:
                        # recording mean diffs of F
                        F_diff = 0
                        for i in range(len(F_accum)):
                            F_diff += np.sum(np.absolute(F_accum[i] / (v + 1) - F_prev[i]))
                        # FIXME 为什么是mean
                        mean_diff = np.mean(F_diff)
                        mean_diffs = np.append(mean_diffs, mean_diff)
                        for x in range(len(F_accum)):
                            F_prev[x] = F_accum[x] / (v + 1)
            plt.switch_backend('agg')
            plt.plot(range(disp_freq+1, v+2, disp_freq), mean_diffs)
            plt.xlabel("Number of samples")
            plt.ylabel("Mean absolute Fisher difference")
            plt.grid()
            plt.savefig('fisher_convergence')

            for i in range(len(var_list)):
                F_accum[i] /= num_sample
            return F_accum

        self.compute_exact_fisher = compute_exact_fisher

        self.train_model = train_model
        self.step_model = step_model
        self.eval_model = eval_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.act = eval_model.step
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)

    def compute_fisher(self, obs,  plot_diffs=False, disp_freq=10):
        probs = tf.nn.softmax(self.eval_model.pi)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs),1)[0][0])

        var_list = []
        F_accum = []
        l = len(self.params)
        for i in range(l - 4):
            # 不考虑value function对应的两部分权值
            F_accum.append(np.zeros(self.params[i].get_shape().as_list()))
            var_list.append(self.params[i])

        num_sample = obs.shape[0]
        if (plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(F_accum)
            mean_diffs = np.zeros(0)
        sess = tf_util.get_session()
        print('num_sample', num_sample)

        for v in range(num_sample):
            ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), var_list),
                            feed_dict={self.eval_model.X: obs[v:v+1], self.eval_model.keep_prob:1.0})
            start = time.time()

            for j in range(len(var_list)):
                F_accum[j] += np.square(ders[j])

            if (plot_diffs):
                print(v)
                if v % disp_freq == 0 and v > 0:
                    # recording mean diffs of F
                    F_diff = 0
                    for i in range(len(F_accum)):
                        F_diff += np.sum(np.absolute(F_accum[i] / (v + 1) - F_prev[i]))
                    # FIXME 为什么是mean
                    mean_diff = np.mean(F_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)
                    for x in range(len(F_accum)):
                        F_prev[x] = F_accum[x] / (v + 1)
                    # display.display(plt.gcf())
                    # display.clear_output(wait=True)
                    # plt.switch_backend('agg')
                    plt.plot(range(disp_freq + 1, v + 2, disp_freq), mean_diffs)
                    plt.xlabel("Number of samples")
                    plt.ylabel("Mean absolute Fisher difference")
                    plt.grid()
        plt.savefig('fisher_convergence_simple_agent22')

        for i in range(len(F_accum)):
            F_accum[i] /= num_sample
        return F_accum


def learn(
    network,
    env,
    save_path,
    seed=None,
    nsteps=10,
    total_timesteps=int(80e6),
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=7e-4,
    lrschedule='linear',
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    # log_interval=100,
    log_interval=10,
    load_path=None,
    **network_kwargs):


    if network == 'cnn':
        network_kwargs['one_dim_bias'] = True
    set_global_seeds(seed)
    assert save_path is not None
    # Get the nb of env
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    # Calculate the batch_size,这里将nsteps设为1
    nbatch = nenvs*nsteps

    observation = []
    action = []
    for update in range(1, total_timesteps//nbatch+1):
        # Get mini batch of experiences
        obs, states, rewards, masks, actions, values,output = runner.run()
        observation.append(obs)
        print('times', update)
    obs = np.concatenate(observation)

    #compute fisher matrix
    FM = model.compute_fisher(obs,plot_diffs=True,disp_freq=10)
    # FM = model.compute_exact_fisher(obs,plot_diffs=True,disp_freq=10)

    joblib.dump(FM, save_path)

