import os.path as osp
import time
import functools
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.policies import build_policy
from baselines.common.tf_util import get_session, save_variables, load_variables

from baselines.a2c.runner import Runner
from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.acktr import kfac
from testing import *
import joblib

class Model(object):

    def __init__(self, policy,  nenvs,total_timesteps, nprocs=32, nsteps=20,
                 ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear', is_async=True):

        sess = get_session()
        nbatch = nenvs*nsteps

        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess)

            # train_model is used to train our network
            train_model = policy(nbatch, nsteps, sess)
            eval_model = policy(1,1,sess)

        # A = train_model.pdtype.sample_placeholder([None])
        # A = tf.placeholder(step_model.action.dtype, step_model.action.shape)
        probs = tf.nn.softmax(step_model.pi)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs),1)[0][0])
        self.pg_fisher = pg_fisher_loss = tf.log(probs[0,class_ind])

        ##Fisher loss construction
        # self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(neglogpac)
        # sample_net = train_model.vf + tf.random_normal(tf.shape(train_model.vf))
        # self.vf_fisher = vf_fisher_loss = - vf_fisher_coef*tf.reduce_mean(tf.pow(train_model.vf - tf.stop_gradient(sample_net), 2))
        # self.joint_fisher = joint_fisher_loss = pg_fisher_loss + vf_fisher_loss
        self.joint_fisher = joint_fisher_loss = pg_fisher_loss

        self.params=params = find_trainable_variables("a2c_model")

        with tf.device('/gpu:0'):
            self.optim = optim = kfac.KfacOptimizer()

            # update_stats_op = optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
            stats =  optim.compute_and_apply_stats(joint_fisher_loss, var_list=params[:-4])

        def compute_fisher(obs):
            # action = action[:, np.newaxis]
            td_map = {step_model.X:obs,step_model.keep_prob: 1.0}

            fisher = sess.run(
                stats,
                td_map
            )
            return fisher

        self.compute_fisher = compute_fisher
        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)

def learn(network, env, save_path,seed=None, total_timesteps=int(40e6), gamma=0.99, log_interval=1, nprocs=32, nsteps=20,
                 ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, save_interval=None, lrschedule='linear', load_path=None, is_async=False, **network_kwargs):
    set_global_seeds(seed)

    if network == 'cnn':
        network_kwargs['one_dim_bias'] = True

    policy = build_policy(env, network, **network_kwargs)

    nenvs = env.num_envs

    model = Model(policy,  nenvs, total_timesteps, nprocs=nprocs, nsteps
                                =nsteps, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=
                                vf_fisher_coef, lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                                lrschedule=lrschedule, is_async=is_async)

    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    # Calculate the batch_size,这里将nsteps设为1
    nbatch = nenvs*nsteps
    print(nbatch)
    tstart = time.time()
    F = []

    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values,output = runner.run()
        fisher = model.compute_fisher(obs)
        # f = []
        # l = len(obs)
        # efficient = 0.6
        # weight = np.logspace(l,1,l,base=efficient)
        #
        # for index in range(l):
        #     observation = obs[index]
        #     fisher = model.compute_fisher(observation)
        #     for i,j in enumerate(fisher):
        #         if index == 0:
        #             f.append(weight[index] * fisher[j])
        #         else:
        #             f[i]+=weight[index] * fisher[j]
        # for i in range(len(f)):
        #     f[i] = f[i]/np.sum(weight)
        #
        model.old_obs = obs
        nseconds = time.time()-tstart
        print(update)
        #
        # if update == 1:
        #     for x in f:
        #         F.append(x)
        # else:
        #     for x in range(len(f)):
        #         F[x] += f[x]

        if update == 1:
            for i in fisher:
                F.append(fisher[i])
        else:
            for i,j in enumerate(fisher):
                F[i] += fisher[j]
    for i in range(len(F)):
        F[i] /= total_timesteps
    joblib.dump(F,'fisher_matrix/simple_agent_random_4000')

    return model
