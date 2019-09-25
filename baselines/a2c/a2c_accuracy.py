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
import joblib
import numpy as np
import os

class Model(object):

    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model

        train():
        - Make the training part (feedforward and retropropagation of gradients)

        save/load():
        - Save load the model
    """
    def __init__(self, policy, env, nsteps,
            ent_coef=1, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear',lam=None,batch_size=None):

        sess = tf_util.get_session()
        nbatch = batch_size


        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):

            # train_model is used to train our network
            train_model = policy(nbatch, 1, sess)
            eval_model = policy(1,1,sess)


        OUTPUT = tf.placeholder(tf.int64, [None],name='sample_action')


        def test_accuracy(obs, actions):
            td_map = {train_model.X: obs, OUTPUT: actions, train_model.keep_prob: 1.0}
            correct_preds = tf.equal(tf.argmax(tf.nn.softmax(train_model.pi), 1), OUTPUT)
            eval_accur = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
            acc = sess.run(eval_accur, td_map)
            return acc


        self.train_model = train_model
        self.accuracy= test_accuracy
        self.act = eval_model.step
        self.act2 = eval_model.step2
        self.act3 = eval_model.step3
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)



def learn(
    network,
    env,
    save_path,
    load_fm=None,
    seed=None,
    nsteps=5,
    total_timesteps=int(80e6),
    vf_coef=0.5,
    ent_coef=1,
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

    '''

    '''
    if any(save_path):
        os.makedirs(save_path,exist_ok=True)

    set_global_seeds(seed)
    if network == 'cnn':
        network_kwargs['one_dim_bias'] = True
    # Get the nb of env
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    data = np.load('distillation_data/static_random_3.npz')
    observation = data['observation']
    action = np.argmax(data['action'],1)
    data_size = action.shape[0]

    batch_size = data_size
    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                  lrschedule=lrschedule,lam=1000,batch_size=batch_size)
    if load_path is not None:
        model.load(load_path)
    # load data
    # data = np.load('distillation_data/simple_nobomb_random_3.npz')

    print('size',data_size)
    sess = tf_util.get_session()
    acc = model.accuracy(observation,action)

    print(acc)
    # for epoch in range(50):
    #     inds = np.arange(data_size)
    #     np.random.shuffle(inds)
    #     nbatch = 0
    #     kl_loss = 0
    #     ewc_loss = 0
    #     loss = 0
    #     for start in range(0, data_size, batch_size):
    #         end = start + batch_size
    #         mb_inds = inds[start:end]
    #         kl,ewc,l = model.train(observation[mb_inds],action[mb_inds])
    #
    #         kl_loss += kl
    #         ewc_loss += ewc
    #         loss += l
    #         nbatch += 1
    #         if nbatch % 50 == 0:
    #             print('Average kl_loss at epoch {0}: {1}'.format(epoch, kl_loss / nbatch))
    #             print('Average ewc_loss at epoch {0}: {1}'.format(epoch, ewc_loss / nbatch))
    #             logger.record_tabular("kl_loss", kl)
    #             logger.record_tabular("ewc_loss", ewc)
    #             logger.dump_tabular()
    #
    #     # win_rate_simple, lose_rate_simple, tie_rate_simple, win_rate_static, lose_rate_static, tie_rate_static = test_TwoAgent(model)
    #     # win_rate_static_0 = test_static_0(model)
    #     win_rate_simple = test_simple(model)
    #     # win_rate_simple_3 = test_3ffa(model)
    #
    #     # win_rate_nobomb = test_nobomb(model)
    #
    win_rate_static_3 = test_static(model)
    print(win_rate_static_3)
    #
    #     logger.record_tabular("win_rate_simple", win_rate_simple)
    #     # logger.record_tabular("lose_rate_simple", lose_rate_simple)
    #     # logger.record_tabular("tie_rate_simple", tie_rate_simple)
    #
    #     # logger.record_tabular("win_rate_simple_3", win_rate_simple_3)
    #     # logger.record_tabular("lose_rate_simple_3", lose_rate_simple_3)
    #     # logger.record_tabular("tie_rate_simple_3", tie_rate_simple_3)
    #
    #       logger.record_tabular("win_rate_static_3", win_rate_static_3)
    #     # # logger.record_tabular("lose_rate_static", lose_rate_static)
    #     # # logger.record_tabular("tie_rate_static", tie_rate_static)
    #
    #     # logger.record_tabular("win_rate_nobomb", win_rate_nobomb)
    #
    #     model.save(save_path + 'updates' + str(epoch))
    #     logger.dump_tabular()

    return model