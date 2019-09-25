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
            ent_coef=1, vf_coef=0.5, max_grad_norm=0.5, fisher_matrix=None,star_param=None,lam=None,batch_size=None,fisher_matrix2=None,
                 star_param2=None):

        sess = tf_util.get_session()
        nbatch = batch_size


        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):

            # train_model is used to train our network
            train_model = policy(nbatch, 1, sess)
            eval_model = policy(1,1,sess)

        # Update parameters using lossa.
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")
        print(len(params))

        ewc_loss = 0
        ewc_per_layer = []
        j = 0
        #star_param是load进去的array|params是模型的参数，也是tensor
        for v in range(len(params)-4):
            if v % 2 == 0:
                C = star_param[v].shape[-1]
                W = tf.transpose(tf.reshape(params[v] - star_param[v],[-1,C]))
                right = tf.reshape(tf.transpose(tf.matmul(tf.matmul(fisher_matrix[j+1], W), fisher_matrix[j])),[-1,1])
                W_ = tf.reshape(params[v] - star_param[v], [1,-1])

                ewc_loss += tf.matmul(W_, right)
                ewc_per_layer.append(tf.matmul(W_, right))
                j = j + 2
            else:
                B = tf.reshape(params[v] - star_param[v], [-1,1])
                right_B = tf.matmul(fisher_matrix[j], B)
                B_ = tf.reshape(params[v] - star_param[v], [1,-1])

                ewc_loss += tf.matmul(B_, right_B)
                ewc_per_layer.append(tf.matmul(B_,right_B))

                j += 1

        ewc_loss2 = 0
        j = 0
        for v in range(len(params)-4):
            if v % 2 == 0:
                C2 = star_param2[v].shape[-1]
                W2 = tf.transpose(tf.reshape(params[v] - star_param2[v],[-1,C2]))
                right2 = tf.reshape(tf.transpose(tf.matmul(tf.matmul(fisher_matrix2[j+1], W2), fisher_matrix2[j])),[-1,1])
                W_2 = tf.reshape(params[v] - star_param2[v], [1,-1])

                ewc_loss2 += tf.matmul(W_2, right2)
                j = j + 2
            else:
                B2 = tf.reshape(params[v] - star_param2[v], [-1,1])
                right_B2 = tf.matmul(fisher_matrix2[j], B2)
                B_2 = tf.reshape(params[v] - star_param2[v], [1,-1])

                ewc_loss2 += tf.matmul(B_2, right_B2)
                j += 1


        loss1 = ewc_loss * (lam/2)
        loss2 = ewc_loss2 * (lam/2)
        loss =  loss1 + loss2

        # 2. Calculate the gradients
        trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
        # grads_and_var = trainer.compute_gradients(loss, params)
        # grads, var = zip(*grads_and_var)
        # grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
        #
        # grads_and_var = list(zip(grads, var))

        grads = tf.gradients(loss, params)

        grads_and_var = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        # trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, decay=alpha, epsilon=epsilon)

        # trainer = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9,use_nesterov=True)
        _train = trainer.apply_gradients(grads_and_var)
        #
        #
        # grads2 = tf.gradients(loss2,params)
        # grads2 = list(zip(grads2, params))
        # trainer2 = tf.train.RMSPropOptimizer(learning_rate=7e-4, decay=alpha, epsilon=epsilon)
        # trainer2 = tf.train.AdamOptimizer(learning_rate=1e-4)
        # trainer2 = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9, use_nesterov=True)
        # _train2 = trainer2.apply_gradients(grads2)

        def train():

            td_map = {train_model.keep_prob:1.0}


            ewc1,ewc2,l,_ = sess.run([loss1,loss2,loss,_train],td_map)

            return ewc1,ewc2,l

            # KL,ewc,_ = sess.run([KL_loss,ewc_loss,_train],td_map)

            # return KL, ewc

        def creat_star_param():
            star_list = []
            for i in params[:-2]:
                star_list.append(star_param[params[i].name])
            return star_list

        self.creat_star_param = creat_star_param
        self.train = train
        self.train_model = train_model
        self.act = eval_model.step
        self.act2 = eval_model.step2
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
    set_global_seeds(seed)
    if network == 'cnn':
        network_kwargs['one_dim_bias'] = True
    # Get the nb of env
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    # print('--------------------------------',load_fm)
    # if load_fm is not None:
    # fisher_matrix = joblib.load('fisher_matrix/FM30000_static.npz')
    fisher_matrix = joblib.load('fisher_matrix/simple_agent_1_random_50000')
    fisher_matrix2 = joblib.load('fisher_matrix/fish_100000')
    print('fisher matrix done')
    # else:
    #     fisher_matrix = None
    star_param = joblib.load('initial_parameter/420000')
    star_param2 = joblib.load('initial_parameter/26000_kfac')
    # if load_path is not None:
    #     star_param = joblib.load(load_path)
    # else:
    #     star_param = None

    batch_size = 400
    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,fisher_matrix=fisher_matrix,star_param=star_param,lam=100,batch_size=batch_size,
                  fisher_matrix2=fisher_matrix2,star_param2=star_param2)
    if load_path is not None:

        model.load(load_path)

    sess = tf_util.get_session()
    kl_loss = 0
    ewc_loss = 0
    loss = 0
    for epoch in range(1,100000):

        kl,ewc,l = model.train()

        kl_loss += kl
        ewc_loss += ewc
        loss += l
        if epoch % 100 == 0:
            print('Average kl_loss at epoch {0}: {1}'.format(epoch, kl_loss / epoch))
            print('Average ewc_loss at epoch {0}: {1}'.format(epoch, ewc_loss / epoch))
            logger.record_tabular("kl_loss", kl)
            logger.record_tabular("ewc_loss", ewc)
            # logger.dump_tabular()

        # if epoch % 50 == 0 and epoch != 0:

        # win_rate_simple, lose_rate_simple, tie_rate_simple, win_rate_static, lose_rate_static, tie_rate_static = test_TwoAgent(model)
        # win_rate_static_0 = test_static_0(model)
            win_rate_simple = test_simple(model)

            win_rate_static_3 = test_static(model)

            logger.record_tabular("win_rate_simple", win_rate_simple)
            # logger.record_tabular("lose_rate_simple", lose_rate_simple)
            # logger.record_tabular("tie_rate_simple", tie_rate_simple)

            logger.record_tabular("win_rate_static_3", win_rate_static_3)
            # logger.record_tabular("lose_rate_static", lose_rate_static)
            # logger.record_tabular("tie_rate_static", tie_rate_static)
            model.save(save_path + 'updates' + str(epoch))
            logger.dump_tabular()

    return model