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
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear',fisher_matrix=None,star_param=None,lam=None,batch_size=None):

        sess = tf_util.get_session()
        nbatch = batch_size


        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):

            # train_model is used to train our network
            train_model = policy(nbatch, 1, sess)
            eval_model = policy(1,1,sess)


        OUTPUT = tf.placeholder(tf.float32, [None,6],name='sample_action')
        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
        # constant = tf.constant(0.01,shape=train_model.pi.get_shape())


        pi = tf.nn.softmax(OUTPUT)
        # u = tf.argmax(pi,axis=-1)
        # x = tf.one_hot(u, pi.get_shape().as_list()[-1])
        # cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_model.pi,labels=x))

        train_model_pi = tf.nn.softmax(train_model.pi)
        KL = pi * tf.log(pi/(train_model_pi + 1e-4))
        KL_loss = tf.reduce_mean(tf.reduce_sum(KL,1))
        # Update parameters using lossa.
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")
        print(len(params))

        ewc_loss = 0
        #star_param是load进去的array|params是模型的参数，也是tensor
        for i in range(len(fisher_matrix)):
            j = 0
            for v in range(len(params)-4):
                if v % 2 == 0:
                    C = star_param[v].shape[-1]
                    W = tf.transpose(tf.reshape(params[v] - star_param[v],[-1,C]))
                    right = tf.reshape(tf.transpose(tf.matmul(tf.matmul(fisher_matrix[i][j+1], W), fisher_matrix[i][j])),[-1,1])
                    W_ = tf.reshape(params[v] - star_param[v], [1,-1])
                    ewc_loss += tf.matmul(W_, right)
                    j = j + 2
                else:
                    B = tf.reshape(params[v] - star_param[v], [-1,1])
                    right_B = tf.matmul(fisher_matrix[i][j], B)
                    B_ = tf.reshape(params[v] - star_param[v], [1,-1])
                    ewc_loss += tf.matmul(B_, right_B)
                    j += 1

        # for v in range(len(params)-4):
        #     if v % 2 == 0:
        #         C = int(star_param[v].shape[-1])
        #         W_hat = tf.concat([tf.reshape(params[v], [-1, C]), tf.reshape(params[v + 1], [1, -1])], 0)
        #         W_hat_fixed = tf.concat(
        #             [tf.reshape(star_param[v], [-1, C]), tf.reshape(star_param[v + 1], [1, -1])], 0)
        #
        #         W = tf.transpose(W_hat - W_hat_fixed)
        #         right = tf.reshape(tf.transpose(tf.matmul(tf.matmul(fisher_matrix[v + 1], W), fisher_matrix[v])),[-1, 1])
        #         W_ = tf.reshape(W_hat - W_hat_fixed, [1, -1])
        #         ewc_loss += tf.matmul(W_, right)
        # for i in range(len(fisher_matrix)):
        #     for v in range(len(params)-4):
        #     # if v == 6:
        #         ewc_loss += tf.reduce_sum(tf.multiply(fisher_matrix[i][v].astype(np.float32), tf.square(params[v] - star_param[v])))


        loss1 = KL_loss * ent_coef
        loss2 = ewc_loss * (lam/2)
        loss = loss1 + loss2
        # 2. Calculate the gradients
        trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
        # trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, decay=alpha, epsilon=epsilon)
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



        def train(obs, actions):

            td_map = {train_model.X:obs, OUTPUT:actions,train_model.keep_prob:1.0}


            kl,ewc,l,_ = sess.run([KL_loss,ewc_loss,loss,_train],td_map)

            return kl,ewc,l

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
    if any(save_path):
        os.makedirs(save_path,exist_ok=True)

    set_global_seeds(seed)
    if network == 'cnn':
        network_kwargs['one_dim_bias'] = True
    # Get the nb of env
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    # print('--------------------------------',load_fm)
    # if load_fm is not None:
    # fisher_matrix = joblib.load('fisher_matrix_ewc/static_agent_3_random_400')
    fisher_matrix2 = joblib.load('fisher_matrix/simple_agent_random_4000')
    # fisher_matrix = joblib.load('fisher_matrix_tf/static_agent_3_random_200')
    fisher_matrix = [fisher_matrix2]
    print('fisher matrix done')
    # else:
    #     fisher_matrix = None
    # star_param = joblib.load('parameter/parameter69/22')
    star_param = joblib.load('initial_parameter/420000')
    # if load_path is not None:
    #     star_param = joblib.load(load_path)
    # else:
    #     star_param = None
    batch_size = 400
    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                  lrschedule=lrschedule,fisher_matrix=fisher_matrix,star_param=star_param,lam=5000,batch_size=batch_size)
    if load_path is not None:

        model.load(load_path)

    # load data
    data = np.load('distillation_data/simple_random_3.npz')
    observation = data['observation'][:32000]
    action = data['action'][:32000]

    # data2 = np.load('distillation_data/static_random_3.npz')
    # observation2 = data2['observation']
    # action2 = data2['action']
    # observation2 = np.repeat(observation2, 10, axis=0)
    # action2 = np.repeat(action2,10,axis=0)
    #
    data3 = np.load('distillation_data/simple_nobomb_random_3.npz')
    observation3 = data3['observation']
    action3 = data3['action']
    #
    observation = np.concatenate([observation3, observation])
    action = np.concatenate([action3, action])

    data_size = action.shape[0]
    print('size',data_size)
    sess = tf_util.get_session()

    for epoch in range(50):
        inds = np.arange(data_size)
        np.random.shuffle(inds)
        nbatch = 0
        kl_loss = 0
        ewc_loss = 0
        loss = 0
        for start in range(0, data_size, batch_size):
            end = start + batch_size
            mb_inds = inds[start:end]
            kl,ewc,l = model.train(observation[mb_inds],action[mb_inds])

            kl_loss += kl
            ewc_loss += ewc
            loss += l
            nbatch += 1
            if nbatch % 50 == 0:
                print('Average kl_loss at epoch {0}: {1}'.format(epoch, kl_loss / nbatch))
                print('Average ewc_loss at epoch {0}: {1}'.format(epoch, ewc_loss / nbatch))
                logger.record_tabular("kl_loss", kl)
                logger.record_tabular("ewc_loss", ewc)
                logger.dump_tabular()


        # win_rate_simple, lose_rate_simple, tie_rate_simple, win_rate_static, lose_rate_static, tie_rate_static = test_TwoAgent(model)
        # win_rate_static_0 = test_static_0(model)
        win_rate_simple = test_simple(model)
        # win_rate_simple_3 = test_3ffa(model)

        win_rate_nobomb = test_nobomb(model)

        # win_rate_static_3 = test_static(model)

        logger.record_tabular("win_rate_simple", win_rate_simple)
        # logger.record_tabular("lose_rate_simple", lose_rate_simple)
        # logger.record_tabular("tie_rate_simple", tie_rate_simple)

        # logger.record_tabular("win_rate_simple_3", win_rate_simple_3)
        # logger.record_tabular("lose_rate_simple_3", lose_rate_simple_3)
        # logger.record_tabular("tie_rate_simple_3", tie_rate_simple_3)

        # logger.record_tabular("win_rate_static_3", win_rate_static_3)
        # # logger.record_tabular("lose_rate_static", lose_rate_static)
        # # logger.record_tabular("tie_rate_static", tie_rate_static)

        logger.record_tabular("win_rate_nobomb", win_rate_nobomb)

        model.save(save_path + 'updates' + str(epoch))
        logger.dump_tabular()

    return model