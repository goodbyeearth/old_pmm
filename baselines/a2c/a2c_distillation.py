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
    def __init__(self, policy,
            ent_coef=1, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear',batch_size=None):

        sess = tf_util.get_session()
        nbatch = batch_size


        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):

            # train_model is used to train our network
            train_model = policy(nbatch, 1, sess)
            eval_model = policy(1,1,sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)

        OUTPUT = tf.placeholder(tf.float32, [None,6])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
        pi = tf.nn.softmax(OUTPUT)

        train_model_pi = tf.nn.softmax(train_model.pi2)
        # Policy loss
        # neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        # entropy_loss = tf.reduce_mean(neglogpac)

        KL = pi * tf.log(pi/train_model_pi)
        KL_loss = tf.reduce_mean(tf.reduce_sum(KL,1))



        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")
        # regularization_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(1e-4),
        #                                                              weights_list=params)
        loss = KL_loss * ent_coef
        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        grads = list(zip(grads, params))
        # 3. Make op for one policy and value update step of A2C
        # trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, decay=alpha, epsilon=epsilon)
        trainer = tf.train.AdamOptimizer(learning_rate=1e-3)

        _train = trainer.apply_gradients(grads)


        def train(obs, actions):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')

            td_map = {train_model.X:obs, OUTPUT:actions, train_model.keep_prob: 1.0}

            l, _ = sess.run(
                [loss, _train],
                td_map
            )
            return l

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
    assert save_path is not None
    # if network == 'cnn':
    #     network_kwargs['one_dim_bias'] = True
    # Get the nb of env
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    # Instantiate the model object (that creates step_model and train_model)
    batch_size = 1280
    model = Model(policy=policy, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule,batch_size=batch_size)
    if load_path is not None:
        model.load(load_path)

    #load data
    data = np.load('distillation_data/simple_agent.npz')
    observation = data['observation']
    action = data['action']

    # data1 = np.load('distillation_data/3FFA_agent_supple.npz')
    # a1 = data1['action']
    # o1 = data1['observation']
    #
    # data2 = np.load('distillation_data/Simple_nobomb_agent.npz')
    # o2 = data2['observation']
    # a2 = data2['action']
    #
    # action = np.concatenate([action,a1])
    # observation = np.concatenate([observation,o1])
    data_size = action.shape[0]
    print('size',data_size)

    for epoch in range(1,200):
        inds = np.arange(data_size)
        np.random.shuffle(inds)
        nbatch = 0
        total_loss = 0
        for start in range(0, data_size, batch_size):
            end = start + batch_size
            mb_inds = inds[start:end]
            loss = model.train(observation[mb_inds],action[mb_inds])
            total_loss += loss
            nbatch += 1
            if nbatch % 25 == 0:
                print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / nbatch))
                logger.record_tabular("loss", loss)
                logger.dump_tabular()

        win_rate_simple, lose_rate_simple, tie_rate_simple, win_rate_static, lose_rate_static, tie_rate_static = test_TwoAgent(model)
        # win_static = test_static(model)
        # win_nobomb = test_nobomb(model)
        # win_3ffa = test_3ffa(model)

        logger.record_tabular("win_static", win_rate_static)
        # logger.record_tabular("lose_rate_simple", lose_rate_simple)
        # logger.record_tabular("tie_rate_simple", tie_rate_simple)

        logger.record_tabular("win_rate_simple", win_rate_simple)
        # logger.record_tabular("lose_rate_static", lose_rate_static)
        # logger.record_tabular("tie_rate_static", tie_rate_static)
        # logger.record_tabular("win_3ffa", win_3ffa)
        model.save(save_path + 'updates' + str(epoch))
        logger.dump_tabular()

    return model

