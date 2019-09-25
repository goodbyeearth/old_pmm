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

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)

        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy


        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.act = eval_model.step
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)


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

    '''

    '''


    set_global_seeds(seed)
    # Get the nb of env
    if network == 'cnn':
        network_kwargs['one_dim_bias'] = True
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    # Calculate the batch_size
    nbatch = nenvs*nsteps

    # Start total timer
    tstart = time.time()
    # win_rate_static_0,_,_ = test_CNN(model)
    # print(win_rate_static_0)
    observation = []
    action = []
    for update in range(1, total_timesteps//nbatch+1):
    #采集100局的数据
    # for update in range(100):
        # Get mini batch of experiences
        print(update)
        obs, states, rewards, masks, actions, values,output = runner.run()
        observation.append(obs)
        action.append(output)

    obs = np.concatenate(observation)
    act = np.concatenate(action).reshape(-1,6)
    np.savez_compressed(save_path,observation=obs,action=act)
    # np.savez_compressed(save_path,observation=obs,action=act)
    # joblib.dump(observation,'distillation_data/static_obs')
    # joblib.dump(action,'distillation_data/static_act')

    return model
