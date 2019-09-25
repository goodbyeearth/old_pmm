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

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)

        OUTPUT = tf.placeholder(tf.float32, [None,6])
        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        pi = tf.nn.softmax(OUTPUT)
        train_model_pi = tf.nn.softmax(train_model.pi)

        KL = pi * tf.log(pi/train_model_pi)
        KL_loss = tf.reduce_mean(tf.reduce_sum(KL,1))
        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")
        print(len(params))

        #compute elastic weight consolitation loss
        # star_list = []
        # for i in params[:-2]:
        #     star_list.append(star_param[i.name])
        # ewc_loss = 0
        # for v in range(len(star_list)):
        ewc_loss = 0
        for v in range(len(params)-2):
            # if v == 6:
            ewc_loss += tf.reduce_sum(tf.multiply(fisher_matrix[v].astype(np.float32), tf.square(params[v] - star_param[v])))

            # else:
            #     p = tf.to_double(params[v] - star_param[v])
            #
            #     ewc_loss += tf.matmul(tf.matmul(tf.reshape(p,(1,-1)),fisher_matrix[v]),tf.reshape(p,(-1,1)))

        loss1 = KL_loss * ent_coef
        #
        loss2 = ewc_loss * (lam/2)
        # loss = ewc_loss * (lam/2) + KL_loss * ent_coef

        # 2. Calculate the gradients
        grads = tf.gradients(loss1, params)

        grads = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        # trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, decay=alpha, epsilon=epsilon)
        # trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        trainer = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9,use_nesterov=True)
        _train = trainer.apply_gradients(grads)
        #
        #
        grads2 = tf.gradients(loss2,params)
        grads2 = list(zip(grads2, params))
        # trainer2 = tf.train.RMSPropOptimizer(learning_rate=7e-4, decay=alpha, epsilon=epsilon)
        # trainer2 = tf.train.AdamOptimizer(learning_rate=1e-4)
        trainer2 = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9, use_nesterov=True)
        _train2 = trainer2.apply_gradients(grads2)



        def train(obs, actions,opt):

            td_map = {train_model.X:obs, OUTPUT:actions}

            if opt == 0 :
                kl,_ = sess.run([loss1,_train],td_map)
                return kl
            else:
                ewc,_ = sess.run([loss2,_train2],td_map)
                return ewc

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
    # Get the nb of env
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    # print('--------------------------------',load_fm)
    # if load_fm is not None:
    # fisher_matrix = joblib.load('fisher_matrix/FM30000_static.npz')
    fisher_matrix = joblib.load('fisher_matrix/fm')
    print('fisher matrix done')
    # else:
    #     fisher_matrix = None
    star_param = joblib.load('initial_parameter/26000')
    # if load_path is not None:
    #     star_param = joblib.load(load_path)
    # else:
    #     star_param = None

    batch_size = 400
    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                  lrschedule=lrschedule,fisher_matrix=fisher_matrix,star_param=star_param,lam=100,batch_size=batch_size)
    if load_path is not None:

        model.load(load_path)


    # load data
    data = np.load('distillation_data/simple_agent.npz')
    observation = data['observation']
    action = data['action']
    data_size = action.shape[0]
    print('size',data_size)

    for epoch in range(200):
        inds = np.arange(data_size)
        np.random.shuffle(inds)
        nbatch = 0
        kl_loss = 0
        ewc_loss = 0
        for start in range(0, data_size, batch_size):
            end = start + batch_size
            mb_inds = inds[start:end]
            kl= model.train(observation[mb_inds],action[mb_inds],0)
            ewc = model.train(observation[mb_inds],action[mb_inds],1)
            kl_loss += kl
            ewc_loss += ewc
            nbatch += 1
            if nbatch % 25 == 0:
                print('Average kl_loss at epoch {0}: {1}'.format(epoch, kl_loss / nbatch))
                print('Average ewc_loss at epoch {0}: {1}'.format(epoch, ewc_loss / nbatch))
                logger.record_tabular("kl_loss", kl)
                logger.record_tabular("ewc_loss", ewc)
                logger.dump_tabular()

        win_rate_simple, lose_rate_simple, tie_rate_simple, win_rate_static, lose_rate_static, tie_rate_static = test_TwoAgent(model)

        logger.record_tabular("win_rate_simple", win_rate_simple)
        # logger.record_tabular("lose_rate_simple", lose_rate_simple)
        # logger.record_tabular("tie_rate_simple", tie_rate_simple)

        logger.record_tabular("win_rate_static", win_rate_static)
        # logger.record_tabular("lose_rate_static", lose_rate_static)
        # logger.record_tabular("tie_rate_static", tie_rate_static)
        model.save(save_path + 'updates' + str(epoch))
        logger.dump_tabular()

    return model