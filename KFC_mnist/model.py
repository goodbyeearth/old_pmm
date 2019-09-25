import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display
import joblib
# variable initialization functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

#池化层
def max_pool_2_2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

class Model:
    def __init__(self, x, y_):

        in_dim = int(x.get_shape()[1]) # 784 for MNIST
        out_dim = int(y_.get_shape()[1]) # 10 for MNIST

        self.x = x # input placeholder

        # simple 2-layer network
        W1 = weight_variable([in_dim,100])
        b1 = bias_variable([100])

        W2 = weight_variable([100,out_dim])
        b2 = bias_variable([out_dim])

        h1 = tf.nn.relu(tf.matmul(x,W1) + b1) # hidden layer
        self.y = tf.matmul(h1,W2) + b2 # output layer

        self.var_list = [W1, b1, W2, b2]

        # vanilla single-task loss
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
        self.set_vanilla_loss()

        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.ewc_loss = 0
        # self.star_vars = []
        self.F_accum = []

    # def compute_fisher(self, imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10,task=None):
    #     # computer Fisher information for each parameter
    #
    #     # initialize Fisher information for most recent task
    #     self.F_accum = []
    #     for v in range(len(self.var_list)):
    #         self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))
    #
    #     # sampling a random class from softmax
    #     probs = tf.nn.softmax(self.y)
    #     class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])
    #
    #     if(plot_diffs):
    #         # track differences in mean Fisher info
    #         F_prev = deepcopy(self.F_accum)
    #         mean_diffs = np.zeros(0)
    #
    #     for i in range(num_samples):
    #         # select random input image
    #         im_ind = np.random.randint(imgset.shape[0])
    #         # compute first-order derivatives
    #         ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
    #         # square the derivatives and add to total
    #         for v in range(len(self.F_accum)):
    #             self.F_accum[v] += np.square(ders[v])
    #
    #         #画图(判断fisher matrix是否收敛)
    #         if(plot_diffs):
    #             if i % disp_freq == 0 and i > 0:
    #                 # recording mean diffs of F
    #                 F_diff = 0
    #                 for v in range(len(self.F_accum)):
    #                     F_diff += np.sum(np.absolute(self.F_accum[v]/(i+1) - F_prev[v]))
    #                 mean_diff = np.mean(F_diff)
    #                 mean_diffs = np.append(mean_diffs, mean_diff)
    #                 for v in range(len(self.F_accum)):
    #                     F_prev[v] = self.F_accum[v]/(i+1)
    #                 plt.plot(range(disp_freq+1, i+2, disp_freq), mean_diffs)
    #                 plt.xlabel("Number of samples")
    #                 plt.ylabel("Mean absolute Fisher difference")
    #                 display.display(plt.gcf())
    #                 display.clear_output(wait=True)
    #
    #     # divide totals by number of samples
    #     for v in range(len(self.F_accum)):
    #         self.F_accum[v] /= num_samples
    #     if task == 1:
    #         joblib.dump(self.F_accum,str(task))
    #     elif task == 2:
    #         joblib.dump(self.F_accum,str(task))

    def compute_fisher(self, imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10,task=None):
        # computer Fisher information for each parameter

        # initialize Fisher information for most recent task
        self.F_accum.append([])
        for v in range(len(self.var_list)):
            self.F_accum[-1].append(np.zeros(self.var_list[v].get_shape().as_list()))

        # sampling a random class from softmax
        probs = tf.nn.softmax(self.y)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(self.F_accum[-1])
            mean_diffs = np.zeros(0)

        for i in range(num_samples):
            # select random input image
            im_ind = np.random.randint(imgset.shape[0])
            # compute first-order derivatives
            ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
            # square the derivatives and add to total
            for v in range(len(self.F_accum[-1])):
                self.F_accum[-1][v] += np.square(ders[v])

            #画图(判断fisher matrix是否收敛)
            if(plot_diffs):
                if i % disp_freq == 0 and i > 0:
                    # recording mean diffs of F
                    F_diff = 0
                    for v in range(len(self.F_accum[-1])):
                        F_diff += np.sum(np.absolute(self.F_accum[-1][v]/(i+1) - F_prev[v]))
                    mean_diff = np.mean(F_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)
                    for v in range(len(self.F_accum[-1])):
                        F_prev[v] = self.F_accum[-1][v]/(i+1)
                    plt.plot(range(disp_freq+1, i+2, disp_freq), mean_diffs)
                    plt.xlabel("Number of samples")
                    plt.ylabel("Mean absolute Fisher difference")
                    display.display(plt.gcf())
                    display.clear_output(wait=True)

        # divide totals by number of samples
        for v in range(len(self.F_accum[-1])):
            self.F_accum[-1][v] /= num_samples
        if task == 1:
            joblib.dump(self.F_accum,str(task))
        elif task == 2:
            joblib.dump(self.F_accum,str(task))

    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())

    def store_param(self):
        self.star_vars.append([])
        for v in range(len(self.var_list)):
            self.star_vars[-1].append(self.var_list[v].eval())

    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))

    def set_vanilla_loss(self):
        # self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)
        # self.train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.cross_entropy)
        self.train_step = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9,use_nesterov=True).minimize(self.cross_entropy)



    # def update_ewc_loss(self, lam):
    #     # elastic weight consolidation
    #     # lam is weighting for previous task(s) constraints
    #
    #     if not hasattr(self, "ewc_loss"):
    #         self.ewc_loss = self.cross_entropy
    #
    #     for v in range(len(self.var_list)):
    #         self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
    #     self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.ewc_loss)

    def update_ewc_loss(self, lam):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints

        self.ewc_loss = 0
        print('----------------fisher---------------',len(self.F_accum))
        print('----------------paramm---------------',len(self.star_vars))
        # for v in range(len(self.var_list)):
        #     self.ewc_loss +=  tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
        for i in range(len(self.F_accum)):
            for v in range(len(self.var_list)):
                self.ewc_loss += tf.reduce_sum(tf.multiply(self.F_accum[i][v].astype(np.float32), tf.square(self.var_list[v] - self.star_vars[v])))

        self.loss = (lam/2) * self.ewc_loss + self.cross_entropy
        # self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)
        # self.train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)
        self.train_step = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True).minimize(self.loss)


class CNN_Model:
    def __init__(self, x, y_):

        self.x = x
        input = tf.reshape(self.x,[-1,28,28,1]) # input placeholder

        with tf.variable_scope('conv1'):
        # simple 2-layer network
            W1 = weight_variable([5,5,1,32])
            b1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(input ,W1) + b1)
            h_pool1 = max_pool_2_2((h_conv1))

        with tf.variable_scope('conv2'):
            W2 = weight_variable([5,5,32,64])
            b2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W2) + b2)
            h_pool2 = max_pool_2_2(h_conv2)

        with tf.variable_scope('fc1'):
            h_pool2_flatten = tf.reshape(h_pool2, [-1,7*7*64])
            W3 = weight_variable(([7*7*64,1024]))
            b3 = bias_variable([1024])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flatten,W3) + b3)

        with tf.variable_scope('output'):
            W4 = weight_variable(([1024,10]))
            b4 = bias_variable([10])
            self.y = tf.matmul(h_fc1,W4) + b4

        self.var_list = [W1, b1, W2, b2, W3, b3, W4, b4]

        # vanilla single-task loss
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=self.y))
        self.set_vanilla_loss()

        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.ewc_loss = 0
        # self.star_vars = []
        self.F_accum = []

    # def compute_fisher(self, imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10,task=None):
    #     # computer Fisher information for each parameter
    #
    #     # initialize Fisher information for most recent task
    #     self.F_accum = []
    #     for v in range(len(self.var_list)):
    #         self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))
    #
    #     # sampling a random class from softmax
    #     probs = tf.nn.softmax(self.y)
    #     class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])
    #
    #     if(plot_diffs):
    #         # track differences in mean Fisher info
    #         F_prev = deepcopy(self.F_accum)
    #         mean_diffs = np.zeros(0)
    #
    #     for i in range(num_samples):
    #         # select random input image
    #         im_ind = np.random.randint(imgset.shape[0])
    #         # compute first-order derivatives
    #         ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
    #         # square the derivatives and add to total
    #         for v in range(len(self.F_accum)):
    #             self.F_accum[v] += np.square(ders[v])
    #
    #         #画图(判断fisher matrix是否收敛)
    #         if(plot_diffs):
    #             if i % disp_freq == 0 and i > 0:
    #                 # recording mean diffs of F
    #                 F_diff = 0
    #                 for v in range(len(self.F_accum)):
    #                     F_diff += np.sum(np.absolute(self.F_accum[v]/(i+1) - F_prev[v]))
    #                 mean_diff = np.mean(F_diff)
    #                 mean_diffs = np.append(mean_diffs, mean_diff)
    #                 for v in range(len(self.F_accum)):
    #                     F_prev[v] = self.F_accum[v]/(i+1)
    #                 plt.plot(range(disp_freq+1, i+2, disp_freq), mean_diffs)
    #                 plt.xlabel("Number of samples")
    #                 plt.ylabel("Mean absolute Fisher difference")
    #                 display.display(plt.gcf())
    #                 display.clear_output(wait=True)
    #
    #     # divide totals by number of samples
    #     for v in range(len(self.F_accum)):
    #         self.F_accum[v] /= num_samples
    #     if task == 1:
    #         joblib.dump(self.F_accum,str(task))
    #     elif task == 2:
    #         joblib.dump(self.F_accum,str(task))

    def compute_fisher(self, imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10,task=None):
        # computer Fisher information for each parameter

        # initialize Fisher information for most recent task
        self.F_accum.append([])
        for v in range(len(self.var_list)):
            self.F_accum[-1].append(np.zeros(self.var_list[v].get_shape().as_list()))

        # sampling a random class from softmax
        probs = tf.nn.softmax(self.y)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(self.F_accum[-1])
            mean_diffs = np.zeros(0)

        for i in range(num_samples):
            # select random input image
            im_ind = np.random.randint(imgset.shape[0])
            # compute first-order derivatives
            ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
            # square the derivatives and add to total
            for v in range(len(self.F_accum[-1])):
                self.F_accum[-1][v] += np.square(ders[v])

            #画图(判断fisher matrix是否收敛)
            if(plot_diffs):
                if i % disp_freq == 0 and i > 0:
                    # recording mean diffs of F
                    F_diff = 0
                    for v in range(len(self.F_accum[-1])):
                        F_diff += np.sum(np.absolute(self.F_accum[-1][v]/(i+1) - F_prev[v]))
                    mean_diff = np.mean(F_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)
                    for v in range(len(self.F_accum[-1])):
                        F_prev[v] = self.F_accum[-1][v]/(i+1)
                    plt.plot(range(disp_freq+1, i+2, disp_freq), mean_diffs)
                    plt.xlabel("Number of samples")
                    plt.ylabel("Mean absolute Fisher difference")
                    display.display(plt.gcf())
                    display.clear_output(wait=True)

        # divide totals by number of samples
        for v in range(len(self.F_accum[-1])):
            self.F_accum[-1][v] /= num_samples
        if task == 1:
            joblib.dump(self.F_accum,str(task))
        elif task == 2:
            joblib.dump(self.F_accum,str(task))

    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())

    def store_param(self):
        self.star_vars.append([])
        for v in range(len(self.var_list)):
            self.star_vars[-1].append(self.var_list[v].eval())

    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))

    def set_vanilla_loss(self):
        # self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)
        # self.train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.cross_entropy)
        self.train_step = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9,use_nesterov=True).minimize(self.cross_entropy)



    # def update_ewc_loss(self, lam):
    #     # elastic weight consolidation
    #     # lam is weighting for previous task(s) constraints
    #
    #     if not hasattr(self, "ewc_loss"):
    #         self.ewc_loss = self.cross_entropy
    #
    #     for v in range(len(self.var_list)):
    #         self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
    #     self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.ewc_loss)

    def update_ewc_loss(self, lam):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints

        self.ewc_loss = 0
        # for v in range(len(self.var_list)):
        #     self.ewc_loss +=  tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
        for i in range(len(self.F_accum)):
            for v in range(len(self.var_list)):
                self.ewc_loss += tf.reduce_sum(tf.multiply(self.F_accum[i][v].astype(np.float32), tf.square(self.var_list[v] - self.star_vars[v])))

        self.loss = (lam/2) * self.ewc_loss + self.cross_entropy
        # self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)
        # self.train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)
        self.train_step = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True).minimize(self.loss)
