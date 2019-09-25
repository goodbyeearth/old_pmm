import tensorflow as tf
import numpy as np
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display
from model import Model

def mnist_imshow(img):
    plt.imshow(img.reshape([28,28]), cmap="gray")
    plt.axis('off')

# return a new mnist dataset w/ pixels randomly permuted
def permute_mnist(mnist):
    perm_inds = np.arange(mnist.train.images.shape[1])
    np.random.shuffle(perm_inds)
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name) # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:,c] for c in perm_inds]))
    return mnist2


def plot_test_acc(plot_handles):
    plt.legend(handles=plot_handles, loc="center right")
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1)
    display.display(plt.gcf())
    display.clear_output(wait=True)


# train/compare vanilla sgd and ewc
def train_task(model, num_iter, disp_freq, trainset, testsets, x, y_, picture,lams=[0]):

    num_iter = int(num_iter)
    disp_freq = int(disp_freq)

    for l in range(len(lams)):
        # lams[l] sets weight on old task(s)
        # model.restore(sess)  # reassign optimal weights from previous training session
        if (lams[l] == 0):
            model.set_vanilla_loss()
            sess.run(tf.global_variables_initializer())
            model.restore(sess)

        else:
            model.update_ewc_loss(lams[l])
            sess.run(tf.global_variables_initializer())
            model.restore(sess)
        # initialize test accuracy array for each task
        test_accs = []
        ewc = []
        entropy = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(int(num_iter / disp_freq)))
        # train on current task
        for iter in range(num_iter):
            batch = trainset.train.next_batch(100)
            model.train_step.run(feed_dict={x: batch[0], y_: batch[1]})
            # if model.ewc_loss == 0:
            #     model.train_step.run(feed_dict={x: batch[0], y_: batch[1]})
            # else:
            #     _,ewc_loss,cross_entropy = sess.run([model.train_step,model.ewc_loss,model.cross_entropy],feed_dict={x: batch[0], y_: batch[1]})
            #     ewc.append(ewc_loss)
            #     entropy.append(cross_entropy)
            #这个下面是画图
            if iter % disp_freq == 0:
                plt.subplot(1, len(lams), l + 1)
                plots = []
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w','gold','goldenrod']
                for task in range(len(testsets)):
                    feed_dict = {x: testsets[task].test.images, y_: testsets[task].test.labels}
                    test_accs[task][iter // disp_freq] = model.accuracy.eval(feed_dict=feed_dict)
                    c = chr(ord('A') + task)
                    plt.grid()
                    plot_h, = plt.plot(range(1, iter + 2, disp_freq), test_accs[task][:iter // disp_freq + 1],
                                       colors[task], label="task " + c)

                    plots.append(plot_h)
                plot_test_acc(plots)
                if l == 0:
                    plt.title("vanilla sgd")
                else:
                    plt.title("ewc")
                plt.gcf().set_size_inches(len(lams) * 5, 3.5)
        plt.grid()
        plt.savefig(picture + '.jpg')
    plt.close()
        # if l == 1:
        #     plt.figure()
        #     plt.plot(ewc,c='r')
        #     plt.plot(entropy,c='y')
        #     plt.legend(['ewc_loss', 'cross_entropy'], loc='upper right', fontsize=10)
        #     plt.grid()
        #     plt.savefig('loss.jpg')

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
print(mnist.train.images.shape)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

model = Model(x, y_) # simple 2-layer network

sess.run(tf.global_variables_initializer())

###########train on task A, test on task A###############
train_task(model, 800, 20, mnist, [mnist], x, y_, '1',lams=[0])

###########compute fisher information matrix#############
model.compute_fisher(mnist.validation.images, sess, num_samples=200, plot_diffs=True)
#
# F_row_mean = np.mean(model.F_accum[0], 1)
# mnist_imshow(F_row_mean)
# plt.title("W1 row-wise mean Fisher")

#########permuting mnist for 2nd task################
mnist2 = permute_mnist((mnist))
# plt.subplot(1,2,1)
# mnist_imshow(mnist.train.images[5])
# plt.title("original task image")
# plt.subplot(1,2,2)
# mnist_imshow(mnist2.train.images[5])
# plt.title("new task image")
# plt.savefig('new_task.jpg')
#########save current optimal weights
model.star()

#########training 2nd task################
train_task(model,800,20,mnist2,[mnist,mnist2],x,y_,'2',lams=[0,15])
model.compute_fisher(mnist2.validation.images, sess, num_samples=200, plot_diffs=True)

mnist3 = permute_mnist((mnist))
model.star()
train_task(model,800,20,mnist3,[mnist,mnist2,mnist3],x,y_,'3',lams=[0,15])
model.compute_fisher(mnist3.validation.images, sess, num_samples=200, plot_diffs=True)

mnist4 = permute_mnist((mnist))
model.star()
train_task(model,800,20,mnist4,[mnist,mnist2,mnist3,mnist4],x,y_,'4',lams=[0,15])
model.compute_fisher(mnist4.validation.images, sess, num_samples=200, plot_diffs=True)

mnist5 = permute_mnist((mnist))
model.star()
train_task(model,800,20,mnist5,[mnist,mnist2,mnist3,mnist4,mnist5],x,y_,'5',lams=[0,15])
model.compute_fisher(mnist5.validation.images, sess, num_samples=200, plot_diffs=True)

mnist6 = permute_mnist((mnist))
model.star()
train_task(model,800,20,mnist6,[mnist,mnist2,mnist3,mnist4,mnist5,mnist6],x,y_,'6',lams=[0,15])
model.compute_fisher(mnist6.validation.images, sess, num_samples=200, plot_diffs=True)

mnist7 = permute_mnist((mnist))
model.star()
train_task(model,800,20,mnist7,[mnist,mnist2,mnist3,mnist4,mnist5,mnist6,mnist7],x,y_,'7',lams=[0,15])
model.compute_fisher(mnist7.validation.images, sess, num_samples=200, plot_diffs=True)

mnist8 = permute_mnist((mnist))
model.star()
train_task(model,800,20,mnist8,[mnist,mnist2,mnist3,mnist4,mnist5,mnist6,mnist7,mnist8],x,y_,'8',lams=[0,15])
model.compute_fisher(mnist8.validation.images, sess, num_samples=200, plot_diffs=True)

mnist9 = permute_mnist((mnist))
model.star()
train_task(model,800,20,mnist9,[mnist,mnist2,mnist3,mnist4,mnist5,mnist6,mnist7,mnist8,mnist9],x,y_,'9',lams=[0,15])
# model.compute_fisher(mnist9.validation.images, sess, num_samples=200, plot_diffs=True)