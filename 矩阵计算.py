import numpy as np
from functools import reduce
import joblib
import tensorflow as tf

# a = [[3, 3, 19, 32],[1, 32, 1, 1],[3, 3, 32, 64],[1, 64, 1, 1],[3, 3, 64, 64],[1, 64, 1, 1]]
# array = np.array(a, dtype='int32')
#
# input= np.array([1,2,3,4]).reshape(-1,1)
# # b = tf.constant([1,2,3,4])
# # b = tf.reshape(b,[-1,1])
# b = tf.placeholder(tf.int32,shape=[4,1])
# ewc = 0
# for i in range(6):
#     # a[i] = a[i].reshape(1,-1)
#     ewc += tf.matmul(array[i].reshape(1,-1), b)
# sess = tf.InteractiveSession()
# writer = tf.summary.FileWriter("test",sess.graph)
# print(sess.run(ewc,feed_dict={b:input}))
# writer.close()

fisher = joblib.load('fisher_cnn_mnist')
lamb = 0.03
gamma = 1e-3

F = [fisher]

efficient = np.sqrt(lamb + gamma)
for i in range(0, len(F[0]), 2):
    # if i % 2 == 0:
    L = F[0][i].shape[0] * np.ndarray.trace(F[0][i])
    R = F[0][i + 1].shape[0] * np.ndarray.trace(F[0][i + 1])
    pi = np.sqrt(L / R)
    identity1 = np.identity(F[0][i].shape[0])
    identity2 = np.identity(F[0][i + 1].shape[0])
    F[0][i] = (F[0][i] + pi * efficient * identity1).astype(np.float32)
    F[0][i + 1] = (F[0][i + 1] + 1 / pi * efficient * identity2).astype(np.float32)

joblib.dump(F[0],'fisher_damping')
