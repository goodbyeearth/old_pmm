import joblib
import numpy as np

def all_fisher_overlap(filename1,filename2):
    # -----------------mnist------------------------------#
    task1 = joblib.load(filename1)
    task2 = joblib.load(filename2)
    static = []
    simple = []
    for i in range(len(task1)):
        static.append(task1[i].reshape(-1))
        simple.append(task2[i].reshape(-1))
    # -----------------mnist------------------------------#

    #-----------------pommmerman------------------------------#
    # static_fm = joblib.load('fisher_data/FM30000_static.npz')
    # simple_fm = joblib.load('fisher_data/FM10000_simple.npz')
    #
    # static = []
    # simple = []
    # for i in range(len(static_fm)):
    #     static.append(static_fm[i].reshape(-1))
    #     simple.append(simple_fm[i].reshape(-1))
    # normalize fisher matrix
    # -----------------pommmerman------------------------------#
    x = np.hstack(static)
    x /= sum(x)
    y = np.hstack(simple)
    y /= sum(y)

    # sqrt fisher matrix
    # distance = np.linalg.norm(np.sqrt(x) - np.sqrt(y),ord='fro')/2
    matrix = np.sqrt(x) - np.sqrt(y)
    distance = np.sqrt(np.sum(np.square(matrix)))/2
    f_overlap = 1 - distance
    print('fisher overlap:', f_overlap)

def each_layer_fisher(filename1,filename2):
    task1 = joblib.load(filename1)
    task2 = joblib.load(filename2)
    static = []
    simple = []
    j = 0
    for i in range(len(task1)):
        static.append(task1[i].reshape(-1))
        simple.append(task2[i].reshape(-1))
        j += 1
        if j % 2 == 0 and j > 0:
            x = np.hstack(static)
            x /= sum(x)
            y = np.hstack(simple)
            y /= sum(y)
            matrix = np.sqrt(x) - np.sqrt(y)
            distance = np.sqrt(np.sum(np.square(matrix))) / 2
            f_overlap = 1 - distance
            print('fisher overlap:', f_overlap)
            static = []
            simple = []


if __name__ == '__main__':
    # filename1 = 'fisher_data/1'
    filename1 = 'fisher_data/fm'
    # filename2 = 'fisher_data/2'
    filename2 = 'fisher_data/fm2'
    each_layer_fisher(filename1,filename2)
    # all_fisher_overlap(filename1, filename2)