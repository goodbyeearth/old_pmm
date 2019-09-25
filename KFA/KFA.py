import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
from torch import autograd

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import KFA.data as data
from KFA.model import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 100
task_number = 50
mnist_size = (28, 28)

random_seed = 1
torch.manual_seed(random_seed)

train_datasets, test_datasets = data.get_datasets(random_seed=random_seed,
                                                  task_number=task_number,
                                                  batch_size_train=batch_size_train,
                                                  batch_size_test=batch_size_test)

torch.set_default_tensor_type('torch.cuda.FloatTensor')

kfa_model = KFAModel(28 * 28, 10)
kfa_model.cuda()

optimizer = optim.SGD(kfa_model.parameters(), lr=learning_rate,
                      momentum=momentum)
log = True
test_acc = []

for n_datasets, train_loader in enumerate(train_datasets[:10], 1):
    kfa_model.train()
    # standart training
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            output = kfa_model(data.view(len(data), -1).cuda())
            criteriton = nn.CrossEntropyLoss()
            ce_loss = criteriton(output, target.cuda())
            kfa_loss = calc_kfa_loss()
            loss = ce_loss + 3 * kfa_loss
            #             loss = ce_loss

            loss.backward()
            optimizer.step()

            if log and batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), kfa_loss.item()))

        test_acc.append(test(kfa_model, test_datasets[:n_datasets]))

    #     update cycle
    n_data = 0
    kfa_model.store_factors = True
    prev_H = []  # kfa_model.H_factors.clone()
    prev_Q = []  # kfa_model.Q_factors.clone()
    for ind in range(len(kfa_model.H_factors)):
        prev_H.append(kfa_model.H_factors[ind].data.clone().detach())
        prev_Q.append(kfa_model.Q_factors[ind].data.clone().detach())

    for data, target in tqdm(train_loader):
        #         print(prev_Q)
        kfa_model.WB_factors = []
        for layer in kfa_model.layers:
            if isinstance(layer, nn.Linear):
                kfa_model.WB_factors.append(torch.zeros(layer.in_features, layer.in_features))

        data = data.view(-1, 28 * 28).cuda()
        output = kfa_model(data)

        # backward
        E = F.cross_entropy(output, target.cuda())
        dE_dh = autograd.grad(E, output, create_graph=True)[0]
        H_L = torch.stack([autograd.grad(dE_dhi, output, retain_graph=True)[0][0] for dE_dhi in dE_dh[0]])
        #     print(H_L.shape)
        # in reversed order
        H_factors = [H_L]

        for ind, BW_t in enumerate(reversed(kfa_model.WB_factors[1:])):
            H_next = H_factors[-1]
            #         print(BW_t.shape, H_next.shape)
            H_factors.append(BW_t @ H_next @ torch.transpose(BW_t, dim1=0, dim0=1))
        #             if torch.all(torch.eig(kfa_model.H_factors[-1])[0][:, 0] > 0) == 0:
        #                 print(BW_t)
        #                 break
        for l, Hl in enumerate(reversed(H_factors)):
            kfa_model.H_factors[l] += Hl

        n_data += len(data)
    #         break

    for ind in range(len(kfa_model.Q_factors)):
        kfa_model.Q_factors[ind] /= n_data
        kfa_model.Q_factors[ind] += prev_Q[ind]
        kfa_model.H_factors[ind] /= n_data
        kfa_model.H_factors[ind] += prev_H[ind]

    kfa_model.prev_W = []
    for layer in kfa_model.layers:
        if isinstance(layer, nn.Linear):
            kfa_model.prev_W.append(layer.weight.data.clone())
    # end of Q, H update. Don't change during other cycle
    kfa_model.store_factors = False