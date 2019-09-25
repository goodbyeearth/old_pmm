import torch.nn as nn
import torch
from torch import autograd

class KFAModel(nn.Module):
    def __init__(self, input_size, output_size,
                 hidden_size=100):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layers = nn.ModuleList([
            nn.Linear(self.input_size, self.hidden_size, bias=False), nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=False), nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size, bias=False)
        ])

        self.store_factors = False
        self.WB_factors = None

        # factors init
        self.Q_factors = []
        self.H_factors = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.Q_factors.append(torch.zeros(layer.in_features, layer.in_features))
                self.H_factors.append(torch.zeros(layer.out_features, layer.out_features))
        # optimal weights init with random
        self.prev_W = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.prev_W.append(layer.weight.data.clone())

    def forward(self, inputs):
        curr_linear_layer = 0
        for layer_ind, layer in enumerate(self.layers):
            # add bias before linear layer
            # BIAS
            #             if isinstance(layer, nn.Linear):
            #                 inputs = torch.cat([inputs, torch.ones(inputs.size(0), 1)],dim=1)
            output = layer(inputs)

            if self.store_factors:
                if isinstance(layer, nn.Linear):

                    self.Q_factors[curr_linear_layer] += torch.bmm(inputs.unsqueeze(2).data.clone(),
                                                                   inputs.unsqueeze(1).data.clone()).sum(dim=0)
                    self.WB_factors[curr_linear_layer] = torch.mm(self.WB_factors[curr_linear_layer],
                                                                  layer.weight.transpose(dim0=1, dim1=0).data.clone())
                    curr_linear_layer += 1
                elif isinstance(layer, nn.ReLU):
                    df_dh = autograd.grad(output, inputs, grad_outputs=torch.ones_like(inputs))[0]
                    #                     BIAS
                    #                     self.WB_factors[curr_linear_layer][:-1, :-1] = torch.diag(df_dh[0])
                    self.WB_factors[curr_linear_layer] = torch.diag(df_dh[0].data.clone())

            inputs = output
        return output

# loss
def calc_kfa_loss():
    loss = 0
    ind = 0
    for layer in kfa_model.layers:
        if isinstance(layer, nn.Linear):
            kfa_model.H_factors[ind].detach_()
            kfa_model.Q_factors[ind].detach_()
#             loss += 0.5 * torch.sum((layer.weight - kfa_model.prev_W[ind]).transpose(dim0=1, dim1=0) @ \
#                     kfa_model.H_factors[ind] @ \
#                     (layer.weight - kfa_model.prev_W[ind]) @ \
#                     kfa_model.Q_factors[ind])
            loss += 0.5 * (layer.weight - kfa_model.prev_W[ind]).view(1, -1) @(kfa_model.H_factors[ind] @ \
                            (layer.weight - kfa_model.prev_W[ind]) @ \
                            kfa_model.Q_factors[ind]).view(-1, 1)
            ind += 1
    return loss


def test(model, test_datasets):
    model.eval()
    test_loss = 0
    correct = 0
    all_data_size = 0

    with torch.no_grad():
        for test_loader in test_datasets:
            for data, target in test_loader:
                output = model(data.view(len(data), -1).cuda())
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred).cuda()).sum().item()
                all_data_size += len(data)
    print(100. * correct / all_data_size)
    return (100. * correct / all_data_size)

