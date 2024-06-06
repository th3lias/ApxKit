import torch


class LS_NN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LS_NN, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear(x)