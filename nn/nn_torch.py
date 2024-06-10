import torch

class LS_NN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, weights=None):
        super(LS_NN, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

        if weights is not None:
            # Check if the shape of weights matches the expected shape
            if weights.ndim == 1:
                weights = weights.reshape(output_dim, input_dim)

            elif not weights.shape == (input_dim,output_dim):
                raise ValueError(f"Expected weights of shape {(output_dim, input_dim)}, but got {weights.shape}")

            self.linear.weight = torch.nn.Parameter(torch.Tensor(weights))

    def forward(self, x):
        return self.linear(x)