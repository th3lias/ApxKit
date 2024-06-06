import numpy as np
from torch.utils.data import DataLoader
import torch

from genz.genz_function_types import GenzFunctionType
from genz.genz_functions import get_genz_function
from grid.grid_provider import GridProvider
from grid.grid_type import GridType
from interpolate.basis_types import BasisType
from interpolate.least_squares import LeastSquaresInterpolator
from nn.dataloader_torch import LS_Dataset
from nn.nn_torch import LS_NN
from utils.utils import calculate_num_points


def train(model, criterion, optimizer, scheduler, dataloader, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.float().to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')

        scheduler.step()
        print(f'Epoch: {epoch}, Learning Rate: {scheduler.get_last_lr()}')

    print('Finished Training')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dim = 5
    scale = 5

    n_samples = calculate_num_points(dimension=dim, scale=scale)

    c = np.random.uniform(low=0.0, high=1.0, size=dim)
    c = c / np.sum(c) * dim
    w = np.random.uniform(low=0.0, high=1.0, size=dim)

    f = get_genz_function(c=c, w=w, d=dim, function_type=GenzFunctionType.OSCILLATORY)

    gp = GridProvider(dimension=dim, lower_bound=0.0, upper_bound=1.0)
    multiplier = np.log(n_samples)
    data = gp.generate(scale=scale, grid_type=GridType.RANDOM_CHEBYSHEV, multiplier=multiplier)

    y = f(data.grid)

    ls = LeastSquaresInterpolator(include_bias=True, basis_type=BasisType.CHEBYSHEV, grid=data)
    ls.basis = ls._build_basis()
    basis = ls.basis

    f_hat_exact = ls.interpolate(f)

    dataset = LS_Dataset(basis, y)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)

    model = LS_NN(input_dim=basis.shape[1], output_dim=1)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.995)

    criterion = torch.nn.MSELoss()

    train(model, criterion, optimizer, scheduler, dataloader, num_epochs=1000, device=device)

    test_grid = np.random.uniform(low=0.0, high=1.0, size=(50, dim))

    y_hat_exact = f_hat_exact(test_grid)

    test_grid = ls._build_basis(basis_type=BasisType.CHEBYSHEV, grid=test_grid)

    test_grid = torch.tensor(test_grid, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_hat_nn = model(test_grid).cpu().numpy()

    print(f'Exact: {y_hat_exact} vs NN: {y_hat_nn}')
