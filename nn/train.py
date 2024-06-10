import os.path

import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter

from genz.genz_function_types import GenzFunctionType
from genz.genz_functions import get_genz_function
from grid.grid_provider import GridProvider
from grid.grid_type import GridType
from interpolate.basis_types import BasisType
from interpolate.least_squares import LeastSquaresInterpolator
from nn.dataloader_torch import LS_Dataset
from nn.nn_torch import LS_NN
from utils.utils import calculate_num_points


def train(model, criterion, optimizer, scheduler, dataloader, num_epochs, device, writer):
    model.to(device)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.float().to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        # Log the model parameters and gradients (optional)
        for name, param in model.named_parameters():
            writer.add_histogram(f'{name}/weights', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'{name}/gradients', param.grad, epoch)

        if epoch % 25 == 0:
            print(f'Epoch: {epoch}, Loss: {avg_loss}, Learning Rate: {scheduler.get_last_lr()[0]}')

    print('Finished Training')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dim = 4
    scale = 4
    batch_size = 64
    n_epochs = 1000

    n_samples = calculate_num_points(dimension=dim, scale=scale)

    c = np.random.uniform(low=0.0, high=1.0, size=dim)
    c = c / np.sum(c) * dim
    w = np.random.uniform(low=0.0, high=1.0, size=dim)

    f = get_genz_function(c=c, w=w, d=dim, function_type=GenzFunctionType.CORNER_PEAK)

    gp = GridProvider(dimension=dim, lower_bound=0.0, upper_bound=1.0)
    multiplier = np.log(n_samples)
    data = gp.generate(scale=scale, grid_type=GridType.RANDOM_CHEBYSHEV, multiplier=multiplier)

    y = f(data.grid)

    ls = LeastSquaresInterpolator(include_bias=True, basis_type=BasisType.CHEBYSHEV, grid=data)
    ls.basis = ls._build_basis()
    basis = ls.basis


    f_hat_exact = ls.interpolate(f)
    coeff = ls.coeff

    dataset = LS_Dataset(basis, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LS_NN(input_dim=basis.shape[1], output_dim=1, weights = coeff)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)

    criterion = torch.nn.MSELoss()

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join('logs', 'tensorboard'))

    train(model, criterion, optimizer, scheduler, dataloader, num_epochs=n_epochs, device=device, writer=writer)

    # Close the writer
    writer.close()

    test_grid = np.random.uniform(low=0.0, high=1.0, size=(50, dim))

    y_hat_exact = f_hat_exact(test_grid)

    test_grid = ls._build_basis(basis_type=BasisType.CHEBYSHEV, grid=test_grid)

    test_grid = torch.tensor(test_grid, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_hat_nn = model(test_grid).cpu().numpy()

    print(f'Exact: {y_hat_exact} vs NN: {y_hat_nn}')

    print('_'*100)

    # get model weights and compare it with the coefficients from the exact solution
    weights = model.state_dict()
    weights = np.sort(weights['linear.weight'].cpu().numpy().flatten())
    print(f'Weights: {weights}')
    print(f'Coefficients: {np.sort(ls.coeff)}')
