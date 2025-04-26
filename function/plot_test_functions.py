import numpy as np
import matplotlib.pyplot as plt

from function import ParametrizedFunctionProvider as provider
from function import FunctionType


def plot_multiple_r2_to_r_functions_with_captions(funcs, xlim=(0, 1), ylim=(0, 1), resolution=150,
                                                  translation_dict: dict = None):
    """
    Plots multiple f: R^2 -> R functions in a 3x3 grid with captions under each plot.
    """

    x1 = np.linspace(*xlim, resolution)
    x2 = np.linspace(*ylim, resolution)
    grid_x, grid_y = np.meshgrid(x1, x2)

    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    fig = plt.figure(figsize=(15, 15))

    captions = [f.name for f in functions]

    for i, func in enumerate(funcs):
        z_flat = func(points)
        z = z_flat.reshape(resolution, resolution)
        ax = fig.add_subplot(3, 3, i + 1, projection='3d')
        ax.plot_surface(grid_x, grid_y, z, cmap='jet', edgecolor='k', linewidth=0.3)

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f(x_1, x_2)$')

        caption = captions[i]

        # translate function name
        if caption in translation_dict:
            caption = translation_dict[caption] if translation_dict else caption

        ax.text2D(0.5, -0.1, caption, transform=ax.transAxes, ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('function_visualization.pdf')
    plt.show()
    plt.close()


if __name__ == '__main__':
    f1 = provider().get_function(FunctionType.ZHOU, d=2, c=np.array([1, 1]), w=np.array([0.35, 0.65]))
    f2 = provider().get_function(FunctionType.CONTINUOUS, d=2, c=np.array([1, 1]), w=np.array([0.5, 0.5]))
    f3 = provider().get_function(FunctionType.CORNER_PEAK, d=2, c=np.array([1, 1]), w=np.array([0.5, 0.5]))
    f4 = provider().get_function(FunctionType.DISCONTINUOUS, d=2, c=np.array([1, 1]), w=np.array([0.5, 0.5]))
    f5 = provider().get_function(FunctionType.GAUSSIAN, d=2, c=np.array([1, 1]), w=np.array([0.5, 0.5]))
    f6 = provider().get_function(FunctionType.MOROKOFF_CALFISCH_1, d=2, c=np.array([1, 1]), w=np.array([0, 0]))
    f7 = provider().get_function(FunctionType.G_FUNCTION, d=2, c=np.array([-0.5, 0]), w=np.array([0, 0]))
    f8 = provider().get_function(FunctionType.OSCILLATORY, d=2, c=np.array([1, 1]), w=np.array([0.5, 0.5]))
    f9 = provider().get_function(FunctionType.PRODUCT_PEAK, d=2, c=np.array([1, 1]), w=np.array([0.5, 0.5]))

    functions = [f1, f2, f3, f4, f5, f6, f7, f8, f9]

    translation_dict = {
        "Continuous": "Continuous",
        "Corner Peak": "Corner Peak",
        "Discontinuous": "Discontinuous",
        "Gaussian": "Gaussian",
        "G Function": "Modified Ridge Product",
        "Morokoff Calfisch 1": "Modified Geometric Mean",
        "Oscillatory": "Oscillatory",
        "Product Peak": "Product Peak",
        "Zhou": "Bimodal Gaussian"
    }

    plot_multiple_r2_to_r_functions_with_captions(functions, translation_dict=translation_dict)
