import numpy as np
import matplotlib.pyplot as plt

from function import ParametrizedFunctionProvider as FP
from function import FunctionType


def plot_multiple_r2_to_r_functions_with_captions(funcs, xlim=(0, 1), ylim=(0, 1), resolution=250, captions=None):
    """
    Plots multiple f: R^2 -> R functions in a 3x3 grid with captions under each plot.

    Parameters:
    - funcs: list of functions
    - xlim, ylim: tuple, domain for plotting
    - resolution: int, mesh resolution
    - captions: list of str, captions for each subplot
    """

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

    x1 = np.linspace(*xlim, resolution)
    x2 = np.linspace(*ylim, resolution)
    X1, X2 = np.meshgrid(x1, x2)

    points = np.stack([X1.ravel(), X2.ravel()], axis=1)  # shape (N, 2)

    fig = plt.figure(figsize=(15, 15))

    for i, func in enumerate(funcs):
        Z_flat = func(points)
        Z = Z_flat.reshape(resolution, resolution)
        ax = fig.add_subplot(3, 3, i + 1, projection='3d')
        ax.plot_surface(X1, X2, Z, cmap='jet', edgecolor='k', linewidth=0.3)

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f(x_1, x_2)$')

        caption = captions[i] if captions and i < len(captions) else f'Function {i + 1}'

        # translate function name
        if caption in translation_dict:
            caption = translation_dict[caption]

        ax.text2D(0.5, -0.1, caption, transform=ax.transAxes, ha='center', fontsize=10)

    plt.rcParams['text.usetex'] = True

    plt.tight_layout()
    plt.savefig('function_visualization.pdf')
    plt.show()
    plt.close()

if __name__ == '__main__':

    f1 = FP().get_function(FunctionType.CONTINUOUS, d=2, c=np.array([1, 1]), w=np.array([0.5, 0.5]))
    f2 = FP().get_function(FunctionType.CORNER_PEAK, d=2, c=np.array([1, 1]), w=np.array([0.5, 0.5]))
    f3 = FP().get_function(FunctionType.DISCONTINUOUS, d=2, c=np.array([1, 1]), w=np.array([0.5, 0.5]))
    f4 = FP().get_function(FunctionType.GAUSSIAN, d=2, c=np.array([1, 1]), w=np.array([0.5, 0.5]))
    f5 = FP().get_function(FunctionType.G_FUNCTION, d=2, c=np.array([-0.5, 0]), w=np.array([0, 0]))
    f6 = FP().get_function(FunctionType.MOROKOFF_CALFISCH_1, d=2, c=np.array([1, 1]), w=np.array([0, 0]))
    f7 = FP().get_function(FunctionType.OSCILLATORY, d=2, c=np.array([1, 1]), w=np.array([0.5, 0.5]))
    f8 = FP().get_function(FunctionType.PRODUCT_PEAK, d=2, c=np.array([1, 1]), w=np.array([0.5, 0.5]))
    f9 = FP().get_function(FunctionType.ZHOU, d=2, c=np.array([1, 1]), w=np.array([0.35, 0.65]))

    functions = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
    captions = [f.name for f in functions]

    plot_multiple_r2_to_r_functions_with_captions(functions, captions=captions)

    # TODO: Make a own method for this and make a main method with RunTime Error
