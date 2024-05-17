import numpy as np


def _remove_almost_identical_rows(arr: np.ndarray, tol=1e-8):
    """
    This method is only reference for testing purposes. It should not be used in production.
    :param arr:
    :param tol:
    :return:
    """
    unique_rows = [arr[0]]
    for row in arr[1:]:
        if not any(np.allclose(row, unique_row, atol=tol) for unique_row in unique_rows):
            unique_rows.append(row)
    return np.array(unique_rows)


def _remove_duplicates_squared_memory(arr: np.ndarray, tol: np.float32 = np.float32(1e-8)):
    """
    This method is only reference for testing purposes. It should not be used in production.
    :param arr:
    :param tol:
    :return:
    """
    if arr.size == 0:
        return arr
    diffs = np.sqrt(((arr[:, np.newaxis] - arr[np.newaxis, :]) ** 2).sum(axis=2))
    close = diffs <= tol
    not_dominated = ~np.any(np.triu(close, k=1), axis=0)
    unique_rows = arr[not_dominated]
    return unique_rows


def _remove_duplicates_linear_memory_naive(arr: np.ndarray, tol: np.float32 = np.float32(1e-8)):
    """
    This method is only reference for testing purposes. It should not be used in production.
    :param arr:
    :param tol:
    :return:
    """
    if arr.size == 0:
        return arr

    unique_rows = []
    # Iterate over each row
    for row in arr:
        # Compute the distance from the current row to all unique rows
        if unique_rows:
            diffs = np.linalg.norm(np.array(unique_rows) - row, axis=1)
            # Check if there is any row in the unique_rows close to the current row
            if not np.any(diffs <= tol):
                unique_rows.append(row)
        else:
            unique_rows.append(row)

    return np.array(unique_rows)
