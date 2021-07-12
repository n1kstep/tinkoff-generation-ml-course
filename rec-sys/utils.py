import numpy as np


def euclidean_distance(x: np.array, y: np.array) -> float:
    """
    Calculate euclidean distance between points x and y
    Args:
        x, y: two points in Euclidean n-space
    Returns:
        Length of the line segment connecting given points
    """
    return np.sqrt(np.dot(x - y, x - y))


def euclidean_similarity(x: np.array, y: np.array) -> float:
    """
    Calculate euclidean similarity between points x and y
    Args:
        x, y: two points in Euclidean n-space
    Returns:
        Similarity between points x and y
    """
    return 1 / (1 + euclidean_distance(x, y))


def pearson_similarity(x: np.array, y: np.array) -> float:
    """
    Calculate a Pearson correlation coefficient given 1-D data arrays x and y
    Args:
        x, y: two points in n-space
    Returns:
        Pearson correlation between x and y
    """
    x_ = x - np.mean(x)
    y_ = y - np.mean(y)
    sq = np.sqrt(np.dot(x_, x_) * np.dot(y_, y_))
    if sq == 0:
        return np.nan
    else:
        return np.dot(x_, y_) / sq


def apk(actual: np.array, predicted: np.array, k: int = 10) -> float:
    """
    Compute the average precision at k
    Args:
        actual: a list of elements that are to be predicted (order doesn't matter)
        predicted: a list of predicted elements (order does matter)
        k: the maximum number of predicted elements
    Returns:
        The average precision at k over the input lists
    """
    apk_sum = 0
    for i in range(k):
        apk_sum += len(set(actual).intersection(set(predicted[:i + 1]))) / len(predicted[:i + 1]) * (
                predicted[i] in actual)

    return apk_sum / k


def mapk(actual: np.array, predicted: np.array, k: int = 10) -> float:
    """
    Compute the mean average precision at k
    Args:
        actual: a list of lists of elements that are to be predicted
        predicted: a list of lists of predicted elements
        k: the maximum number of predicted elements
    Returns:
        The mean average precision at k over the input lists
    """
    mapk_sum = 0
    for i in range(len(actual)):
        mapk_sum += apk(actual[i], predicted[i], k)

    return mapk_sum / len(actual)
