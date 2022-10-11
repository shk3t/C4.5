import math


def probability(value, column):
    return (column == value).sum() / column.shape[0]


def info(column):
    sum = 0
    for value in column.unique():
        prob = probability(value, column)
        sum += prob * math.log2(prob)
    return -sum


def infox(x_column, y_column):
    sum = 0
    for value in x_column.unique():
        y_specific = y_column[x_column == value]
        sum += y_specific.shape[0] * info(y_specific)
    return sum / x_column.shape[0]


def split_info(x_column):
    sum = 0
    for value in x_column.unique():
        rat = probability(value, x_column)
        sum += rat * math.log2(rat)
    return -sum


def gain_ratio(x_column, y_column):
    if len(x_column.unique()) == 1:
        return 0
    return (info(y_column) - infox(x_column, y_column)) / split_info(x_column)
