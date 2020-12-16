import pickle
from copy import deepcopy
import random


def square(list):
    return map(lambda x: x ** 2, list)


def save_obj(obj, name):
    with open('./drive/My Drive/AnC/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('./drive/My Drive/BSc Thesis/AnC/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def random_step_single(weight):
    return weight + random.uniform(-1 - weight, 1 - weight)


def input_random_step(input_matrix, dimensions, window):
    matrix = deepcopy(input_matrix)
    for i in range(dimensions):
        for j in range(window):
            matrix[i][i + dimensions * j] = random_step_single(matrix[i][i + dimensions * j])
    return matrix