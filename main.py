import numpy as np
from scipy.stats import logistic
import random
import math
from time import time
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from src.csv_reader import read_csv, flatten_input
from src.utils import random_step_single, input_random_step, square

OUTPUT_NODES_NUMBER = 2


def f(weights, input):
    return logistic.cdf(weights.dot(input))


def apply_weights(input, weights):
    return f(weights, input)


def apply_weights2(input, weights):
    return np.tanh(weights.dot(input))


def classify(series, weights, input_weights, output_weights, true_outcome):
    """
    Classifies the whole series based on the sequence of window, where each window is classified separately and the
    most frequent class is chosen as the prediction

    """
    results = np.zeros(OUTPUT_NODES_NUMBER)
    error = 0
    for input in series:
        processed_input = apply_weights2(input, input_weights)
        map_output = apply_weights2(processed_input, weights)
        map_result = apply_weights(map_output, output_weights)
        error += evaluate(map_result, true_outcome)
        results = np.add(map_result, results)
    return results, error


def evaluate(prediction, true_outcome):
    return sum(square(np.subtract(prediction, true_outcome)))


def mse(weights, input_weights, output_weights, input_matrix, output_matrix):
    """
    Counts the wrong predictions of the time series (naming does not indicate mean squared error)

    """
    ret = 0
    for i, series in enumerate(input_matrix):
        classification_result, error = classify(series, weights, input_weights, output_weights, output_matrix[i])
        if np.argmax(np.array(output_matrix[i])) != np.argmax(np.array(classification_result)):
            ret += 1
    return ret


def mse2(weights, input_weights, output_weights, input_matrix, output_matrix):
    """
    Add the difference between the desired probability for a class and the predicted of the time series
    (naming does not indicate mean squared error)

    """
    ret = 0
    for i, series in enumerate(input_matrix):
        classification_result, error = classify(series, weights, input_weights, output_weights, output_matrix[i])
        ret += error
    return ret


def change_weights(start_weights, start_input_weights, start_output_weights,
                   start_energy, T, input_matrix, output_matrix, dimensions, window):
    """
    Performs the change of the state by shuffling the weights of all matrices and checking the new energy
    """
    new_weights = np.vectorize(random_step_single)(start_weights)
    new_output_weights = np.vectorize(random_step_single)(start_output_weights)
    new_input_weights = input_random_step(start_input_weights, dimensions, window)

    new_energy = mse(new_weights, new_input_weights, new_output_weights, input_matrix, output_matrix)
    diff = new_energy - start_energy

    if diff <= 0 or random.uniform(0, 1) < math.exp(-diff / T):
        return new_weights, new_input_weights, new_output_weights, new_energy
    else:
        return start_weights, start_input_weights, start_output_weights, start_energy


def decrease_temp(T, cool_parameter):
    return T * cool_parameter


def train(_file_type, _start_temp, _end_temp, _eq_number, _cool_parameter, _dimensions, _window, _stride):
    """
    Performs the training of the model with the optimization by simulated annealing

    """
    print("================")
    print("Train")
    input_matrix, output_matrix = read_csv('TRAIN', _file_type, _dimensions)
    input_matrix, _window = flatten_input(input_matrix, _window, _stride)
    # initialize the weights matrices
    weights = np.zeros((_dimensions, _dimensions))
    output_weights = np.zeros((OUTPUT_NODES_NUMBER, _dimensions))
    input_weights = np.zeros((_dimensions, (_window + 1) * _dimensions))

    # initialize the temperatures
    start_temp = _start_temp
    end_temp = _end_temp
    T = start_temp

    energy = mse(weights, input_weights, output_weights, input_matrix, output_matrix)

    # initial states
    energies = [energy]
    best_energies = [energy]

    best_weights = weights
    best_input_weights = input_weights
    best_output_weights = output_weights
    best_energy = energy

    # initialize optimization hyperparameters
    eq_number = _eq_number
    cool_parameter = _cool_parameter
    wait = 0

    start = time()
    # main optimization loop
    # minimizing the mse function by calling change_weights to find the best energy
    while T >= end_temp:
        print(T)
        # stay on the same temp (in the equilibrium) for eq_number iterations
        for _ in range(eq_number):
            weights, input_weights, output_weights, energy = change_weights(
                weights, input_weights,
                output_weights, best_energy,
                T, input_matrix, output_matrix,
                _dimensions, _window
            )
            wait += 1
            if energy < best_energy:
                best_energy = energy
                best_weights = weights
                best_input_weights = input_weights
                best_output_weights = output_weights
                wait = 0
            energies.append(energy)
            best_energies.append(best_energy)
        T = decrease_temp(T, cool_parameter)

    end = time()
    processing_time = end - start
    print("Processing time: ", processing_time)
    plt.plot(energies, label="Energy")
    plt.plot(best_energies, label="Best energy")
    plt.xlabel("Epochs")
    plt.ylabel("Wrong predictions")
    plt.legend()
    plt.show()

    return best_weights, best_input_weights, best_output_weights, energies, best_energies


def test(_file_type, best_weights, best_input_weights, best_output_weights, _dimensions, _window, _stride):
    """
    Tests the output of the model by applying the pre-trained weights to the input and evaluating the loss function

    """
    print("================")
    print("Test")
    input_matrix, output_matrix = read_csv('TEST', _file_type, _dimensions)
    input_matrix, _window = flatten_input(input_matrix, _window, _stride)

    predicted_output = []
    for i, input in enumerate(input_matrix):
        result, error = classify(input, best_weights, best_input_weights, best_output_weights, output_matrix[i])
        predicted_output.append(np.argmax(np.array(result)))

    np_output = np.array(output_matrix)
    true_output = np.argmax(np_output, axis=1)

    print(accuracy_score(true_output, predicted_output))
    print(classification_report(true_output, predicted_output))
    cm = confusion_matrix(true_output, predicted_output)

    df_cm = pd.DataFrame(cm, range(OUTPUT_NODES_NUMBER), range(OUTPUT_NODES_NUMBER))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="YlGnBu")  # font size

    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()


def get_parameters():
    """
    Gets the parameters of the file and simulated annealing from the user

    """
    print("================")
    print("File parameters\n")
    file_type = ""
    start_temp = end_temp = eq_number = cool_number = dimensions = window = stride = 0
    while file_type.upper() != "LIGHTING" and file_type.upper() != "FACE":
        file_type = input("File (LIGHTING/FACE): ").upper()
    while dimensions < 3 or dimensions > 9:
        dimensions = int(input("Dimensions (d: 3 <= d <= 9 and d is an integer): "))
    while window <= 0:
        window = float(input("Window length (w: 0 < w <= 1): "))
    while stride <= 0:
        stride = int(input("Stride (s: s >= 1 and s is an integer): "))

    print("================")
    print("Simulated annealing parameters\n")
    while start_temp <= 0:
        start_temp = float(input("Start temperature (ts: ts > 0): "))
    while 0 >= end_temp or end_temp > start_temp:
        end_temp = float(input("End temperature (te: te > 0 and te < ts): "))
    while eq_number <= 0:
        eq_number = int(input("Equilibrium number (e: e >= 1 and e is an integer): "))
    while cool_number <= 0 or cool_number > 1:
        cool_number = float(input("Cooling parameter (c: 0 < c < 1): "))

    return file_type, start_temp, end_temp, eq_number, cool_number, dimensions, window, stride


if __name__ == '__main__':
    params = get_parameters()
    results = train(*params)
    test(params[0], *results[:3], *params[5:])
