from src.config import FACE_FILE, LIGHTING_FILE
import pandas as pd
import numpy as np


def read_csv(file_type, dataset, dimensions):
    """
    Reads the csv, upsamples the lacking classes and returns the file divided into an input and output matrix

    """
    global OUTPUT_NODES_NUMBER
    csvs = []
    input = []
    output = []

    for i in range(1, dimensions + 1):
        if dataset == 'LIGHTING':
            csvs.append(pd.read_csv(
                LIGHTING_FILE + 'Lighting2_' + str(dimensions) + '_Dimensions/Lighting2Dimension' + str(i) + '_' + str(
                    file_type).upper() + '.csv', sep=';', decimal=','))
            OUTPUT_NODES_NUMBER = 2
        elif dataset == 'FACE':
            csvs.append(pd.read_csv(
                FACE_FILE + 'FaceFour_' + str(dimensions) + '_Dimensions/FaceFourDimension' + str(i) + '_' + str(
                    file_type).upper() + '.csv', sep=';', decimal=','))
            OUTPUT_NODES_NUMBER = 4

    max_count = max(csvs[0]['V1'].value_counts().tolist())
    classes = csvs[0]['V1'].value_counts().keys().tolist()

    if file_type == 'TRAIN':
        for i in range(dimensions):
            for class_id in classes:
                class_list = csvs[i].loc[csvs[i]['V1'] == class_id].reset_index(drop=True)
                padding = max_count - len(class_list)
                for j in range(padding):
                    csvs[i] = csvs[i].append(class_list.iloc[j % len(class_list)])
            csvs[i] = csvs[i].reset_index(drop=True)

        new_idx = np.random.permutation(csvs[0].index)
        for i in range(dimensions):
            csvs[i] = csvs[i].reindex(new_idx).reset_index(drop=True)

    first_dim = True
    # input[dimension number][series number]
    for i, df in enumerate(csvs):
        input.append([])
        for j, row in df.iterrows():
            input[i].append([])
            for element in row[1:].tolist():
                input[i][j].append(element)
            if first_dim:
                output.append(row[0])
        first_dim = False


    # input_matrix[dimension number][series number][element in the series]
    input_matrix = []
    output_matrix = []

    series_number = len(input[0])
    elements_number = len(input[0][0])

    for series_idx in range(series_number):
        # output
        output_value = output[series_idx]

        if dataset == 'LIGHTING':
            if output_value > 0:
                output_matrix.append([1, 0])
            else:
                output_matrix.append([0, 1])
        elif dataset == 'OLIVE' or dataset == 'ECG' or dataset == 'FACE':
            result = np.zeros(OUTPUT_NODES_NUMBER)
            result[int(output_value) - 1] = 1
            output_matrix.append(result)

        # input
        input_matrix.append([])
        for element_idx in range(elements_number):
            input_matrix[series_idx].append([])
            for dim_idx in range(dimensions):
                input_matrix[series_idx][element_idx].append(input[dim_idx][series_idx][element_idx])

    return input_matrix, output_matrix


def flatten_input(input_matrix, _window, stride):
    """
    Flattens a window of inputs into a 1-D array

    """
    flattened_input = []
    window = int(_window * len(input_matrix[0])) - 1
    for i, series in enumerate(input_matrix):
        flattened_input.append([])
        for j in range(0, len(input_matrix[0]) - window, stride):
            input = list(np.concatenate(series[j:j + window + 1]))
            flattened_input[i].append(input)
    return flattened_input, window
