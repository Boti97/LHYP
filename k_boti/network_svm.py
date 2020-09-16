from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as torch_fun
from neuralinput import process_patient_files, read_neural_inputs_from_pickle

from sklearn import svm
from neuralinput import NeuralInput


class BatchedIterator:
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.batch_starts = list(range(0, len(self.X), batch_size))

    def iterate_once(self):
        for start in range(0, len(self.X), self.batch_size):
            end = start + self.batch_size
            yield self.X[start:end], self.y[start:end]


def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss


def get_all_diagnoses(patients):
    all_diagnoses = []
    for patient in patients:
        all_diagnoses.append(patient.diagnosis)
    return all_diagnoses


def get_wall_thicknesses(patients):
    all_wall_thicknesses = []
    max_wall_thickness_list_length = -1
    for patient in patients:
        if len(patient.wall_thicknesses[0]) == 0:
            print("Need to be deleted: " + patient.patiend_id)
        else:
            if len(patient.wall_thicknesses[0]) > max_wall_thickness_list_length:
                max_wall_thickness_list_length = len(patient.wall_thicknesses[0])
            all_wall_thicknesses.append(patient.wall_thicknesses)

    # completing all the lists to the same length
    all_wall_thicknesses = torch.Tensor(
        get_completed_wall_thicknesses(all_wall_thicknesses, max_wall_thickness_list_length))

    # flattening
    all_wall_thicknesses = all_wall_thicknesses.view(-1, 4 * max_wall_thickness_list_length)

    return all_wall_thicknesses


def get_distances(patients):
    all_distances = []
    max_length_of_list = -1
    for patient in patients:
        for d in patient.distances:
            if len(d) > max_length_of_list:
                max_length_of_list = len(d)
        all_distances.append(patient.distances)

    # completing all the lists to the same length
    all_distances = torch.Tensor(get_completed_distances(all_distances, max_length_of_list))

    # flattening
    all_distances = all_distances.view(-1, 2 * max_length_of_list).squeeze(1)

    return all_distances


def get_polygons(patients):
    all_polygons_ln = []
    all_polygons_lp = []
    max_length_of_list_ln = -1
    max_length_of_list_lp = -1
    for patient in patients:
        if len(patient.ln_polygons) > max_length_of_list_ln:
            max_length_of_list_ln = len(patient.ln_polygons)
        if len(patient.lp_polygons) > max_length_of_list_lp:
            max_length_of_list_lp = len(patient.lp_polygons)
        all_polygons_ln.append(patient.ln_polygons)
        all_polygons_lp.append(patient.lp_polygons)

    # the two length should be the same for concatenating
    if max_length_of_list_ln < max_length_of_list_lp:
        max_length_of_list_ln = max_length_of_list_lp
    else:
        max_length_of_list_lp = max_length_of_list_ln

    # completed all the lists to the same length
    all_polygons_ln_tensor = torch.Tensor(get_completed_polygons(all_polygons_ln, max_length_of_list_ln))
    all_polygons_lp_tensor = torch.Tensor(get_completed_polygons(all_polygons_lp, max_length_of_list_lp))

    # flattening
    completed_polygons_ln = all_polygons_ln_tensor.view(-1, max_length_of_list_ln * 4 * 2).squeeze(1)
    completed_polygons_lp = all_polygons_lp_tensor.view(-1, max_length_of_list_lp * 4 * 2).squeeze(1)

    all_competed_polygons = torch.cat((completed_polygons_ln, completed_polygons_lp), 1)

    return all_competed_polygons


def diagnoses_converter(all_diagnoses):
    category_dir = {
        'HCM': 0,
        'Normal': 1,
        'Other': 2
    }
    diagnoses_converted = []
    for diagnoses in all_diagnoses:
        if diagnoses in category_dir:
            diagnoses_converted.append(category_dir[diagnoses])
        else:
            diagnoses_converted.append(2)
    occurence_count = Counter(diagnoses_converted)
    return diagnoses_converted


def get_completed_wall_thicknesses(all_wall_thicknesses, max_wall_thickness_length):
    completed_wall_thicknesses = []
    for wall_thicknesses in all_wall_thicknesses:
        patient_wall_thicknesses = []
        for wall_thickness in wall_thicknesses:
            if len(wall_thickness) < max_wall_thickness_length:
                wall_thickness += [0] * (max_wall_thickness_length - len(wall_thickness))
                patient_wall_thicknesses.append(wall_thickness)
            elif len(wall_thickness) == max_wall_thickness_length:
                patient_wall_thicknesses.append(wall_thickness)
            else:
                print("There's a longer list. Finding the max wall thickness length was not successful.")

        completed_wall_thicknesses.append(patient_wall_thicknesses)
    return completed_wall_thicknesses


def get_completed_distances(all_distances, max_distance_list_length):
    completed_distances = []
    for distances in all_distances:
        patient_distances = []
        for distance in distances:
            if len(distance) < max_distance_list_length:
                distance += [0] * (max_distance_list_length - len(distance))
                patient_distances.append(distance)
            elif len(distance) == max_distance_list_length:
                patient_distances.append(distance)
            else:
                print("There's a longer list. Finding the max distance length was not successful.")

        completed_distances.append(patient_distances)
    return completed_distances


def get_completed_polygons(all_polygons, max_length_of_list):
    completed_polygons = []
    for polygon in all_polygons:
        if len(polygon) < max_length_of_list:
            for i in range((max_length_of_list - len(polygon))):
                polygon.append([np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])])
            # polygon += [[0,0],[0,0],[0,0],[0,0]] * (max_length_of_list - len(polygon))
            completed_polygons.append(polygon)
        elif len(polygon) == max_length_of_list:
            completed_polygons.append(polygon)
        else:
            print("There's a longer list. Finding the max polygon list length was not successful.")
    return completed_polygons


def train_neural_network(neural_inputs_pickles_path):
    neural_inputs = read_neural_inputs_from_pickle(neural_inputs_pickles_path)

    wall_thicknesses = get_wall_thicknesses(neural_inputs)
    distances = get_distances(neural_inputs)
    polygons = get_polygons(neural_inputs)

    all_data_y = torch.Tensor(diagnoses_converter(get_all_diagnoses(neural_inputs))).type(torch.int64)

    # concatenate all the available data to one tensor as the input
    concat_all_data_x = torch.cat((distances, wall_thicknesses, polygons), 1)

    # concat_all_data_x = torch.cat((distances, polygons), 1)
    # concat_all_data_x = torch.cat((distances, wall_thicknesses), 1)
    # concat_all_data_x = torch.cat((polygons, wall_thicknesses), 1)

    # concat_all_data_x = distances
    # concat_all_data_x = polygons
    # concat_all_data_x = wall_thicknesses

    # normalize input
    concat_all_data_x = torch_fun.normalize(concat_all_data_x)

    # set train, test and dev
    train_dev_x = concat_all_data_x[:int(len(concat_all_data_x) * 0.8)].type(torch.float32)
    train_dev_y = all_data_y[:int(len(all_data_y) * 0.8)].type(torch.int64)

    test_x = concat_all_data_x[int(len(concat_all_data_x) * 0.8):].type(torch.float32)
    test_y = all_data_y[int(len(all_data_y) * 0.8):].type(torch.int64)

    all__train_idx = np.arange(len(train_dev_x))
    train_idx = all__train_idx[:round(len(all__train_idx) * 0.8)]
    dev_idx = all__train_idx[round(len(all__train_idx) * 0.8):]
    dev_x = train_dev_x[dev_idx]
    dev_y = train_dev_y[dev_idx]
    train_x = train_dev_x[train_idx]
    train_y = train_dev_y[train_idx]

    print("Train size:", train_x.size(), train_y.size())
    print("Dev size:", dev_x.size(), dev_y.size())
    print("Test size:", test_x.size(), test_y.size())

    model = svm.LinearSVR(C=1.0)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.loss)

    batch_size = 20
    train_iter = BatchedIterator(train_x, train_y, batch_size)

    all_train_loss = []
    all_dev_loss = []
    all_train_acc = []
    all_dev_acc = []

    for epoch in range(200):
        # training loop
        for bi, (batch_x, batch_y) in enumerate(train_iter.iterate_once()):
            y_out = model.fit(batch_x, batch_y)
            # optimizer.zero_grad()
            # optimizer.step()

        # one train epoch finished, evaluate on the train and the dev set (NOT the test)
        train_out = torch.Tensor(model.predict(train_x)).type(torch.float32)
        train_loss = my_loss(train_out, train_y.type(torch.float32))
        all_train_loss.append(train_loss.item())
        train_pred = train_out
        train_acc = torch.eq(train_pred, train_y.type(torch.float32)).sum().float() / len(train_x)
        all_train_acc.append(train_acc)

        dev_out = torch.Tensor(model.predict(dev_x)).type(torch.float32)
        dev_loss = my_loss(dev_out, dev_y.type(torch.float32))
        all_dev_loss.append(dev_loss.item())
        dev_pred = dev_out
        dev_acc = torch.eq(dev_pred, dev_y.type(torch.float32)).sum().float() / len(dev_x)
        all_dev_acc.append(dev_acc)

        print(f"Epoch: {epoch}\n  train accuracy: {train_acc}  train loss: {train_loss}")
        print(f"  dev accuracy: {dev_acc}  dev loss: {dev_loss}")

    test_pred = torch.Tensor(model.predict(test_x)).type(torch.float32)
    test_acc = torch.eq(test_pred, test_y.type(torch.float32)).sum().float() / len(test_x)
    print(test_acc)

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(all_train_loss, label='train loss')
    ax1.plot(all_dev_loss, label='dev loss')
    ax1.legend()

    ax2.plot(all_train_acc, label='train acc')
    ax2.plot(all_dev_acc, label='dev acc')
    ax2.legend()
    plt.show()


def main():
    neural_inputs_pickles_path = 'C:/MyLife/School/MSc/8.felev/Onlab/k_boti/neural_input_pickles'
    train_neural_network(neural_inputs_pickles_path)


if __name__ == '__main__':
    main()
