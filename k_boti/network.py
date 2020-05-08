import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from patient import Patient
from neuralinput import NeuralInput
from neuralinput import process_patient_files, read_neural_inputs_from_pickle


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


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        h = self.input_layer(X)
        h = self.relu(h)
        out = self.output_layer(h)
        return out


def get_wall_thicknesses(patients):
    all_wall_thicknesses = []
    min_wall_thickness_list_length = 1000
    for patient in patients:
        if len(patient.wall_thicknesses[0]) < min_wall_thickness_list_length:
            min_wall_thickness_list_length = len(patient.wall_thicknesses[0])
        all_wall_thicknesses.append(patient.wall_thicknesses)

    # trimming all the lists to the same length
    all_wall_thicknesses = torch.Tensor(
        get_trimmed_wall_thicknesses(all_wall_thicknesses, min_wall_thickness_list_length))

    # flattening
    all_wall_thicknesses = all_wall_thicknesses.view(-1, 4 * min_wall_thickness_list_length)

    return all_wall_thicknesses


def get_all_diagnoses(patients):
    all_diagnoses = []
    for patient in patients:
        all_diagnoses.append(patient.diagnosis)
    return all_diagnoses


def get_distances(patients):
    all_distances = []
    min_length_of_list = 1000
    for patient in patients:
        for d in patient.distances:
            if len(d) < min_length_of_list:
                min_length_of_list = len(d)
        all_distances.append(patient.distances)

    # trimming all the lists to the same length
    all_distances = torch.Tensor(get_trimmed_distances(all_distances, min_length_of_list))

    # flattening
    all_distances = all_distances.view(-1, 2 * min_length_of_list).squeeze(1)

    return all_distances


def get_polygons(patients):
    all_polygons_ln = []
    all_polygons_lp = []
    min_length_of_list_ln = 1000
    min_length_of_list_lp = 1000
    for patient in patients:
        if len(patient.ln_polygons) < min_length_of_list_ln:
            min_length_of_list_ln = len(patient.ln_polygons)
        if len(patient.lp_polygons) < min_length_of_list_lp:
            min_length_of_list_lp = len(patient.lp_polygons)
        all_polygons_ln.append(patient.ln_polygons)
        all_polygons_lp.append(patient.lp_polygons)

    if min_length_of_list_ln > min_length_of_list_lp:
        min_length_of_list_ln = min_length_of_list_lp
    else:
        min_length_of_list_lp = min_length_of_list_ln

    # trimming all the lists to the same length
    all_polygons_ln_tensor = torch.Tensor(get_trimmed_polygons(all_polygons_ln, min_length_of_list_ln))
    all_polygons_lp_tensor = torch.Tensor(get_trimmed_polygons(all_polygons_lp, min_length_of_list_lp))

    # flattening
    trimmed_polygons_ln = all_polygons_ln_tensor.view(-1, min_length_of_list_ln * 4 * 2).squeeze(1)
    trimmed_polygons_lp = all_polygons_lp_tensor.view(-1, min_length_of_list_lp * 4 * 2).squeeze(1)

    all_trimmed_polygons = torch.cat((trimmed_polygons_ln, trimmed_polygons_lp), 1)

    return all_trimmed_polygons


def diagnoses_converter(all_diagnoses):
    category_dir = {
        'HCM': 0,
        'Normal': 1,
        'U18_m': 2,
        'U18_f': 3,
        'Aortastenosis': 4,
        'AMY': 5,
        'Other': 6
    }
    diagnoses_converted = []
    for diagnoses in all_diagnoses:
        if diagnoses in category_dir:
            diagnoses_converted.append(category_dir[diagnoses])
        else:
            diagnoses_converted.append(6)
    return diagnoses_converted


def get_trimmed_wall_thicknesses(all_wall_thicknesses, min_wall_thickness_number):
    trimmed_wall_thicknesses = []
    for wall_thicknesses in all_wall_thicknesses:
        patient_wall_thicknesses = []
        for w in wall_thicknesses:
            if len(w) > min_wall_thickness_number:
                patient_wall_thicknesses.append(w[:min_wall_thickness_number])
            else:
                patient_wall_thicknesses.append(w)

        trimmed_wall_thicknesses.append(patient_wall_thicknesses)
    return trimmed_wall_thicknesses


def get_trimmed_distances(all_distances, min_distance_list_length):
    trimmed_distances = []
    for distances in all_distances:
        patient_distances = []
        for w in distances:
            if len(w) > min_distance_list_length:
                patient_distances.append(w[:min_distance_list_length])
            else:
                patient_distances.append(w)

        trimmed_distances.append(patient_distances)
    return trimmed_distances


def get_trimmed_polygons(all_polygons, min_length_of_list):
    trimmed_polygons = []
    for polygon in all_polygons:
        if len(polygon) > min_length_of_list:
            trimmed_polygons.append(polygon[:min_length_of_list])
        else:
            trimmed_polygons.append(polygon)
    return trimmed_polygons


def train_neural_network(neural_inputs_pickles_path):
    neural_inputs = read_neural_inputs_from_pickle(neural_inputs_pickles_path)

    wall_thicknesses = get_wall_thicknesses(neural_inputs)
    distances = get_distances(neural_inputs)
    polygons = get_polygons(neural_inputs)

    all_data_y = torch.Tensor(diagnoses_converter(get_all_diagnoses(neural_inputs))).type(torch.int64)

    # concatenate all the available data to one tensor as the input
    # concat_all_data_x = torch.cat((distances, wall_thicknesses, polygons), 1)
    concat_all_data_x = wall_thicknesses

    train_x = concat_all_data_x[:int(len(concat_all_data_x) * 0.6)].type(torch.float32)
    test_x = concat_all_data_x[int(len(concat_all_data_x) * 0.6):].type(torch.float32)
    train_y = all_data_y[:int(len(all_data_y) * 0.6)].type(torch.int64)
    test_y = all_data_y[int(len(all_data_y) * 0.6):].type(torch.int64)

    all_idx = np.arange(len(train_x))
    train_idx = all_idx[:round(len(all_idx) * 0.7)]
    dev_idx = all_idx[round(len(all_idx) * 0.7):]
    dev_x = train_x[dev_idx]
    dev_y = train_y[dev_idx]
    train_x = train_x[train_idx]
    train_y = train_y[train_idx]
    print("Train size:", train_x.size(), train_y.size())
    print("Dev size:", dev_x.size(), dev_y.size())
    print("Test size:", test_x.size(), train_y.size())

    model = SimpleClassifier(
        input_dim=train_x.size(1),
        output_dim=7,  # number of diagnoses
        hidden_dim=50
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    batch_size = 10
    train_iter = BatchedIterator(train_x, train_y, batch_size)
    dev_iter = BatchedIterator(dev_x, dev_y, batch_size)
    test_iter = BatchedIterator(test_x, test_y, batch_size)

    all_train_loss = []
    all_dev_loss = []
    all_train_acc = []
    all_dev_acc = []

    for epoch in range(10):
        # training loop
        for bi, (batch_x, batch_y) in enumerate(train_iter.iterate_once()):
            y_out = model(batch_x)
            loss = criterion(y_out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # one train epoch finished, evaluate on the train and the dev set (NOT the test)
        train_out = model(train_x)
        train_loss = criterion(train_out, train_y)
        all_train_loss.append(train_loss.item())
        train_pred = train_out.max(axis=1)[1]
        train_acc = torch.eq(train_pred, train_y).sum().float() / len(train_x)
        all_train_acc.append(train_acc)

        dev_out = model(dev_x)
        dev_loss = criterion(dev_out, dev_y)
        all_dev_loss.append(dev_loss.item())
        dev_pred = dev_out.max(axis=1)[1]
        dev_acc = torch.eq(dev_pred, dev_y).sum().float() / len(dev_x)
        all_dev_acc.append(dev_acc)

        print(f"Epoch: {epoch}\n  train accuracy: {train_acc}  train loss: {train_loss}")
        print(f"  dev accuracy: {dev_acc}  dev loss: {dev_loss}")

    test_pred = model(test_x).max(axis=1)[1]
    test_acc = torch.eq(test_pred, test_y).sum().float() / len(test_x)
    print(test_acc)

    plt.plot(all_train_loss, label='train')
    plt.plot(all_dev_loss, label='dev')
    plt.legend()

    plt.plot(all_train_acc, label='train')
    plt.plot(all_dev_acc, label='dev')
    plt.legend()
    plt.show()


def main():
    neural_inputs_pickles_path = 'C:/MyLife/School/MSc/8.felev/Onlab/k_boti/neural_input_pickles'
    train_neural_network(neural_inputs_pickles_path)


if __name__ == '__main__':
    main()

