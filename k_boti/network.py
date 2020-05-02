import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import datasets
from patient import Patient
import torch
import torch.nn as nn
import torch.optim as optim
import time
from neuralinput import process_patient_files


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


def get_minimum_wall_thickness_number_and_all_wall_thicknesses(patients):
    all_wall_thicknesses = []
    min_wall_thickness_list_length = 1000
    for patient in patients:
        if len(patient.wall_thicknesses[0]) < min_wall_thickness_list_length:
            min_wall_thickness_list_length = len(patient.wall_thicknesses[0])
        all_wall_thicknesses.append(patient.wall_thicknesses)
    return min_wall_thickness_list_length, all_wall_thicknesses


def get_all_diagnoses(patients):
    all_diagnoses = []
    for patient in patients:
        all_diagnoses.append(patient.diagnosis)
    return all_diagnoses


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


def main():
    patients = process_patient_files()

    min_wall_thickness_list_length, all_wall_thicknesses = get_minimum_wall_thickness_number_and_all_wall_thicknesses(
        patients)

    all_data_x = torch.Tensor(get_trimmed_wall_thicknesses(all_wall_thicknesses, min_wall_thickness_list_length))
    all_data_y = torch.Tensor(diagnoses_converter(get_all_diagnoses(patients))).type(torch.int64)
    train_x = all_data_x[:int(len(all_data_x) * 0.6)]
    test_x = all_data_x[int(len(all_data_x) * 0.6):]
    train_y = all_data_y[:int(len(all_data_x) * 0.6)]
    test_y = all_data_y[int(len(all_data_x) * 0.6):]

    # flattening
    train_x = train_x.view(-1, 4 * 9).squeeze(1)
    test_x = test_x.view(-1, 4 * 9).squeeze(1)

    all_idx = np.arange(len(train_x))
    train_idx = all_idx[:15]
    dev_idx = all_idx[15:]
    dev_x = train_x[dev_idx]
    dev_y = train_y[dev_idx]
    train_x = train_x[train_idx]
    train_y = train_y[train_idx]
    print("Train size:", train_x.size(), train_y.size())
    print("Dev size:", dev_x.size(), dev_y.size())
    print("Test size:", test_x.size(), train_y.size())

    model = SimpleClassifier(
        input_dim=train_x.size(1),
        output_dim=5,
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
    test_acc


main()
