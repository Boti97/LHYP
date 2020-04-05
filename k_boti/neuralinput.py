import os

from geometry import get_center, get_closest_point_to_line, get_array_len
from patient import Patient

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


class NeuralInput:
    def __init__(self, patient):
        self.wall_thicknesses = get_wall_thicknesses(patient.dia_ln_contours, patient.dia_lp_contours,
                                                     patient.dia_rn_contours)


def get_wall_thicknesses(dia_ln_contours, dia_lp_contours, dia_rn_contours):
    wall_thicknesses = []
    right_ventricle_centers = []
    left_ventricle_centers = []
    for right_ventricle_contour in dia_rn_contours:
        right_ventricle_centers.append(get_center(right_ventricle_contour))
    for left_ventricle_contour in dia_lp_contours:
        center_x, center_y = get_center(left_ventricle_contour)
        left_ventricle_centers.append([center_x, center_y])
    closest_points_ln = []
    closest_points_lp = []

    for i in range(len(left_ventricle_centers)):
        closest_points_ln.append(
            get_closest_point_to_line(left_ventricle_centers[i], right_ventricle_centers[i], dia_ln_contours[i]))
    for i in range(len(left_ventricle_centers)):
        closest_points_lp.append(
            get_closest_point_to_line(left_ventricle_centers[i], right_ventricle_centers[i], dia_lp_contours[i]))

    for i in range(len(left_ventricle_centers)):
        wall_thicknesses.append(get_array_len(closest_points_ln[i] - closest_points_lp[i]))

    return wall_thicknesses


def read_patient_pickle():
    """
    print("Pickles' directory path:")
    pickles_images = input()
    """

    pickles_images = 'C:/MyLife/School/MSc/8.felev/Onlab/k_boti/pickles'

    pickle_list = os.listdir(pickles_images)
    patients = []
    for pickle_file in pickle_list:
        full_out_path = pickles_images + "/" + pickle_file
        with open(full_out_path, 'rb') as input_file:
            patients.append(pickle.load(input_file))

    return patients


def process_patient_files(patients):
    neural_inputs = []
    for patient in patients:
        neural_inputs = NeuralInput(patient)
    return neural_inputs


def main():
    patients = read_patient_pickle()
    neural_inputs = process_patient_files(patients)


if __name__ == '__main__':
    main()
