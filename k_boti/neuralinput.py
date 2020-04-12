import os

from geometry import get_center, get_closest_point_to_line, get_array_len, rotate_point_around_point
from patient import Patient

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


class NeuralInput:
    def __init__(self, patient):
        self.wall_thicknesses = get_wall_thicknesses(patient.dia_ln_contours, patient.dia_lp_contours,
                                                     patient.dia_rn_contours)
        self.diagnosis = patient.diagnosis


def rotate_right_center_and_get_wall_thicknesses(left_center_points, right_center_points, dia_ln_contours, dia_lp_contours, angle):
    right_center_points_for_region = []
    for i in range(len(right_center_points)):
        right_center_points_for_region.append(
            rotate_point_around_point(right_center_points[i], left_center_points[i], angle))

    return get_wall_thicknesses_by_region(left_center_points, right_center_points_for_region,dia_ln_contours, dia_lp_contours)


def get_wall_thicknesses_by_region(left_center_points, right_center_points, ln_contours, lp_contours):
    wall_thicknesses_by_region = []
    closest_points_ln = []
    closest_points_lp = []

    for i in range(len(left_center_points)):
        closest_points_ln.append(
            get_closest_point_to_line(left_center_points[i], right_center_points[i], ln_contours[i]))
    for i in range(len(left_center_points)):
        closest_points_lp.append(
            get_closest_point_to_line(left_center_points[i], right_center_points[i], lp_contours[i]))

    for i in range(len(left_center_points)):
        wall_thicknesses_by_region.append(get_array_len(closest_points_ln[i] - closest_points_lp[i]))

    return wall_thicknesses_by_region


def get_wall_thicknesses(dia_ln_contours, dia_lp_contours, dia_rn_contours):
    wall_thicknesses = []
    right_center_points_for_septal = []
    left_center_points = []
    for right_ventricle_contour in dia_rn_contours:
        center_x, center_y = get_center(right_ventricle_contour)
        right_center_points_for_septal.append([center_x, center_y])
    for left_ventricle_contour in dia_ln_contours:
        center_x, center_y = get_center(left_ventricle_contour)
        left_center_points.append([center_x, center_y])

    wall_thicknesses.append(
        get_wall_thicknesses_by_region(left_center_points, right_center_points_for_septal,
                                       dia_ln_contours, dia_lp_contours))

    # anterior region
    wall_thicknesses.append(
        rotate_right_center_and_get_wall_thicknesses(left_center_points, right_center_points_for_septal,
                                                     dia_ln_contours,
                                                     dia_lp_contours, 90))

    # lateral region
    wall_thicknesses.append(
        rotate_right_center_and_get_wall_thicknesses(left_center_points, right_center_points_for_septal,
                                                     dia_ln_contours,
                                                     dia_lp_contours, 180))

    # inferior region
    wall_thicknesses.append(
        rotate_right_center_and_get_wall_thicknesses(left_center_points, right_center_points_for_septal,
                                                     dia_ln_contours,
                                                     dia_lp_contours, 270))

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


def process_patient_files():
    patients = read_patient_pickle()
    neural_inputs = []
    for patient in patients:
        neural_inputs .append(NeuralInput(patient))
    return neural_inputs


def main():
    patients = read_patient_pickle()
    neural_inputs = []
    for patient in patients:
        neural_inputs = NeuralInput(patient)
    return neural_inputs


if __name__ == '__main__':
    main()
