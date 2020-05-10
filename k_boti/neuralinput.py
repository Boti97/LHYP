import os
import time

from geometry import get_center, get_closest_point_to_line, get_array_len, rotate_point_around_point, get_line_length
from patient import Patient, read_patient_pickle

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


class NeuralInput:
    def __init__(self, patient):
        self.wall_thicknesses, self.ln_polygons, self.lp_polygons, self.distances = get_wall_thicknesses(
            patient.dia_ln_contours,
            patient.dia_lp_contours,
            patient.dia_rn_contours)
        self.diagnosis = patient.diagnosis
        self.study_id = patient.study_id.strip()


def rotate_right_center_and_get_wall_thicknesses(left_center_points, right_center_points, dia_ln_contours,
                                                 dia_lp_contours, angle):
    right_center_points_for_region = []
    if angle > 0:
        for i in range(len(right_center_points)):
            right_center_points_for_region.append(
                rotate_point_around_point(right_center_points[i], left_center_points[i], angle))
    else:
        right_center_points_for_region = right_center_points

    wall_thicknesses_by_region, closest_points_ln, closest_points_lp = get_wall_thicknesses_by_region(
        left_center_points, right_center_points_for_region, dia_ln_contours,
        dia_lp_contours)

    return wall_thicknesses_by_region, closest_points_ln, closest_points_lp


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

    return wall_thicknesses_by_region, closest_points_ln, closest_points_lp


def get_polygons_from_closest_point(septal_closest_points, anterior_closest_points, lateral_closest_points,
                                    inferior_closest_points):
    polygons = []
    for i in range(len(septal_closest_points)):
        polygons.append([septal_closest_points[i], anterior_closest_points[i], lateral_closest_points[i],
                         inferior_closest_points[i]])
    return polygons


def get_distances_between_opposite_points(closest_points_a, closest_points_b):
    distances = []
    for i in range(len(closest_points_a)):
        distances.append(get_line_length(closest_points_a[i], closest_points_b[i]))

    return distances


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

    # apical septal region
    septal_wall_thicknesses, septal_closest_points_ln, septal_closest_points_lp = rotate_right_center_and_get_wall_thicknesses(
        left_center_points, right_center_points_for_septal,
        dia_ln_contours,
        dia_lp_contours, 0)

    wall_thicknesses.append(septal_wall_thicknesses)

    # apical anterior region
    anterior_wall_thicknesses, anterior_closest_points_ln, anterior_closest_points_lp = rotate_right_center_and_get_wall_thicknesses(
        left_center_points, right_center_points_for_septal,
        dia_ln_contours,
        dia_lp_contours, 90)

    wall_thicknesses.append(anterior_wall_thicknesses)

    # apical lateral region
    lateral_wall_thicknesses, lateral_closest_points_ln, lateral_closest_points_lp = rotate_right_center_and_get_wall_thicknesses(
        left_center_points, right_center_points_for_septal,
        dia_ln_contours,
        dia_lp_contours, 180)

    wall_thicknesses.append(lateral_wall_thicknesses)

    # apical inferior region
    inferior_wall_thicknesses, inferior_closest_points_ln, inferior_closest_points_lp = rotate_right_center_and_get_wall_thicknesses(
        left_center_points, right_center_points_for_septal,
        dia_ln_contours,
        dia_lp_contours, 270)

    wall_thicknesses.append(inferior_wall_thicknesses)

    # create polygons from points found in the different regions
    polygons_for_slices_ln = get_polygons_from_closest_point(septal_closest_points_ln, anterior_closest_points_ln,
                                                             lateral_closest_points_ln, inferior_closest_points_ln)

    polygons_for_slices_lp = get_polygons_from_closest_point(septal_closest_points_lp, anterior_closest_points_lp,
                                                             lateral_closest_points_lp, inferior_closest_points_lp)

    distances_between_opposite_points = [
        get_distances_between_opposite_points(anterior_closest_points_ln, inferior_closest_points_ln),
        get_distances_between_opposite_points(septal_closest_points_ln, lateral_closest_points_ln)]

    return wall_thicknesses, polygons_for_slices_ln, polygons_for_slices_lp, distances_between_opposite_points


def save_neural_input_to_pickle(neural_input, output_path, study_id):
    filename = "neural_input_" + study_id + ".pkl"
    full_out_path = output_path + "/" + filename

    with open(full_out_path, 'wb') as output:
        pickle.dump(neural_input, output)


def read_neural_inputs_from_pickle(pickles_path):
    pickle_list = os.listdir(pickles_path)
    neural_inputs = []
    for pickle_file in pickle_list:
        full_out_path = pickles_path + "/" + pickle_file
        with open(full_out_path, 'rb') as input_file:
            neural_inputs.append(pickle.load(input_file))

    return neural_inputs


def process_patient_files(output_path, patient_pickles_path):
    patients = read_patient_pickle(patient_pickles_path)

    for patient in patients:
        if patient.study_id is not None:
            neural_input = NeuralInput(patient)
            if len(neural_input.wall_thicknesses) != 0 and len(neural_input.ln_polygons) != 0 and len(
                    neural_input.lp_polygons) != 0 and len(neural_input.distances) != 0:
                save_neural_input_to_pickle(neural_input, output_path, neural_input.study_id)
                print("Patient processed:" + neural_input.study_id + '\n')
            else:
                print("Error during patient processing! One or more essential data is not filled.\n")
        else:
            print("Patient cannot be processed: study_id UNKNOWN\n")


def main():
    output_path = 'C:/MyLife/School/MSc/8.felev/Onlab/k_boti/neural_input_pickles'
    patient_pickles_path = 'C:/MyLife/School/MSc/8.felev/Onlab/k_boti/pickles'
    process_patient_files(output_path, patient_pickles_path)


if __name__ == '__main__':
    main()
