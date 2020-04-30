from geometry import get_contour_area
from con_reader import CONreaderVM
import time
import numpy as np
import os

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

start_time = 0


class ContourFileError(Exception):
    def __init__(self, message):
        super().__init__(message)


class Patient:
    def __init__(self, contour_reader):
        print(time.time() - start_time)
        contours = contour_reader.get_hierarchical_contours()

        # separate diastole and systole by the first available frame
        # if we dont find a lp, we look for and ln
        slc_fr_list, slc_fr_cont = get_frames_and_contours(contours)

        dia_frame, sys_frame = get_diastole_and_systole_frame(slc_fr_list, slc_fr_cont)

        self.dia_ln_contours, self.dia_lp_contours, self.dia_rn_contours = \
            get_all_necessary_contours_by_frame(contours, dia_frame)

        self.study_id = contour_reader.volume_data["Study_id="]
        self.patient_weight = contour_reader.volume_data["Patient_weight="]
        self.Patient_height = contour_reader.volume_data["Patient_height"]
        self.Patient_gender = contour_reader.volume_data["Patient_gender="]


def get_all_necessary_contours_by_frame(contours, frame):
    dia_ln_contours = []
    dia_lp_contours = []
    dia_rn_contours = []
    for slc in contours:
        if "ln" in contours[slc][frame] and "lp" in contours[slc][frame] and "rn" in contours[slc][frame]:
            dia_ln_contours.append(contours[slc][frame]["ln"])
            dia_lp_contours.append(contours[slc][frame]["lp"])
            dia_rn_contours.append(contours[slc][frame]["rn"])

    return dia_ln_contours, dia_lp_contours, dia_rn_contours


# not used
def simplify_contours(contours):
    cont_list = []
    for slc in contours:
        for frm in contours[slc]:
            for mode in contours[slc][frm]:
                cont_list.append((slc, frm, mode))
    return cont_list


def simplify_slc_frm_list(slc_frm_list, slc):
    simple_slc_frm = []
    for frm in slc_frm_list:
        simple_slc_frm.append((slc, frm))
    return simple_slc_frm


def get_frames_and_contours(contours):
    max_frames_in_slc = 1
    cont_list = []
    frames = []
    for slc in contours:
        if len(contours[slc]) > max_frames_in_slc:
            has_ln = True
            has_lp = True
            for con_frame in contours[slc]:
                if has_ln and "ln" not in contours[slc][con_frame]:
                    has_ln = False
                if has_lp and "lp" not in contours[slc][con_frame]:
                    has_lp = False
            if has_ln:
                cont_list.clear()
                max_frames_in_slc = len(contours[slc])
                frames = simplify_slc_frm_list(contours[slc], slc)
                for sl, frm in frames:
                    cont_list.append(contours[slc][frm]["ln"])
            elif has_lp:
                cont_list.clear()
                max_frames_in_slc = len(contours[slc])
                frames = simplify_slc_frm_list(contours[slc], slc)
                for sl, frm in frames:
                    cont_list.append(contours[slc][frm]["lp"])
    return frames, cont_list


# not used
def get_contours_by_positions_and_mode(contours, positions, mode):
    spec_contours = []
    for pos in positions:
        if mode in contours[pos[0]][pos[1]]:
            spec_contours.append(contours[pos[0]][pos[1]][mode])
    return spec_contours


# not used
def get_sub_lists(original_list, number_of_sub_list_wanted):
    sub_lists = list()
    for sub_list_count in range(number_of_sub_list_wanted):
        sub_lists.append(original_list[sub_list_count::number_of_sub_list_wanted])
    return sub_lists


def get_diastole_and_systole_frame(frame_list, cont_list):
    diastole_cont = None
    diastole_frame = None
    systole_cont = None
    systole_frame = None
    for i, cont in enumerate(cont_list):
        cont_area = get_contour_area(cont)
        if diastole_cont is None or diastole_cont < cont_area:
            diastole_cont = cont_area
            diastole_frame = frame_list[i][1]
        if systole_cont is None or systole_cont > cont_area:
            systole_cont = cont_area
            systole_frame = frame_list[i][1]
    return diastole_frame, systole_frame


# not used
def get_images_by_position(dicom, positioins):
    pos_1 = positioins[0]
    pos_2 = positioins[1]
    pos_3 = positioins[2]

    a = dicom.get_image(pos_1[0], pos_1[1]).astype(np.uint8)
    b = dicom.get_image(pos_2[0], pos_2[1]).astype(np.uint8)
    c = dicom.get_image(pos_3[0], pos_3[1]).astype(np.uint8)

    return [a, b, c]


def save_patient_to_pickle(patient, output_path):
    millis = int(round(time.time() * 1000))
    filename = "patient_" + str(millis) + ".pkl"
    full_out_path = output_path + "/" + filename

    with open(full_out_path, 'wb') as output:
        pickle.dump(patient, output)


def get_diagnosis_from_meta(meta_file):
    with open(meta_file, encoding="utf-8") as meta:
        return meta.readline().split(" ")[1]


def create_patient_pickles():
    root_directory = "C:/MyLife/School/MSc/8.felev/sample"
    output_path = "C:/MyLife/School/MSc/8.felev/Onlab/k_boti/pickles"

    global start_time
    start_time = time.time()

    patient_list = os.listdir(root_directory)
    for folder in patient_list:
        con_file = root_directory + "/" + folder + "/sa/contours.con"
        meta_file = root_directory + "/" + folder + "/meta.txt"

        if os.path.isfile(con_file):
            cr = CONreaderVM(con_file)
            try:
                patient = Patient(cr)
                patient.diagnosis = get_diagnosis_from_meta(meta_file)
                save_patient_to_pickle(patient, output_path)
            except ContourFileError as err:
                print(err)
        else:
            print("Contour file does not exist for patient - " + folder)

    print(time.time() - start_time)


def main():
    create_patient_pickles()


if __name__ == '__main__':
    main()
