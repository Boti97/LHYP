from area import get_contour_area
from con_reader import CONreaderVM
from dicom_reader import DCMreaderVM
import time
import numpy as np
import os
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


class Patient:

    def __init__(self, contour_reader, dicom_reader):
        contours = contour_reader.get_hierarchical_contours()

        # separate diastole and systole by the first available frame
        contour_a = None
        contour_b = None
        a_frame, b_frame = 0, 0
        for slc in contours:
            if len(contours[slc]) > 1:
                if a_frame == 0 and b_frame == 0 and len(contours[slc]) == 2:
                    a_frame, b_frame = contours[slc]

                if "lp" in contours[slc][a_frame]:
                    contour_a = contours[slc][a_frame]["lp"].tolist()
                    if "lp" in contours[slc][b_frame]:
                        contour_b = contours[slc][b_frame]["lp"].tolist()
                        break
                elif "ln" in contours[slc][a_frame]:
                    contour_a = contours[slc][a_frame]["ln"].tolist()
                    if "ln" in contours[slc][b_frame]:
                        contour_b = contours[slc][b_frame]["ln"].tolist()
                        break

        if a_frame == 0 or b_frame == 0 or contour_a is None or contour_b is None:
            return

        dia_frame, sys_frame = get_diastole_and_systole_frame(contour_a, contour_b, a_frame, b_frame)

        # set diastole images
        self.dia_images = get_images_by_position(dicom_reader, get_image_positions(contours, dia_frame))
        self.sys_images = get_images_by_position(dicom_reader, get_image_positions(contours, sys_frame))
        self.study_id = contour_reader.volume_data["Study_id="]
        self.patient_weight = contour_reader.volume_data["Patient_weight="]
        self.Patient_height = contour_reader.volume_data["Patient_height"]
        self.Patient_gender = contour_reader.volume_data["Patient_gender="]

        self.age = 0


def get_sublists(original_list, number_of_sub_list_wanted):
    sublists = list()
    for sub_list_count in range(number_of_sub_list_wanted):
        sublists.append(original_list[sub_list_count::number_of_sub_list_wanted])
    return sublists


def get_diastole_and_systole_frame(cont_a, cont_b, a_frame, b_frame):
    cont_a_area = get_contour_area(cont_a)
    cont_b_area = get_contour_area(cont_b)
    return (a_frame, b_frame) if cont_a_area > cont_b_area else (b_frame, a_frame)


def get_image_positions(contours, frame):
    cont_list = []
    for slc in contours:
        for frm in contours[slc]:
            if frm == frame:
                cont_list.append((slc, frame))

    cont_list = np.asarray(cont_list)
    cont_list = np.array_split(cont_list, 3)

    slc_frame_a = cont_list[0][len(cont_list[0]) // 2]
    slc_frame_b = cont_list[1][len(cont_list[1]) // 2]
    slc_frame_c = cont_list[2][len(cont_list[2]) // 2]

    return slc_frame_a, slc_frame_b, slc_frame_c


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


def read_patient_pickle():
    print("Pickles' directory path:")
    input_path = input()
    pickle_list = os.listdir(input_path)
    for pickle_file in pickle_list:
        full_out_path = input_path + "/" + pickle_file
        with open(full_out_path, 'rb') as input_file:
            patient = pickle.load(input_file)
            patient


def create_patient_pickles():
    print("Root directory path:")
    root_directory = input()
    print("Output pickle files' directory path:")
    output_path = input()
    """

    image_folder = 'C:/MyLife/School/MSc/8.felev/sample/8494150AMR806/sa/images'
    con_file = 'C:/MyLife/School/MSc/8.felev/sample/8494150AMR806/sa/contours.con'
    output_path = 'C:/MyLife/School/MSc/8.felev/Onlab/k_boti/pickles/'
    

    root_directory = 'C:/MyLife/School/MSc/8.felev/sample'
    output_path = 'C:/MyLife/School/MSc/8.felev/Onlab/k_boti/pickles'
    """

    patient_list = os.listdir(root_directory)
    for folder in patient_list:
        image_folder = root_directory + "/" + folder + "/sa/images"
        con_file = root_directory + "/" + folder + "/sa/contours.con"
        dr = DCMreaderVM(image_folder)
        cr = CONreaderVM(con_file)
        patient = Patient(cr, dr)
        save_patient_to_pickle(patient, output_path)


#create_patient_pickles()
#read_patient_pickle()


