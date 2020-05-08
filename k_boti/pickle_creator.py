import time
import numpy as np
from copy import deepcopy
import os
from utils import get_logger
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


logger = get_logger(__name__)


class ContourFileError(Exception):
    def __init__(self, message):
        super().__init__(message)


class Patient:
    def __init__(self, contour_reader):
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


class CONreaderVM:

    def __init__(self, file_name):
        """
        Reads in a con file and saves the curves grouped according to its corresponding slice, frame and place.
        Finds the tags necessary to calculate the volume metrics.
        """
        self.file_name = file_name
        self.container = []
        self.contours = None

        con_tag = "XYCONTOUR"  # start of the contour data
        stop_tag = "POINT"  # if this is available, prevents from reading unnecessary lines
        volumerelated_tags = [
            'Study_id=',
            'Field_of_view=',
            'Image_resolution=',
            'Slicethickness=',
            'Patient_weight=',
            'Patient_height',
            'Study_description=',
            'Patient_gender='
        ]

        self.volume_data = {
            volumerelated_tags[0]: None,
            volumerelated_tags[1]: None,
            volumerelated_tags[2]: None,
            volumerelated_tags[3]: None,
            volumerelated_tags[4]: None,
            volumerelated_tags[5]: None,
            volumerelated_tags[6]: None,
            volumerelated_tags[7]: None
        }

        con = open(file_name, 'rt')

        def find_volumerelated_tags(line):
            for tag in volumerelated_tags:
                if line.find(tag) != -1:
                    value = line.split(tag)[1]  # the place of the tag will be an empty string, second part: value
                    self.volume_data[tag] = value

        def mode2colornames(mode):
            if mode == 0:
                return 'ln'  # left (endo)
            elif mode == 1:
                return 'lp'  # left (epi) contains the myocardium
            elif mode == 2:
                return 'rp'  # right (epi)
            elif mode == 5:
                return 'rn'  # right (endo)
            else:
                logger.warning('Unknown mode {}'.format(mode))
                return 'other'

        def find_xycontour_tag():
            line = con.readline()
            find_volumerelated_tags(line)
            while line.find(con_tag) == -1 and line.find(stop_tag) == -1 and line != "":
                line = con.readline()
                find_volumerelated_tags(line)
            return line

        def identify_slice_frame_mode():
            line = con.readline()
            splitted = line.split(' ')
            return int(splitted[0]), int(splitted[1]), mode2colornames(int(splitted[2]))

        def number_of_contour_points():
            line = con.readline()
            return int(line)

        def read_contour_points(num):
            contour = []
            for _ in range(num):
                line = con.readline()
                xs, ys = line.split(' ')
                contour.append((float(xs), float(ys)))  # unfortubately x and y are interchanged
            return contour

        line = find_xycontour_tag()
        while line.find(stop_tag) == -1 and line != "":
            slice, frame, mode = identify_slice_frame_mode()
            num = number_of_contour_points()
            contour = read_contour_points(num)
            self.container.append((slice, frame, mode, contour))
            line = find_xycontour_tag()

        con.close()
        return

    def get_hierarchical_contours(self):
        # if it is not initializedyet, then create it
        if self.contours is None:

            self.contours = {}
            for item in self.container:
                slice = item[0]
                frame = item[1]  # frame in a hearth cycle
                mode = item[2]  # mode can be red, green, yellow
                contour = item[3]

                # rearrange the contour
                d = {'x': [], 'y': []}
                for point in contour:
                    d['x'].append(point[0])
                    d['y'].append(point[1])

                if not (slice in self.contours):
                    self.contours[slice] = {}

                if not (frame in self.contours[slice]):
                    self.contours[slice][frame] = {}

                if not (mode in self.contours[slice][frame]):
                    x = d['x']
                    y = d['y']
                    N = len(x)
                    contour_mtx = np.zeros((N, 2))
                    contour_mtx[:, 0] = np.array(x)
                    contour_mtx[:, 1] = np.array(y)
                    self.contours[slice][frame][mode] = contour_mtx

        return self.contours

    def contour_iterator(self, deep=True):
        self.get_hierarchical_contours()
        for slice, frame_level in self.contours.items():
            for frame, mode_level in frame_level.items():
                if deep:
                    mode_level_cp = deepcopy(mode_level)
                else:
                    mode_level_cp = mode_level
                yield slice, frame, mode_level_cp

    def get_volume_data(self):
        # process field of view
        fw_string = self.volume_data['Field_of_view=']
        sizexsize_mm = fw_string.split('x')  # variable name shows the format
        size_h = float(sizexsize_mm[0])
        size_w = float(sizexsize_mm[1].split(' mm')[0])  # I cut the _mm ending

        # process image resolution
        img_res_string = self.volume_data['Image_resolution=']
        sizexsize = img_res_string.split('x')
        res_h = float(sizexsize[0])
        res_w = float(sizexsize[1])

        # process slice thickness
        width_string = self.volume_data['Slicethickness=']
        width_mm = width_string.split(' mm')
        width = float(width_mm[0])

        # process weight
        weight_string = self.volume_data['Patient_weight=']
        weight_kg = weight_string.split(' kg')
        weight = float(weight_kg[0])

        # process height
        # Unfortunately, patient height is not always available.
        # Study description can help in that case but its form changes heavily.
        if 'Patient_height=' in self.volume_data.keys():
            height_string = self.volume_data['Patient_height=']
            height = height_string.split(" ")[0]
        else:
            height_string = str(self.volume_data['Study_description='])
            height = ''
            for char in height_string:
                if char.isdigit():
                    height += char
        if height == '':
            logger.warning('Unknown height in con file {}'.format(self.file_name))
            height = 178
        else:
            try:
                height = float(height)
            except ValueError:
                height = 178
                logger.error(' Wrong height format in con file {}'.format(self.file_name))

        # gender
        gender = self.volume_data['Patient_gender=']

        return (size_h / res_h, size_w / res_w), width, weight, height, gender


def get_triangle_area(a, b, c):
    s = (a + b + c) / 2
    area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    return area


def get_line_length(point_a, point_b):
    x = point_a[0] - point_b[0] if point_a[0] > point_b[0] else point_b[0] - point_a[0]
    y = point_a[1] - point_b[1] if point_a[1] > point_b[1] else point_b[1] - point_a[1]
    return (x*x + y*y) ** 0.5


def get_center(contour):
    number_of_points = len(contour)
    x_sum = 0
    y_sum = 0
    for i in range(len(contour)):
        x_sum += contour[i][0]
        y_sum += contour[i][1]

    return x_sum/number_of_points, y_sum/number_of_points


def get_contour_area(contour):
    center_x, center_y = get_center(contour)
    contour_area = 0
    for i in range(len(contour)):
        if i+1 < len(contour):
            a = get_line_length(contour[i], contour[i + 1])
            b = get_line_length(contour[i], (center_x, center_y))
            c = get_line_length(contour[i + 1], (center_x, center_y))
            contour_area += get_triangle_area(a, b, c)
    return contour_area


def get_all_necessary_contours_by_frame(contours, frame):
    dia_ln_contours = []
    dia_lp_contours = []
    dia_rn_contours = []
    for slc in contours:
        if "ln" in contours[slc][frame] and "lp" in contours[slc][frame] and "rn" in contours[slc][frame]:
            dia_ln_contours.append(contours[slc][frame]["ln"])
            dia_lp_contours.append(contours[slc][frame]["lp"])
            dia_rn_contours.append(contours[slc][frame]["rn"])

    if dia_ln_contours.__len__() == 0 or dia_lp_contours.__len__() == 0 or dia_rn_contours.__len__() == 0:
        raise ContourFileError("There's no ln, lp, and rn in frame: " + str(frame))
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
    if frames.__len__() == 0:
        raise ContourFileError("Unable to locate diastoly and systoly.")
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
    if patient.study_id is not None:
        filename = "patient_" + patient.study_id + ".pkl"
        full_out_path = output_path + "/" + filename

        with open(full_out_path, 'wb') as output:
            pickle.dump(patient, output)
    else:
        print("Patient cannot be saved, because study_id is none!")


def get_diagnosis_from_meta(meta_file):
    with open(meta_file, encoding="utf-8") as meta:
        return meta.readline().split(" ")[1]


def read_patient_pickle(patient_pickles_path):
    pickle_list = os.listdir(patient_pickles_path)
    patients = []
    for pickle_file in pickle_list:
        full_out_path = patient_pickles_path + "/" + pickle_file
        with open(full_out_path, 'rb') as input_file:
            patients.append(pickle.load(input_file))

    return patients


def create_patient_pickles(root_directory, output_path):
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
