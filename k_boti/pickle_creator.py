# Reads the sa folder wiht dicom files and contours
# then draws the contours on the images.

from con_reader import CONreaderVM
from dicom_reader import DCMreaderVM
import numpy as np

from k_boti.area import get_contour_area


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
    cont_list = np.split(cont_list, 3)

    slc_frame_a = cont_list[0][len(cont_list[0]) // 2]
    slc_frame_b = cont_list[1][len(cont_list[1]) // 2]
    slc_frame_c = cont_list[2][len(cont_list[2]) // 2]

    return slc_frame_a, slc_frame_b, slc_frame_c


def get_images_by_position(dicom, positioins):
    pos_1 = positioins[0]
    pos_2 = positioins[1]
    pos_3 = positioins[2]

    a = dicom.get_image(pos_1[0], pos_1[1])
    b = dicom.get_image(pos_2[0], pos_2[1])
    c = dicom.get_image(pos_3[0], pos_3[1])

    return [a, b, c]


def input_locations_and_create_pickles():
    """
    print("Images folder path:")
    image_folder = input()
    print("Contour file path:")
    con_file = input()
    print("Output pickle files' directory path:")
    pickle_dir = input()
    """

    image_folder = 'C:/MyLife/School/MSc/8.felev/sample/8494150AMR806/sa/images'
    con_file = 'C:/MyLife/School/MSc/8.felev/sample/8494150AMR806/sa/contours.con'

    # TODO: check all the directories for different patients
    # reading the dicom files
    dr = DCMreaderVM(image_folder)
    # reading the contours
    cr = CONreaderVM(con_file)

    patient = Patient(cr, dr)


input_locations_and_create_pickles()
