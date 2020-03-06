# Reads the sa folder wiht dicom files and contours
# then draws the contours on the images.

from con_reader import CONreaderVM
from dicom_reader import DCMreaderVM
from k_boti.area import get_contour_area
from k_boti.patient import Patient
import numpy as np


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

    slc_frame_a = cont_list[0][len(cont_list[0])//2]
    slc_frame_b = cont_list[1][len(cont_list[1])//2]
    slc_frame_c = cont_list[2][len(cont_list[2])//2]

    return slc_frame_a, slc_frame_b, slc_frame_c


def get_images_by_position(dicom, pos_1, pos_2, pos_3):
    a = dicom.get_image(pos_1[0], pos_1[1])
    
    return [dicom.get_image(pos_1[0], pos_1[1])]


image_folder = 'C:/MyLife/School/MSc/8.felev/sample/8494150AMR806/sa/images'
con_file = 'C:/MyLife/School/MSc/8.felev/sample/8494150AMR806/sa/contours.con'

'''
print("Images folder path:")
image_folder = input()
print("Contour file path:")
con_file = input()
print("Output pickle files' directory path:")
pickle_dir = input()
'''

# reading the dicom files
dr = DCMreaderVM(image_folder)
# reading the contours
cr = CONreaderVM(con_file)
contours = cr.get_hierarchical_contours()

# separate diastole and systole by the first available frame
contour_a = None
contour_b = None
a_frame, b_frame = 0, 0
for slc in contours:
    if len(contours[slc]) > 1:
        a_frame, b_frame = contours[slc]
        contour_a = contours[slc][a_frame]["lp"].tolist()
        contour_b = contours[slc][b_frame]["lp"].tolist()


dia_frame, sys_frame = get_diastole_and_systole_frame(contour_a, contour_b, a_frame, b_frame)

patient = Patient()

patient.dia_images = get_images_by_position(dr, get_image_positions(contours, dia_frame))