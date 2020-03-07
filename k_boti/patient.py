from k_boti.pickle_creator import get_diastole_and_systole_frame, get_images_by_position, get_image_positions


class Patient:

    def __init__(self, contour_reader, dicom_reader):
        self.contours = contour_reader.get_hierarchical_contours()
        self.age = 0

        # separate diastole and systole by the first available frame
        contour_a = None
        contour_b = None
        a_frame, b_frame = 0, 0
        for slc in self.contours:
            if len(self.contours[slc]) > 1:
                a_frame, b_frame = self.contours[slc]
                contour_a = self.contours[slc][a_frame]["lp"].tolist()
                contour_b = self.contours[slc][b_frame]["lp"].tolist()

        dia_frame, sys_frame = get_diastole_and_systole_frame(contour_a, contour_b, a_frame, b_frame)

        # set diastole images
        self.dia_images = get_images_by_position(dicom_reader, get_image_positions(self.contours, dia_frame))
        self.sys_images = get_images_by_position(dicom_reader, get_image_positions(self.contours, sys_frame))
        self.study_id = contour_reader.volume_data[0].value()
        self.patient_weight = contour_reader.volume_data[4].value()
        self.Patient_height = contour_reader.volume_data[5].value()
        self.Patient_gender = contour_reader.volume_data[7].value()





