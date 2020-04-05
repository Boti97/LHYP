import numpy as np
import math


def get_center(contour):
    number_of_points = len(contour)
    x_sum = 0
    y_sum = 0
    for i in range(len(contour)):
        x_sum += contour[i][0]
        y_sum += contour[i][1]

    return x_sum/number_of_points, y_sum/number_of_points


def get_triangle_area(a, b, c):
    s = (a + b + c) / 2
    area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    return area


def get_line_length(point_a, point_b):
    x = point_a[0] - point_b[0] if point_a[0] > point_b[0] else point_b[0] - point_a[0]
    y = point_a[1] - point_b[1] if point_a[1] > point_b[1] else point_b[1] - point_a[1]
    return (x*x + y*y) ** 0.5


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

def get_array_len(array):
    return math.sqrt(abs(array[0]) ** 2 + abs(array[1]) ** 2)


def get_tilt_angle(a_array, b_array):
    a_array_len = get_array_len(a_array)
    b_array_len = get_array_len(b_array)
    x = (a_array[0]*b_array[0] + a_array[1]*b_array[1])
    z = (a_array_len*b_array_len)
    y = ((a_array[0]*b_array[0] + (a_array[1]*b_array[1]))/(a_array_len*b_array_len))

    return math.degrees(math.acos((a_array[0]*b_array[0] + (a_array[1]*b_array[1]))/(a_array_len*b_array_len)))


def get_closest_point_to_line(point_a, point_b, contour):
    closest_point = [0, 0]
    a_array = np.array([point_a[0], point_a[1]])
    b_array = np.array([point_b[0], point_b[1]])
    max_tilt_angle = 0

    for contour_point in contour:
        c_array = np.array(contour_point[0], contour_point[1])
        array_a_c = a_array - c_array
        array_b_c = b_array - c_array

        tilt_angle = get_tilt_angle(array_a_c, array_b_c)
        if tilt_angle > max_tilt_angle:
            closest_point = contour_point

    return closest_point


'''
a = np.array([(-3), 4])
b = np.array([6, 2.5])
c = get_tilt_angle(a, b)
'''
