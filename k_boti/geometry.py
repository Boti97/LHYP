import numpy as np
import math
import matplotlib.pyplot as plt


def rotate_point_around_point(point_to_rotate, point_to_rotate_around, angle):
    sin_angle = round(math.sin(math.radians(angle)))
    cos_angle = round(math.cos(math.radians(angle)))

    new_point_x = point_to_rotate[0] - point_to_rotate_around[0]
    new_point_y = point_to_rotate[1] - point_to_rotate_around[1]

    new_point_x = new_point_x * cos_angle - new_point_y * sin_angle
    new_point_y = new_point_x * sin_angle + new_point_y * cos_angle

    new_point_x += point_to_rotate_around[0]
    new_point_y += point_to_rotate_around[1]
    return [new_point_x, new_point_y]


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
    angle_rad = ((a_array[0]*b_array[0] + (a_array[1]*b_array[1]))/(a_array_len*b_array_len))
    return math.degrees(math.acos(angle_rad))


def move_away_point_from_another_point_on_line(point_a, point_b, unit_to_move):
    distance = get_line_length(point_a, point_b)
    point_c_x = point_a[0] + (unit_to_move/distance) * (point_b[0] - point_a[0])
    point_c_y = point_a[1] + (unit_to_move/distance) * (point_b[1] - point_a[1])
    return [point_c_x, point_c_y]


def get_closest_point_to_line(point_a, point_b, contour):
    closest_point = [0, 0]
    point_c = move_away_point_from_another_point_on_line(point_a, point_b, 100)
    a_array = np.array([point_a[0], point_a[1]])
    b_array = np.array([point_c[0], point_c[1]])
    max_tilt_angle = 0

    for contour_point in contour:
        c_array = np.array([contour_point[0], contour_point[1]])
        array_a_c = a_array - c_array
        array_b_c = b_array - c_array
        tilt_angles = []

        tilt_angle = get_tilt_angle(array_a_c, array_b_c)
        tilt_angles.append(tilt_angle)
        if tilt_angle > max_tilt_angle:
            closest_point = contour_point
            max_tilt_angle = tilt_angle

    return closest_point

