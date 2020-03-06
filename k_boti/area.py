def get_point_from_within(point_a, point_b):
    normal = (point_a[1], (-1) * point_a[0])
    sum_prod = normal[0] * point_b[0] + normal[1] * point_b[1]
    x = (point_a[0] - point_b[0]) / 2 if point_a[0] > point_b[0] else (point_b[0] - point_a[0]) / 2
    y = (sum_prod + normal[0]*x)/normal[1]
    return x, y


def get_triangle_area(a, b, c):
    s = (a + b + c) / 2
    area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    print('The area of the triangle is %0.2f' % area)
    return area


def get_line_length(point_a, point_b):
    x = point_a[0] - point_b[0] if point_a[0] > point_b[0] else point_b[0] - point_a[0]
    y = point_a[1] - point_b[1] if point_a[1] > point_b[1] else point_b[1] - point_a[1]
    return (x*x + y*y) ** 0.5


def get_contour_area(contour):
    center_x, center_y = get_point_from_within(contour[0], contour[len(contour) // 2])
    contour_area = 0
    for i in range(len(contour)):
        if i+1 < len(contour):
            a = get_line_length(contour[i], contour[i + 1])
            b = get_line_length(contour[i], (center_x, center_y))
            c = get_line_length(contour[i + 1], (center_x, center_y))
            contour_area += get_triangle_area(a, b, c)
    return contour_area



