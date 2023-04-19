import numpy as np
import math
# import cv2


digits_coordinates = []
digits_meaning = []
unit_of_meas = ''
with open('/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolov5_digits/runs/detect/exp_2/labels/gauge_cropped_0_real.txt', 'r') as file:
  for row in [x.split(' ') for x in file.read().strip().splitlines()]:
  # 0 - gauge, 1 - needle
    if (int(row[0]) != 10) and (int(row[0]) != 11):
      x_mid = int((int(row[1])+int(row[3]))/2)
      y_mid = int((int(row[2])+int(row[4]))/2)
      digits_coordinates.append((x_mid, y_mid))
      digits_meaning.append([int(row[0])])
    else:
      unit_of_meas = row[0]

digits_coordinates = np.array([digits_coordinates])
# print(digits_coordinates)
digits_meaning = np.array(digits_meaning)
# print(digits_meaning)

digits_meaning_copy = digits_meaning
dict_digits = {}
digits_meaning_list = digits_meaning.tolist()

digits_coordinates_deskewed=[[103.8554920433192, 277.7087432101557],
                             [341.13110455953614, 112.45341467687163],
                             [196.21303913468404, 58.237641575232004],
                             [366.3124286944232, 233.00761618863515],
                             [212.63688271827624, 58.754938232638864],
                             [326.4940075918183, 110.49771663695816],
                             [354.58729490339067, 231.28376140159548],
                             [85.28018678189078, 145.82556373955367],
                             [84.5969663673329, 277.75608507541966],
                             [67.80840772177882, 144.43450422231098]]

# digits_coordinates_deskewed=[[111.26730985960461, 82.71706004637313],
#                              [78.08020267165149, 47.23798716108391],
#                              [44.61895280917166, 81.15244868223989],
#                              [50.899066384025886, 98.98736563989364],
#                              [103.6741597442973, 105.8879425742535],
#                              [100.97957353357053, 57.88663550561304],
#                              [55.9331342643688, 57.14597444893869]]
digits_coordinates_deskewed_copy = digits_coordinates_deskewed

digits_meaning_copy = digits_meaning
dict_digits = {}
digits_meaning_list = digits_meaning.tolist()

# digits_coordinates_deskewed_copy = digits_coordinates_deskewed.tolist()


# def angle_between_digits(coord, dig):
#     # вычислить правый сектор
#     x_1 = 214
#     y_1 = 191
#     h = 397
#     w = 504
#     contours_of_sector = [(x_1, y_1), (x_1, h), (w, h), (x_1, 0)]
#     contours_of_sector_ndarray = np.array(contours_of_sector, dtype=np.int)
#     contours_of_sector_ndarray_tuple = tuple()
#     contours_of_sector_ndarray_tuple = (contours_of_sector_ndarray,)
#     contours_of_sector_flat = np.vstack(contours_of_sector_ndarray_tuple).squeeze()
#     # print('!!!')
#     # print(contours_of_sector_flat)
#     gamma_digits = {}
#     gamma_digits_index = []
#     for c,i in enumerate(coord):
#         # принадлежит ли текущая точка сектору?
#         check_point_digits = cv2.pointPolygonTest(contours_of_sector_flat, (int(i[0]), int(i[1])), False)
#         # стороны треугольника (перпиндикуляр вниз к шиирне изображения от центра манометра и линия от центра манометра до цифры )
#         d = math.sqrt((int(i[0]) - x_1) ** 2 + (int(i[1]) - h) ** 2)
#         e = math.sqrt((int(i[0]) - x_1) ** 2 + (int(i[1]) - y_1) ** 2)
#         f = math.sqrt((x_1 - x_1) ** 2 + (y_1 - h) ** 2)
#
#         # если точка в секторе, считать тупой угол, наоборот - острый
#         if int(check_point_digits) == 1:
#             gamma_digits[c] = [((180 - math.acos((e ** 2 + f ** 2 - d ** 2) / (2 * e * f)) * 180 / math.pi) + 180)]
#         else:
#             gamma_digits[c] = [(math.acos((e ** 2 + f ** 2 - d ** 2) / (2 * e * f)) * 180 / math.pi)]
#
#     # сортировка по значению угла, чем больше! угол, тем левее цифра
#     soted_gamma_digits = sorted(gamma_digits.items(), key=lambda x: x[1], reverse=True)
#     sorted_digits = []
#     # сопоставление значения цифры ее порядковому номеру
#     for j in soted_gamma_digits:
#         sorted_digits.append(dig[j[0]][0])
#     return sorted_digits

def digits_order(coord, dig):
    sorted_digits = []
    # если координата х одной цифры меньше, чем у другой, то цифра левее
    if coord[0][0]<coord[1][0]:
        sorted_digits.append(dig[0][0])
        sorted_digits.append(dig[1][0])
    else:
        sorted_digits.append(dig[1][0])
        sorted_digits.append(dig[0][0])
    return sorted_digits

black_list = []
phys_quan = ' LBS/IN_2'
# phys_quan = ' bar'


if phys_quan == ' LBS/IN_2':
    for c,i in enumerate(digits_meaning_list):
        dist = []
        if c in black_list:
            continue
        # расстояние между текущей цифрой и остальными, включая текущую
        for j in range(0, len(digits_coordinates_deskewed_copy)):
            dist.append(math.sqrt( abs((int(digits_coordinates_deskewed_copy[c][0]) - int(digits_coordinates_deskewed_copy[j][0]))^2 + (int(digits_coordinates_deskewed_copy[c][1]) - int(digits_coordinates_deskewed_copy[j][1]))^2)))
        print(dist)
        nearest_digits = []
        # проверка расстояния, если на близком расстоянии, то набор цифр - число
        for a, b in enumerate(dist):
            if ((b <= 5)):
                nearest_digits.append(a)
        x = 0
        y = 0
        coord = []
        dig = []
        for f in nearest_digits:
            f = int(f)
            # для среднего расстояние между цифрами
            x += digits_coordinates_deskewed_copy[f][0]
            y += digits_coordinates_deskewed_copy[f][1]
            # собираем массив близких цифр и их координат
            coord.append(digits_coordinates_deskewed_copy[f])
            dig.append(digits_meaning[f])
            # black_list.append(c)
            # black_list.append(f)

        # вычисляем порядок чисел
        # res = angle_between_digits(coord, dig)
        res = digits_order(coord, dig)
        digit_final = ''
        for h in res:
            digit_final += str(h)
        # считаем среднее
        x = x / len(nearest_digits)
        y = y / len(nearest_digits)
        if len(dig)>=3:
            print('wrong dist')
        else:
            # складываем в словарь
            dict_digits[int(digit_final)] = [x, y]
else:
    for i in range(len(digits_meaning_copy)):
        dict_digits[int(digits_meaning_copy[i])]=digits_coordinates_deskewed_copy[i]

    # for k, d in enumerate(dist):
    #     if ((d <= 5) and (d != 0)):
    #         # добавляем в блэк лист цифры, чтоб не повторялись
    #         black_list.append(c)
    #         black_list.append(k)
    #
    #         # среднее расстояние между цифрами
    #         x = int((digits_coordinates_deskewed_copy[c][0] + digits_coordinates_deskewed_copy[k][0]) / 2)
    #         y = int((digits_coordinates_deskewed_copy[c][1] + digits_coordinates_deskewed_copy[k][1]) / 2)
    #         # ищем с право на лево читать числа или слева на право
    #         coord = [digits_coordinates_deskewed_copy[c], digits_coordinates_deskewed_copy[k]]
    #         dig = [digits_meaning[c], digits_meaning[k]]
    #         res = angle_between_digits(coord, dig)
    #         digit_final = ''
    #         for h in res:
    #             digit_final += str(h)
    #
    #         dict_digits[int(digit_final)] = [x,y]

    # если число не двузначное
    # if int(str(digits_meaning[c][0])) not in dict_digits:
    #     dict_digits[int(str(digits_meaning[c][0]))] = [ int(digits_coordinates_deskewed_copy[c][0]) , int(digits_coordinates_deskewed_copy[c][1]) ]


print(dict_digits)