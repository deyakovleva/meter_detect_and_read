from __future__ import print_function
import cv2 as cv
import numpy as np
# import argparse
# import random as rng
import math
# import matplotlib.pyplot as plt
# from numpy.linalg import inv
from sklearn import linear_model, datasets

path_to_img = '/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/media/gauge_cropped_0.jpg'
path_to_needle_mask = '/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolact/res/masks_needle_0.txt'
path_to_digits = '/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolov5_digits/runs/detect/exp/labels/gauge_cropped_0_real.txt'

        getImageStatus = False
        print('start to wait')

        while (not getImageStatus):
            print('start while')
            rospy.loginfo("Yolodigits is waiting for image.")
            print('start loginfo')
            rospy.sleep(2)
            if os.path.exists(path_to_img)and os.path.exists(path_to_needle_mask) and os.path.exists(path_to_digits) :
                print('true')
                getImageStatus = True

src = cv.imread('/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/media/gauge_cropped_099.jpg')


x_pic = []
y_pic = []
list_of_coord = []

with open(f'/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolact/res/bar_lbs/masks_mano_0.txt', 'r') as file:
  for row in [x.split(' ') for x in file.read().strip().splitlines()]:
    list_of_coord = row

for i in range(len(list_of_coord)):
  if i % 2 == 0:
    x_pic.append(int(list_of_coord[i]))
  else:
    y_pic.append(int(list_of_coord[i]))


contours_of_mask = []
for i in range(len(y_pic)):
  contours_of_mask.append((y_pic[i],x_pic[i]))

contours_of_mask_ndarray = np.array(contours_of_mask , dtype=int)
contours_of_mask_ndarray_tuple = tuple()
contours_of_mask_ndarray_tuple = (contours_of_mask_ndarray,)

ellipse_ar = tuple()


contours_flat = np.vstack(contours_of_mask_ndarray_tuple).squeeze()

ellipse = cv.fitEllipse(contours_flat)
(x_el, y_el), (MA, ma), angle = cv.fitEllipse(contours_flat)

src_with_max_contour = src.copy()
cv.ellipse(src_with_max_contour, ellipse, (0, 255, 0), 3)
box = cv.boxPoints(ellipse)
box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
cv.drawContours(src_with_max_contour, [box], 0, (0,0,255), 3)


rows_src, cols_src = src.shape[0:2]
# Convert image to gray and blur it
# src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# src_blured_center = src.copy()
# src_gray = cv.blur(src_gray, (5,5))

print('Input')
# cv2_imshow(src)


print('Max Contour')
# cv2_imshow(src_with_max_contour)

# convert ellipse to circle
src_el2c = src.copy()
scale = MA/ma
M = cv.getRotationMatrix2D((src_el2c.shape[0]/2, src_el2c.shape[1]/2), angle, 1)
# Let's add the scaling too:
M[:,0:2] = np.array([[1,0],[0,scale]]) @ M[:,0:2]
M[1,2] = M[1,2] * scale # This moves the ellipse so it doesn't end up outside the image (it's not correct to keep the ellipse in the middle of the image)
rows_el2c, cols_el2c = src_el2c.shape[0:2]

# rect = cv.rectangle(src_el2c, xy_left, xy_right, (0,0,255), 1)

ellipce2circle = cv.warpAffine(src_el2c, M, (rows_el2c,cols_el2c), borderMode=cv.BORDER_REPLICATE)

print('Ellipse2circle')
# cv2_imshow(ellipce2circle)


vect_el = np.array([x_el, y_el, 1])
new_el_coord = M.dot(vect_el.T)

# draw circle
src_with_circle = ellipce2circle.copy()
cv.ellipse(src_with_circle, ((new_el_coord[0], new_el_coord[1]), (MA,MA), 180-angle), (0, 255, 0), 3) # менять угол
cir = ((new_el_coord[0], new_el_coord[1]), (MA,MA), 180-angle) # менять угол
box_cir = cv.boxPoints(cir)
box_cir = np.intp(box_cir)
cv.drawContours(src_with_circle, [box_cir], 0, (0,0,255), 3)

print('Circle')
cv.imwrite('/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/measurement/Circle_0.jpg', src_with_circle)
# cv2_imshow(src_with_circle)

# deskewing rotated box
src_deskewing = ellipce2circle.copy()
(h, w) = src_deskewing.shape[:2]
center = (w // 2, h // 2)

if cir[1][0]<cir[1][1]:
# rotate our image by -90 degrees around the image
    print('w<h')
    M_box = cv.getRotationMatrix2D(center, cir[2]-90, 1.0)

else:
    print('h<w')
    M_box = cv.getRotationMatrix2D(center, cir[2]+180, 1.0)



# M_box = cv.getRotationMatrix2D(center, 360-angle, 1.0) # менять угол
src_deskewing = cv.warpAffine(src_deskewing, M_box, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

print('Deskewing')
cv.imshow('Deskw',src_deskewing)
cv.waitKey(0)

src_with_helplines = src_deskewing.copy()

src_scale = src.copy()
src_scale_el = src.copy()


# rect1 = cv.rectangle(src_scale, xy_left, xy_right, (255,0,0), 1)
# cv2_imshow(src_scale)


##############################################################################
# read coordinates from file
xy_left = tuple()
xy_right = tuple()
with open(f'/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolact/res/bar_lbs/boxes_0.txt', 'r') as file:
  for row in [x.split(' ') for x in file.read().strip().splitlines()]:
  # 0 - gauge, 1 - needle
    if row[0] == '0':
      xy_left = (int(row[1]), int(row[2]))
      xy_right = (int(row[3]), int(row[4]))
##############################################################################

rows_scale, cols_scale = ellipce2circle.shape[0:2]

# w = xy_right[0] - xy_left[0]
# h = xy_right[1] - xy_left[1]
# rect = cv.rectangle(src_scale, xy_left, xy_right, (0,0,255), 1)
# cv2_imshow(src_scale)

(x,y) = (xy_right[1] + xy_left[1])/2, (xy_right[0] + xy_left[0])/2

# src_scale[int(x),int(y)]=[0,255,0]
# print('center!!!!!!!!!!!')
# cv2_imshow(src_scale)


vect = np.array([y,x,1])
T = M.dot(vect.T)
ellipce2circle[int(T[1]),int(T[0])] = [0,0,255]

vect_box = np.array([int(T[0]),int(T[1]), 1])
T_box = M_box.dot(vect_box.T) # new center
src_with_helplines[int(T_box[1]),int(T_box[0])] = [0,0,255]

y_1 = int(T_box[1])
x_1 = int(T_box[0])

# helpline_0 = cv.line(src_with_helplines, ( x_1,y_1 ), (x_1,h), (0,0,255), 1 )

# calculate line 45 & 315 using rotation matrix
# vect1 = np.array([x_1 - (w - y_1),h,1])
# vect2 = np.array([x_1 - (w + y_1),h,1])
# T1 = M.dot(vect1.T)
# T2 = M.dot(vect2.T)
x_45 = x_1 - (h - y_1)
y_45 = h
x_315 = x_1 + (h - y_1)
y_315 = h
# helpline_45 = cv.line(src_with_helplines, ( x_1, y_1 ), ( x_45, y_45 ), (0,255,0), 3 )
# helpline_315 = cv.line(src_with_helplines, ( x_1, y_1 ), ( x_315, y_315 ), (255,0,0), 3 )


x_pic_needle = []
y_pic_needle = []
list_of_coord_needle = []

with open(f'/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolact/res/bar_lbs/masks_needle_0.txt', 'r') as file:
  for row in [x.split(' ') for x in file.read().strip().splitlines()]:
    list_of_coord_needle = row

for i in range(len(list_of_coord_needle)):
  if i % 2 == 0:
    x_pic_needle.append(int(list_of_coord_needle[i]))
  else:
    y_pic_needle.append(int(list_of_coord_needle[i]))


src_check_needle = src.copy()
for i in range(len(y_pic_needle)):
  src_check_needle[x_pic_needle[i],y_pic_needle[i]] = [0,0,255]

print('Check needle')
# cv2_imshow(src_check_needle)



########### RANSAc
y_pic_needle_arr = (np.array(y_pic_needle)).reshape(-1, 1)
x_pic_needle_arr = (np.array(x_pic_needle)).reshape(-1, 1)

ransac = linear_model.RANSACRegressor()
ransac.fit(x_pic_needle_arr, y_pic_needle_arr)

line_y_ransac = ransac.predict(x_pic_needle_arr)

start_point = (int(line_y_ransac[0]),x_pic_needle_arr[0])
end_point = (int(line_y_ransac[-1]),x_pic_needle_arr[-1])

# needle_end_x = 80
# needle_end_y = 92
# needle_start_x = 33
# needle_start_y = 124
#
# start_point=(33, 124)
# end_point=(92, 80)

# calculate line coordinates for deskewed pic
# return to ellipse2circle transform
start_point_arr = np.array([start_point[0],start_point[1], 1])
end_point_arr = np.array([end_point[0],end_point[1], 1])

start_point_needle_toelpic = M.dot(start_point_arr.T)
end_point_needle_toelpic = M.dot(end_point_arr.T)



# src_el_check = ellipce2circle.copy()
# cv.line(src_el_check, (int(start_point_needle_toelpic[0]), int(start_point_needle_toelpic[1])), (int(end_point_needle_toelpic[0]),int(end_point_needle_toelpic[1])), (255,255,0), 2, cv.LINE_AA)
# print('src_el_check')
# cv2_imshow(src_el_check)



# return to deskewing transform
start_point_arr_todeskwpic = np.array([start_point_needle_toelpic[0],start_point_needle_toelpic[1], 1])
end_point_arr_todeskwpic = np.array([end_point_needle_toelpic[0],end_point_needle_toelpic[1], 1])

start_point_needle_todeskwpic = M_box.dot(start_point_arr_todeskwpic.T)
end_point_needle_todeskwpic = M_box.dot(end_point_arr_todeskwpic.T)



# src_ds_check = src_deskewing.copy()
# cv.line(src_ds_check, (int(start_point_needle_todeskwpic[0]), int(start_point_needle_todeskwpic[1])), (int(end_point_needle_todeskwpic[0]),int(end_point_needle_todeskwpic[1])), (255,255,0), 2, cv.LINE_AA)
# print('src_ds_check')
# cv2_imshow(src_ds_check)



# cv.line(src_with_helplines, (int(start_point_needle_todeskwpic[0]),int(start_point_needle_todeskwpic[1])), (int(end_point_needle_todeskwpic[0]),int(end_point_needle_todeskwpic[1])), (255,100,200), 2, cv.LINE_AA)

d1 = math.sqrt( abs((int(start_point_needle_todeskwpic[0]) - x_1)^2 + (int(start_point_needle_todeskwpic[1]) - y_1)^2))
d2 = math.sqrt( abs((int(end_point_needle_todeskwpic[0]) - x_1)^2 + (int(end_point_needle_todeskwpic[1]) - y_1)^2))
print('d1d2')
print(d1)
print(d2)

if d1<d2:
  needle_end_x = int(end_point_needle_todeskwpic[0])
  needle_end_y = int(end_point_needle_todeskwpic[1])
  needle_start_x = int(start_point_needle_todeskwpic[0])
  needle_start_y = int(start_point_needle_todeskwpic[1])
else:
  needle_end_x = int(start_point_needle_todeskwpic[0])
  needle_end_y = int(start_point_needle_todeskwpic[1])
  needle_start_x = int(end_point_needle_todeskwpic[0])
  needle_start_y = int(end_point_needle_todeskwpic[1])


# from center to end point of needle
cv.line(src_with_helplines, (needle_start_x, needle_start_y), (needle_end_x,needle_end_y), (255,255,0), 1, cv.LINE_AA)
# cv2_imshow(src_with_helplines)
cv.imshow('Result', src_with_helplines)
cv.waitKey(0)

# l_crop2full = (l_array[1][0]+int(xy_left_needle_new_[0]), l_array[1][1] + int(xy_left_needle_new_[1]))


# needle_line = cv.line(src_with_helplines, (l_crop2full[0], l_crop2full[1]), (x_1,y_1), (0,200,200), 3, cv.LINE_AA)
# print('Help lines')
# cv2_imshow(src_with_helplines)



#### check if angle is bigger than 180 ####

src_check_help_lines = src_with_helplines.copy()

x_215 = x_1 +(h - (h - y_1))
y_215 = 0

# cv.line(src_check_help_lines, (x_45,y_45), (int(x_215), int(y_215)), (200,200,200), 1, cv.LINE_AA)
# cv2_imshow(src_check_help_lines)

src_with_sector = src_deskewing.copy()

pts = np.array([ [x_1, y_1], [x_1, h], [w, h], [x_215, y_215] ])
cv.polylines(src_with_sector, [pts], True, (200,100,200), 2)
# print('Sector')
# cv.imshow('Sector',src_with_sector)
# cv.waitKey(0)

contours_of_sector = [(x_1, y_1), (x_1, h), (w, h), (x_215, y_215)]
contours_of_sector_ndarray = np.array(contours_of_sector, dtype=int)
contours_of_sector_ndarray_tuple = tuple()
contours_of_sector_ndarray_tuple = (contours_of_sector_ndarray,)
contours_of_sector_flat = np.vstack(contours_of_sector_ndarray_tuple).squeeze()

################################################
# calculate angle

a = math.sqrt( (needle_start_x - x_45 )**2 +  (needle_start_y - y_45 )**2)
b = math.sqrt( (needle_start_x - x_1 )**2 +  (needle_start_y - y_1)**2)
c = math.sqrt( (x_1 - x_45 )**2 +  (y_1 - y_45 )**2)

gamma = math.acos( (b**2+c**2-a**2)/(2*b*c) ) *180/math.pi

# check point
check_point = cv.pointPolygonTest(contours_of_sector_flat, (needle_start_x,needle_start_y), False)

if int(check_point)==1:
  print('angle is in sector, + 180')
  gamma = 360 - gamma
elif int(check_point)==-1:
  print('ok')
else:
  print('on line')
################################################
# show the measurement
# print('gamma')
# print(gamma)
# max_scale = 6
# meas = gamma*max_scale/270
# print('Measurement')
# print(meas)
#
# print('Result')

# cv2_imshow(src_with_helplines)

digits_coordinates = []
digits_meaning = []
phys_quan = ''
with open(f'/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolov5_digits/runs/detect/low_font_bar/labels/gauge_cropped_0_real.txt', 'r') as file:
  for row in [x.split(' ') for x in file.read().strip().splitlines()]:
  # 0 - gauge, 1 - needle
    if (int(row[0]) != 10) and (int(row[0]) != 11):
      if int(row[0]) == 0:
          x_mid = int((int(row[1]) + int(row[3])) / 2)
          y_mid = int((int(row[2]) + int(row[4])) / 2)+10
      else:
          x_mid = int((int(row[1])+int(row[3]))/2)
          y_mid = int((int(row[2])+int(row[4]))/2)
      digits_coordinates.append((x_mid, y_mid))
      digits_meaning.append([int(row[0])])
    else:
      phys_quan_classnum = row[0]
print(digits_coordinates)

# phys_quan_classnum = '11'
if phys_quan_classnum == '10':
    phys_quan = ' bar'
else:
    phys_quan = ' LBS/IN_2'

digits_coordinates = np.array([digits_coordinates])
# print(digits_coordinates)
digits_meaning = np.array(digits_meaning)
# print(digits_meaning)

src_digits = src.copy()
for i in digits_coordinates[0]:
    src_digits[i[1], i[0]] = [0, 255, 0]

# cv2_imshow(src_digits)
# deskewing coordinates
src_digits_final = src_deskewing.copy()

digits_desk_arr = []
digits_el_arr = []

digits_coordinates_deskewed = []

for i in digits_coordinates[0]:
    digits_el_arr = np.array([i[0], i[1], 1])
    digits_arr_toelpic = M.dot(digits_el_arr.T)
    digits_desk_arr = np.array([digits_arr_toelpic[0], digits_arr_toelpic[1], 1])
    digits_arr_todeskpic = M_box.dot(digits_desk_arr.T)
    digits_coordinates_deskewed.append([digits_arr_todeskpic[0], digits_arr_todeskpic[1]])
    src_digits_final[int(digits_arr_todeskpic[1]), int(digits_arr_todeskpic[0])] = [0, 255, 0]

# cv.imshow('dig',src_digits_final)
# cv.waitKey(0)
# print(digits_coordinates_deskewed)

digits_meaning_copy = digits_meaning
dict_digits = {}
digits_meaning_list = digits_meaning.tolist()
digits_coordinates_deskewed_copy = digits_coordinates_deskewed


# def angle_between_digits(coord, dig):
#     # вычислить правый сектор
#     # x_1 = 256
#     # y_1 = 444
#     # h = 534
#     # w = 799
#     contours_of_sector = [(x_1, y_1), (x_1, h), (w, h), (x_1, 0)]
#     contours_of_sector_ndarray = np.array(contours_of_sector, dtype=np.int)
#     contours_of_sector_ndarray_tuple = tuple()
#     contours_of_sector_ndarray_tuple = (contours_of_sector_ndarray,)
#     contours_of_sector_flat = np.vstack(contours_of_sector_ndarray_tuple).squeeze()
#     gamma_digits = {}
#     gamma_digits_index = []
#     for c,i in enumerate(coord):
#         # принадлежит ли текущая точка сектору?
#         check_point_digits = cv.pointPolygonTest(contours_of_sector_flat, (int(i[0]), int(i[1])), False)
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
#     # сортировка по значению угла, чем меньше угол, тем левее цифра
#     soted_gamma_digits = sorted(gamma_digits.items(), key=lambda x: x[1])
#     print(soted_gamma_digits)
#     sorted_digits = []
#     # сопоставление значения цифры ее порядковому номеру
#     for j in soted_gamma_digits:
#         sorted_digits.append(dig[j[0]][0])
#
#     print(sorted_digits)
#     return sorted_digits
#
#
# black_list = []
#
# for c,i in enumerate(digits_meaning_list):
#     dist = []
#     if c in black_list:
#         continue
#     # расстояние между текущей цифрой и остальными, включая текущую
#     for j in range(0, len(digits_coordinates_deskewed_copy)):
#         dist.append(math.sqrt( abs((int(digits_coordinates_deskewed_copy[c][0]) - int(digits_coordinates_deskewed_copy[j][0]))^2 + (int(digits_coordinates_deskewed_copy[c][1]) - int(digits_coordinates_deskewed_copy[j][1]))^2)))
#
#     nearest_digits = []
#     # проверка расстояния, если на близком расстоянии, то набор цифр - число
#     for a, b in enumerate(dist):
#         if ((b <= 1)):
#             nearest_digits.append(a)
#     x = 0
#     y = 0
#     coord = []
#     dig = []
#     for f in nearest_digits:
#         f = int(f)
#         # для среднего расстояние между цифрами
#         x += digits_coordinates_deskewed_copy[f][0]
#         y += digits_coordinates_deskewed_copy[f][1]
#         # собираем массив близких цифр и их координат
#         coord.append(digits_coordinates_deskewed_copy[f])
#         dig.append(digits_meaning[f])
#         # black_list.append(c)
#         # black_list.append(f)
#
#     # вычисляем порядок чисел
#     res = angle_between_digits(coord, dig)
#     digit_final = ''
#     for u in res:
#         digit_final += str(u)
#     # считаем среднее
#     x = x / len(nearest_digits)
#     y = y / len(nearest_digits)
#     # складываем в словарь
#     dict_digits[int(digit_final)] = [x, y]

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

print(dict_digits)

gamma_digits = []
# x_1 = x_1-5
# y_1 = y_1-5
(h, w) = src_digits_final.shape[:2]
contours_of_sector = [(w, 0), (w, h), (x_1, h),  (x_1, 0)]
contours_of_sector_ndarray = np.array(contours_of_sector, dtype=int)
contours_of_sector_ndarray_tuple = tuple()
contours_of_sector_ndarray_tuple = (contours_of_sector_ndarray,)
contours_of_sector_flat = np.vstack(contours_of_sector_ndarray_tuple).squeeze()

for i in dict_digits.values():
    print(i)
    check_point_digits = cv.pointPolygonTest(contours_of_sector_flat, (int(i[0]), int(i[1])), False)
    # print(check_point_digits)
    d = math.sqrt((int(i[0]) - x_1) ** 2 + (int(i[1]) - h) ** 2)
    e = math.sqrt((int(i[0]) - x_1) ** 2 + (int(i[1]) - y_1) ** 2)
    f = math.sqrt((x_1 - x_1) ** 2 + (y_1 - h) ** 2)

    if int(check_point_digits) == 1:
        gamma_digits.append((180 - math.acos((e ** 2 + f ** 2 - d ** 2) / (2 * e * f)) * 180 / math.pi) + 180)
    else:
        gamma_digits.append(math.acos((e ** 2 + f ** 2 - d ** 2) / (2 * e * f)) * 180 / math.pi)

    cv.line(src_digits_final, (x_1, y_1), (int(i[0]), int(i[1])), (255, 255, 0), 2, cv.LINE_AA)

# print(gamma_digits)
# cv2_imshow(src_digits_final)

pts = np.array([[w, 0], [w, h], [x_1, h], [x_1, 0]])
cv.polylines(src_digits_final, [pts], True, (200, 100, 200), 2)
cv.imshow('Sector_1', src_digits_final)
cv.waitKey(0)
########### RANSAc
dict_digits_keys_arr = []
for i in dict_digits.keys():
  dict_digits_keys_arr.append(i)

gamma_digits_arr = (np.array(gamma_digits).reshape(-1, 1))
digits_meaning_arr = (np.array(dict_digits_keys_arr).reshape(-1, 1))
print(gamma_digits_arr)
print(digits_meaning_arr)
ransac = linear_model.RANSACRegressor()
ransac.fit(gamma_digits_arr, digits_meaning_arr)
line_y_dig_ransac = ransac.predict(gamma_digits_arr)

# start_point_dig = (int(line_y_dig_ransac[0]),gamma_digits_arr[0])
# end_point_dig = (int(line_y_dig_ransac[-1]),gamma_digits_arr[-1])

# d = math.sqrt( (needle_end_x - x_1 )**2 +  (needle_end_y - h )**2)
# e = math.sqrt( (needle_end_x - x_1 )**2 +  (needle_end_y - y_1)**2)
# f = math.sqrt( (x_1 - x_1 )**2 +  (h - y_1)**2)
d = math.sqrt( (needle_end_x - x_1 )**2 +  (needle_end_y - h )**2)
e = math.sqrt( (needle_end_x - x_1 )**2 +  (needle_end_y - y_1)**2)
f = math.sqrt( (x_1 - x_1 )**2 +  (h - y_1)**2)
# print(d)
# print(e)
# print(f)

gamma_needle = math.acos( (e**2+f**2-d**2)/(2*e*f) ) *180/math.pi-2
print(gamma_needle)

print(min(gamma_digits_arr))
print(max(gamma_digits_arr))
print(min(line_y_dig_ransac))
print(max(line_y_dig_ransac))
# measurement  = ((max(line_y_dig_ransac))*(gamma_needle-min(gamma_digits_arr)))/(max(gamma_digits_arr) - min(gamma_digits_arr))
# print(measurement)


vmin = min(line_y_dig_ransac)
qmin = min(gamma_digits_arr)
qmax = max(gamma_digits_arr)
vmax = max(line_y_dig_ransac)
# if gamma_needle<min(gamma_digits_arr)[0]:
#     vmin = 0
m_1 = (vmax-vmin)
# print(m_1 )
measurement = vmin + ((gamma_needle-qmin)/(qmax-qmin))*(vmax-vmin)

# if measurement<0:
#   measurement =0.0

print('Measurement')
if phys_quan is not None:
    print(str(np.round(measurement[0],5))+phys_quan)
else:
    print(measurement)
cv.putText(src_with_helplines, str(np.round(measurement[0],2))+phys_quan, (w-130,20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv.LINE_AA)

src_with_helplines[y_1, x_1] = (0,0,255)
cv.imshow('Result', src_with_helplines)
cv.imwrite('/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/measurement/gauge_0.jpg', src_with_helplines)

cv.waitKey(0)