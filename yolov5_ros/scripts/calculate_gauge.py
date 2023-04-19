from __future__ import print_function
import cv2 as cv
import numpy as np
import math

from sklearn import linear_model

input_filename = '/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolact/images/0002.jpg'
cv2_img = cv.imread(input_filename)
x_pic = []
y_pic = []
list_of_coord = []

with open(f'/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolact/res/masks_mano.txt', 'r') as file:
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

src_with_max_contour = cv2_img.copy()
cv.ellipse(src_with_max_contour, ellipse, (0, 255, 0), 3)
box = cv.boxPoints(ellipse)
box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
cv.drawContours(src_with_max_contour, [box], 0, (0,0,255), 3)

cv.imshow('Input', cv2_img)

# cv.imshow('Max Contour', src_with_max_contour)

# convert ellipse to circle
src_el2c = cv2_img.copy()
scale = MA/ma
M = cv.getRotationMatrix2D((src_el2c.shape[0]/2, src_el2c.shape[1]/2), angle, 1)
# Let's add the scaling:
M[:,0:2] = np.array([[1,0],[0,scale]]) @ M[:,0:2]
M[1,2] = M[1,2] * scale # This moves the ellipse so it doesn't end up outside the image (it's not correct to keep the ellipse in the middle of the image)
rows_el2c, cols_el2c = src_el2c.shape[0:2]

ellipce2circle = cv.warpAffine(src_el2c, M, (rows_el2c,cols_el2c), borderMode=cv.BORDER_REPLICATE)

# cv.imshow('Ellipse2circle', ellipce2circle)

vect_el = np.array([x_el, y_el, 1])
new_el_coord = M.dot(vect_el.T)

# draw circle
src_with_circle = ellipce2circle.copy()
cv.ellipse(src_with_circle, ((new_el_coord[0], new_el_coord[1]), (MA,MA), 180-angle), (0, 255, 0), 1) # менять угол
cir = ((new_el_coord[0], new_el_coord[1]), (MA,MA), 180-angle) # менять угол
box_cir = cv.boxPoints(cir)
box_cir = np.intp(box_cir)
cv.drawContours(src_with_circle, [box_cir], 0, (0,0,255), 1)

# cv.imshow('Circle', src_with_circle)

# deskewing rotated box
src_deskewing = ellipce2circle.copy()
(h, w) = src_deskewing.shape[:2]
center = (w // 2, h // 2)

# M_box = cv.getRotationMatrix2D(center, 360-angle, 1.0) # менять угол
if cir[1][0] < cir[1][1]:
    # rotate our image by -90 degrees around the image
    print('w<h')
    M_box = cv.getRotationMatrix2D(center, cir[2] - 90, 1.0)
else:
    print('h<w')
    M_box = cv.getRotationMatrix2D(center, cir[2] + 185, 1.0)
src_deskewing = cv.warpAffine(src_deskewing, M_box, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

# cv.imshow('Deskewing', src_deskewing)

# find center of gauge
src_with_helplines = src_deskewing.copy()

# read coordinates from file
xy_left = tuple()
xy_right = tuple()

with open(f'/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolact/res/boxes.txt', 'r') as file:
    for row in [x.split(' ') for x in file.read().strip().splitlines()]:
        # 0 - gauge, 1 - needle
        if row[0] == '0':
            xy_left = (int(row[1]), int(row[2]))
            xy_right = (int(row[3]), int(row[4]))


(x,y) = (xy_right[1] + xy_left[1])/2, (xy_right[0] + xy_left[0])/2
vect = np.array([y,x,1])
T = M.dot(vect.T)
ellipce2circle[int(T[1]),int(T[0])] = [0,0,255]

vect_box = np.array([int(T[0]),int(T[1]), 1])
T_box = M_box.dot(vect_box.T) # new center
src_with_helplines[int(T_box[1]),int(T_box[0])] = [0,0,255]

y_1 = int(T_box[1])
x_1 = int(T_box[0])

helpline_0 = cv.line(src_with_helplines, ( x_1,y_1 ), (x_1,h), (0,0,255), 1 )

# calculate line 45 & 315
x_45 = x_1 - (h - y_1)
y_45 = h
x_315 = x_1 + (h - y_1)
y_315 = h
helpline_45 = cv.line(src_with_helplines, ( x_1, y_1 ), ( x_45, y_45 ), (0,255,0), 1 )
helpline_315 = cv.line(src_with_helplines, ( x_1, y_1 ), ( x_315, y_315 ), (255,0,0), 1 )

# cv.imshow('Help lines', src_with_helplines)

# find needle using yolact masks
src_needle = src_deskewing.copy()
src_check_needle = cv2_img.copy()

# reading needle mask coordinates
x_pic_needle = []
y_pic_needle = []
list_of_coord_needle = []

with open(f'/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolact/res/masks_needle.txt', 'r') as file:
    for row in [x.split(' ') for x in file.read().strip().splitlines()]:
        list_of_coord_needle = row

    for i in range(len(list_of_coord_needle)):
        if i % 2 == 0:
            x_pic_needle.append(int(list_of_coord_needle[i]))
        else:
            y_pic_needle.append(int(list_of_coord_needle[i]))

for i in range(len(y_pic_needle)):
    src_check_needle[x_pic_needle[i],y_pic_needle[i]] = [0,0,255]

# cv.imshow('Check needle', src_check_needle)

y_pic_needle_arr = (np.array(y_pic_needle)).reshape(-1, 1)
x_pic_needle_arr = (np.array(x_pic_needle)).reshape(-1, 1)

ransac = linear_model.RANSACRegressor()
ransac.fit(x_pic_needle_arr, y_pic_needle_arr)

line_y_ransac = ransac.predict(x_pic_needle_arr)

start_point = (int(line_y_ransac[0]),x_pic_needle_arr[0])
end_point = (int(line_y_ransac[-1]),x_pic_needle_arr[-1])

# calculate line coordinates for deskewed pic
# return to ellipse2circle transform
start_point_arr = np.array([start_point[0],start_point[1], 1])
end_point_arr = np.array([end_point[0],end_point[1], 1])

start_point_needle_toelpic = M.dot(start_point_arr.T)
end_point_needle_toelpic = M.dot(end_point_arr.T)

# return to deskewing transform
start_point_arr_todeskwpic = np.array([start_point_needle_toelpic[0],start_point_needle_toelpic[1], 1])
end_point_arr_todeskwpic = np.array([end_point_needle_toelpic[0],end_point_needle_toelpic[1], 1])

start_point_needle_todeskwpic = M_box.dot(start_point_arr_todeskwpic.T)
end_point_needle_todeskwpic = M_box.dot(end_point_arr_todeskwpic.T)

d1 = math.sqrt( abs((int(start_point_needle_todeskwpic[0]) - x_1)^2 + (int(start_point_needle_todeskwpic[1]) - y_1)^2))
d2 = math.sqrt( abs((int(end_point_needle_todeskwpic[0]) - x_1)^2 + (int(end_point_needle_todeskwpic[1]) - y_1)^2))

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

needle_line = cv.line(src_with_helplines, (needle_start_x,needle_start_y), (needle_end_x,needle_end_y), (255,255,0), 2, cv.LINE_AA)


# calculate angle

#### check if angle is bigger than 180 ####

# src_check_help_lines = src_with_helplines.copy()

x_215 = x_1 +(h - (h - y_1))
y_215 = 0

# cv.line(src_check_help_lines, (x_45,y_45), (int(x_215), int(y_215)), (200,200,200), 1, cv.LINE_AA)
# cv2_imshow(src_check_help_lines)

src_with_sector = src_deskewing.copy()

pts = np.array([ [x_1, y_1], [x_1, h], [w, h], [x_215, y_215] ])
cv.polylines(src_with_sector, [pts], True, (200,100,200), 2)
# cv.imshow('Sector', src_with_sector)

contours_of_sector = [(x_1, y_1), (x_1, h), (w, h), (x_215, y_215)]
contours_of_sector_ndarray = np.array(contours_of_sector, dtype=np.int)
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
res = cv.pointPolygonTest(contours_of_sector_flat, (needle_start_x,needle_start_y), False)

if int(res)==1:
    print('angle is in sector, + 180')
    gamma = 360 - gamma
elif int(res)==-1:
    print('ok')
else:
    print('on line')

# show the measurement
print('gamma')
print(gamma)
mano_class_path = input_filename
mano_class_filename = mano_class_path.split("/")[10]
only_filename = mano_class_filename.split(".")[0]
mano_class = only_filename[0:4]
print('Manometers class from image retrieval')
print(mano_class)

if mano_class == '0005':
    scale_start = 0
    scale_end = 6
    meas = scale_start+gamma*int(scale_end)/270
    phys_quan = ' bar'
elif mano_class == '0002':
    scale_start = 7
    scale_end = 55
    meas = scale_start+gamma*int(scale_end)/270
    phys_quan = ' LBS/IN_2'


print('Measurement')
print(str(np.round(meas,5))+phys_quan)
cv.putText(src_with_helplines, str(np.round(meas,2))+phys_quan, (needle_end_x-60,needle_end_y-20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
cv.imshow('Result', src_with_helplines)
cv.imwrite('/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/measurement/gauge_enrob_02.jpg', src_with_helplines)
cv.waitKey(0)


