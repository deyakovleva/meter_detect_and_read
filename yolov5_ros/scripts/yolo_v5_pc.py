#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from cv2 import aruco
import torch
import math
import random

import rospy
import numpy as np
import ros_numpy
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Point
from yolov5_ros_msgs.msg import BoundingBox, BoundingBoxes
from yolov5_ros_msgs.srv import counter_response_crop, counter_response_cropResponse, gauge_response_crop, gauge_response_cropResponse, ssdisplay_response_crop, ssdisplay_response_cropResponse
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pcl2
import pyrealsense2 as rs
import message_filters


class Yolo_Dect:

    flag_for_cropping_counter = False
    flag_for_cropping_gauge = False
    flag_for_cropping_ssdisplay = False
    im_rate = 0
    det_counter_id = String()
    det_gauge_id = String()
    det_ssdisplay_id = String()
    boundingBoxes = BoundingBoxes()
    calibration_matrix_path = '/home/diana/aruco_detect/ArUCo-Markers-Pose-Estimation-Generation-Python/calibration_matrix.npy'
    distortion_coefficients_path = '/home/diana/aruco_detect/ArUCo-Markers-Pose-Estimation-Generation-Python/distortion_coefficients.npy'
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)
    # pointcloud = PointCloud()

    def __init__(self):

        self.cv_brdg = CvBridge()

        # load parameters
        yolov5_path = rospy.get_param('/yolov5_path', '/home/itmo/yolov5_ws/src/yolov5_ros_wmeters/yolov5_ros/yolov5_ros/yolov5')

        weight_path = rospy.get_param('~weight_path', '/home/itmo/yolov5_ws/src/yolov5_ros_wmeters/yolov5_ros/yolov5_ros/weights/meters_weights.pt')
        image_topic = rospy.get_param(
            '~image_topic', '/camera/color/image_raw')
        depth_topic = rospy.get_param(
            '~depth_topic', '/camera/aligned_depth_to_color/image_raw')
        pc_topic = rospy.get_param(
            '~pc_topic','/camera/depth/color/points')

        pub_topic = rospy.get_param('~pub_topic', '/yolov5/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_frame')
        conf = rospy.get_param('~conf', '0.5')

        # torch.cuda.set_per_process_memory_fraction(0.5, 0)
        
        # self.device = torch.device("cpu")
        # load local repository(YoloV5:v6.0)
        self.model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local', force_reload=True)
        # self.model.cpu()

        # which device will be used
        if (rospy.get_param('/use_cpu', 'false')):
            self.model.cpu()
        else:
            self.model.cuda()


        self.model.conf = conf
        self.color_image = Image()
        self.depth_image = Image()
        self.getImageStatus = False

        # Load class color
        self.classes_colors = {}

        # image subscribe
        color_sub = message_filters.Subscriber(image_topic, Image, queue_size=1)

        depth_sub = message_filters.Subscriber(depth_topic, Image, queue_size=1)

        point_cloud_sub = message_filters.Subscriber(pc_topic, PointCloud2, queue_size=1)
        
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, point_cloud_sub], 10, 0.1)
        ts.registerCallback(self.callback_rgbdpc)
        # output publishers
        self.position_pub = rospy.Publisher(
            pub_topic,  BoundingBoxes, queue_size=1)

        self.image_pub = rospy.Publisher(
            '/yolov5/detection_image_new',  Image, queue_size=1)

        self.normals_pub = rospy.Publisher('/normals_pub',  Image, queue_size=1)

        responce_service_counter = rospy.Service('/response_counter', counter_response_crop, self.response_srv_counter)
        responce_service_gauge = rospy.Service('/response_gauge', gauge_response_crop, self.response_srv_gauge)
        responce_service_ssdisplay = rospy.Service('/response_ssdisplay', ssdisplay_response_crop, self.response_srv_ssdisplay)

        # if no image messages
        while (not self.getImageStatus) :
            rospy.loginfo("Waiting for image.")
            rospy.sleep(2)

    def response_srv_counter(self,request):
        self.flag_for_cropping_counter = True
        return counter_response_cropResponse(success = True, counter_id = self.det_counter_id)

    def response_srv_gauge(self,request):
        self.flag_for_cropping_gauge = True
        return gauge_response_cropResponse(success = True, gauge_id = self.det_counter_id)

    def response_srv_ssdisplay(self,request):
        self.flag_for_cropping_ssdisplay = True
        return ssdisplay_response_cropResponse(success = True, ssdisplay_id = self.det_counter_id)


    def callback_rgbdpc(self, color_image, depth_image, pc):
        self.image_processing(color_image)
        self.depth_processing(depth_image)
        self.pc_processing(depth_image, pc)

    def pc_processing(self, depth_image, pc):
        print('pc_processing')   


    def depth_processing(self, depth_image):

        depth_cv = self.cv_brdg.imgmsg_to_cv2(depth_image).copy().astype(np.float32)
        height_depth, width_depth = depth_cv.shape

        # depth_cv /= 1000.0
        # cv2.imshow('depth', depth_cv)

        # zy, zx = np.gradient(depth_cv)  
        # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
        # to reduce noise
        zx = cv2.Sobel(depth_cv, cv2.CV_64F, 1, 0, ksize=5)     
        zy = cv2.Sobel(depth_cv, cv2.CV_64F, 0, 1, ksize=5)

        normal = np.dstack((-zx, -zy, np.ones_like(depth_cv)))
        n = np.linalg.norm(normal, axis=2)
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n

        # offset and rescale values to be in 0-255
        normal += 1
        normal /= 2
        # normal *= 255
        # normal_2 = normal[:, :, ::-1]

        image_64 = (normal[:, :, ::-1] * 255).round().astype(np.uint8)
        # cv2.circle(image_64, (int(self.x_mid), int(self.y_mid)), 5, (255,255,255), -1)
        

        # compute point cloud:
        # FX_DEPTH=387
        # FY_DEPTH=387
        # CX_DEPTH=318
        # CY_DEPTH=246      

        # points_2D = self.points
        # print(points_2D)
        # # print(self.k)
        # # print(self.d)


        # points_3D = np.empty((6,3))
        # for c,i in enumerate(points_2D):
        #     # z_pcd = depth_cv[int(self.y_mid)][int(self.x_mid)]+0.1
        #     z_pcd = depth_cv[int(i[0])][int(i[1])]
        #     x_pcd = (int(i[1]) - CX_DEPTH) * z_pcd / FX_DEPTH
        #     y_pcd = (int(i[0]) - CY_DEPTH) * z_pcd / FY_DEPTH
        #     points_3D[c] = [x_pcd, y_pcd, z_pcd]
        # print(points_3D)
            
        # matrix_camera = np.array(
        #                  [[FX_DEPTH, 0, CX_DEPTH],
        #                  [0, FY_DEPTH, CY_DEPTH],
        #                  [0, 0, 1]], dtype = "double"
        #                  )

        # distortion_coeffs = np.zeros((4,1))
        # success, rotation_vector, translation_vector = cv2.solvePnP(points_3D, points_2D, matrix_camera, distortion_coeffs, flags=0)
        # center_end_point2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, matrix_camera, distortion_coeffs)
        # # print('pose')
        # # print(rotation_vector)
        # # print(translation_vector)
        # point1 = ( int(points_2D[0][0]), int(points_2D[0][1]))
        # point2 = ( int(center_end_point2D[0][0][0]), int(center_end_point2D[0][0][1]))
        # cv2.line(image_64, point1, point2, (255,255,255), 2)
        # cv2.drawFrameAxes(image_64, matrix_camera, distortion_coeffs, rotation_vector, translation_vector, 0.01)
        im = self.cv_brdg.cv2_to_imgmsg(image_64, encoding="8UC3")
        self.normals_pub.publish(im)

        # cv2.imshow('normals', normal_2)
        # cv2.waitKey(0)


    def image_processing(self, image):     
        # tune rate
        # if abs((self.im_rate - image.header.seq))>=1:

        self.getImageStatus = True
        #self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        self.color_image = ros_numpy.numpify(image)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

        results = self.model(self.color_image)
        # xmin    ymin    xmax   ymax  confidence  class(number)  name

        boxs = results.pandas().xyxy[0].sort_values(by='confidence').values
        boxs_ = []
        
        # create ids list (aruco here)
        # ids = np.array([349850, 538746])
        # insert space for id
        for i, n in enumerate(boxs):
            boxes_ = np.append(n, 0)
            boxs_ = np.append(boxs_,boxes_)

        boxs_ = np.reshape(boxs_, (-1, 8))

        gray = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
        arucoDict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        arucoParams = aruco.DetectorParameters_create()
        self.corners, self.aruco_ids, self.rejected = aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)
        
        self.dectshow(self.color_image, boxs_, image.height, image.width, self.corners, self.aruco_ids, self.rejected)
        self.im_rate = image.header.seq
        

    def dectshow(self, org_img, boxs, height, width, corners, aruco_ids, rejected):

        count = 0
        for i in boxs:
            count += 1

        for box in boxs:
            boundingBox = BoundingBox()
            boundingBox.probability =np.float64(box[4])
            boundingBox.xmin = np.int64(box[0])
            boundingBox.ymin = np.int64(box[1])
            boundingBox.xmax = np.int64(box[2])
            boundingBox.ymax = np.int64(box[3])
            boundingBox.num = np.int16(count)
            # self.x_mid = (boundingBox.xmax+boundingBox.xmin)/2
            # self.y_mid = (boundingBox.ymax+boundingBox.ymin)/2

            bX = (boundingBox.xmin + boundingBox.xmax)/2
            bY = (boundingBox.ymin + boundingBox.ymax)/2

            self.points = np.array([(int(bX), int(bY)),
                                    (boundingBox.xmin+70,boundingBox.ymin+70),
                                    (boundingBox.xmax-161,boundingBox.ymax-70),
                                    (boundingBox.xmax-161,boundingBox.ymin+70),
                                    (boundingBox.xmin+70,boundingBox.ymax-70),                                    
                                    (int(bX), int(bY)+10)], dtype="double")
            
            # self.points = np.array([(int(bX), int(bY)),
            #                         (boundingBox.xmin,boundingBox.ymin),
            #                         (boundingBox.xmax,boundingBox.ymax),
            #                         (boundingBox.xmax,boundingBox.ymin),
            #                         (boundingBox.xmin,boundingBox.ymax),                                    
            #                         (int(bX), int(bY)+10)], dtype="double")

            boundingBox.Class = box[6]
            if boundingBox.Class == 'water counter':
                boundingBox.Class = 'counter'
            
            # print('len(corners)')
            # print(len(corners))
            if len(corners) > 0:
            # flatten the ArUco IDs list
                aruco_ids = aruco_ids.flatten()
                # print(aruco_ids)
                # loop over the detected ArUCo corners
                for (markerCorner, markerID) in zip(corners, aruco_ids):
                    # extract the marker corners
                    corners_ = markerCorner.reshape((4, 2))
                    topLeft = corners_[0]
                    topRight = corners_[1]
                    bottomRight = corners_[2]
                    bottomLeft = corners_[3]
                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))
                    # compute and draw the center (x, y)-coordinates of the ArUco
                    # marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    cv2.circle(self.color_image, (cX, cY), 4, (0, 0, 255), -1)

                    distance = math.sqrt(abs((cX - bX)**2 + (cY - bY)**2))
                    # print(distance)
                    if (distance<=500):
                        boundingBox.id = markerID

            
            if box[6] in self.classes_colors.keys():
                color = self.classes_colors[box[6]]
            else:
                color = np.random.randint(0, 183, 3)
                self.classes_colors[box[6]] = color

            cv2.rectangle(org_img, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (int(color[0]),int(color[1]), int(color[2])), 2)


            if box[1] < 20:
                text_pos_y = box[1] + 30
            else:
                text_pos_y = box[1] - 10
                
            cv2.putText(org_img, boundingBox.Class,
                        (int(box[0]), int(text_pos_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(org_img, str(np.round(box[4], 2)),
                        (int(box[0]), int(text_pos_y)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(org_img, 'id '+str(boundingBox.id),
                        (int(box[0]), int(text_pos_y)+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            self.boundingBoxes.bounding_boxes.append(boundingBox)
        
        # print('!!!!!!!!!!!!!!!!!!!!!!!!')
        # print(self.boundingBoxes)
        # print('!!!!!!!!!!!!!!!!!!!!!!!!')
        for i in range(len(self.boundingBoxes.bounding_boxes)):        
            cropped_img = org_img[self.boundingBoxes.bounding_boxes[i].ymin:self.boundingBoxes.bounding_boxes[i].ymax, self.boundingBoxes.bounding_boxes[i].xmin:self.boundingBoxes.bounding_boxes[i].xmax]
            cv2.imwrite('/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/media/meter_cropped'+str(self.boundingBoxes.bounding_boxes[i].id)+'.jpg',cropped_img)
            

        self.boundingBoxes = BoundingBoxes()
        
        # if (count != 0):
        #     rospy.loginfo("Sensor is detected")
        #     rospy.sleep(2)

        # print('boundingBox.Class') 
        # print(boundingBox.Class)

        # print('boundingBox.id')
        # print(boundingBox.id)

        if self.flag_for_cropping_counter:
            if boundingBox.Class=='counter':
                self.det_counter_id.data = str(boundingBox.id)
                print(str(boundingBox.id))                
                cropped_img = org_img[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]
                cv2.imwrite('/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/media/'+str(boundingBox.Class)+'_cropped_'+str(boundingBox.id)+'.jpg',cropped_img)
                rospy.loginfo( '%s is cropped and saved', boundingBox.Class)
                self.flag_for_cropping_counter = False
            else:
                return

        elif self.flag_for_cropping_gauge:
            if boundingBox.Class=='gauge':
                self.det_gauge_id.data = str(boundingBox.id)
                cropped_img = org_img[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]
                cv2.imwrite('/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/media/'+str(boundingBox.Class)+'_cropped_'+str(boundingBox.id)+'.jpg',cropped_img)
                rospy.loginfo( '%s is cropped and saved', boundingBox.Class)
                self.flag_for_cropping_gauge = False
            else:
                return


        elif self.flag_for_cropping_ssdisplay:
            if boundingBox.Class=='ss display':
                self.det_ssdisplay_id.data = str(boundingBox.id)
                cropped_img = org_img[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]
                cv2.imwrite('/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/media/'+str(boundingBox.Class)+'_cropped_'+str(boundingBox.id)+'.jpg',cropped_img)
                rospy.loginfo( '%s is cropped and saved', boundingBox.Class)
                self.flag_for_cropping_ssdisplay = False
            else:
                return

        self.publish_image(org_img, height, width)

    def publish_image(self, imgdata, height, width):
        #image_temp = Image()
        image_temp = ros_numpy.msgify(Image, imgdata, encoding='rgb8')
        header = Header(stamp=rospy.Time.now())
        header.frame_id = 'camera_frame'
        image_temp.header = header
        self.image_pub.publish(image_temp)



def main():
    rospy.init_node('yolov5_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()
    # rate = rospy.Rate(10)
    # while not rospy.is_shutdown():
    #     rate.sleep()


if __name__ == "__main__":

    main()
