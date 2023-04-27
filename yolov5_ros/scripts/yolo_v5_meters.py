#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from cv2 import aruco
import torch
import math
import random
import time

# gpu = torch.device('cuda')
# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# print(torch.cuda.list_gpu_processes())
# torch.cuda.set_per_process_memory_fraction(0.1, 0)
# print(torch.cuda.list_gpu_processes())
import rospy
import numpy as np
import ros_numpy
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image
from yolov5_ros_msgs.msg import BoundingBox, BoundingBoxes
from yolov5_ros_msgs.srv import counter_response_crop, counter_response_cropResponse, gauge_response_crop, gauge_response_cropResponse, ssdisplay_response_crop, ssdisplay_response_cropResponse


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

    def __init__(self):

        # load parameters
        yolov5_path = rospy.get_param('/yolov5_path', '/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolov5')

        weight_path = rospy.get_param('~weight_path', '/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/weights/43.pt')
        image_topic = rospy.get_param(
            '~image_topic', '/camera/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov5/BoundingBoxes')
        # self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_frame')
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
        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback,
                                          queue_size=1)

        # output publishers
        self.position_pub = rospy.Publisher(
            pub_topic,  BoundingBoxes, queue_size=1)

        self.image_pub = rospy.Publisher(
            '/yolov5/detection_image_new',  Image, queue_size=1)

        responce_service_counter = rospy.Service('/crop_counter', counter_response_crop, self.response_srv_counter)
        responce_service_gauge = rospy.Service('/crop_gauge', gauge_response_crop, self.response_srv_gauge)
        responce_service_ssdisplay = rospy.Service('/crop_ssdisplay', ssdisplay_response_crop, self.response_srv_ssdisplay)

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

    def image_callback(self, image):

        # tune rate
        # if abs((self.im_rate - image.header.seq))>=1:

        self.getImageStatus = True
            #self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        self.color_image = ros_numpy.numpify(image)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        # start_model = time.time()
        # results = self.model(self.color_image)
        results = self.model_eval(self.color_image)

        # xmin    ymin    xmax   ymax  confidence  class(number)  name

        # end_model = time.time()
        # print('model time')
        # print(end_model-start_model)

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
        FX_DEPTH=615.85
        FY_DEPTH=615.97
        CX_DEPTH=331.58
        CY_DEPTH=237.13 
        matrix_camera = np.array(
                                [[FX_DEPTH, 0, CX_DEPTH],
                                [0, FY_DEPTH, CY_DEPTH],
                                [0, 0, 1]], dtype = "double"
                                )

        distortion_coeffs = np.array([[ 0.14815903],[-0.37939446],[-0.37939446],[-0.00197926], [-0.00197926]])
        if len(self.corners) > 0:
            for i in range(0, len(self.aruco_ids)):
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(self.corners[i], 0.02, matrix_camera, distortion_coeffs)
                # print(self.corners[i])
                # Draw a square around the markers
                aruco.drawDetectedMarkers(self.color_image, self.corners) 
                print('!!!!!!!!')
                print(rvec)
                print(tvec)
                # Draw Axis
                aruco.drawAxis(self.color_image, matrix_camera, distortion_coeffs, rvec, tvec, 0.01) 
                # print(rvec)
                # print(tvec)    
        self.dectshow(self.color_image, boxs_, image.height, image.width, self.corners, self.aruco_ids, self.rejected)
        self.im_rate = image.header.seq


        
        # verify *at least* one ArUco marker was detected
        # if len(corners) > 0:
        #     # flatten the ArUco IDs list
        #     aruco_ids = aruco_ids.flatten()
        #     # loop over the detected ArUCo corners
        #     for (markerCorner, markerID) in zip(corners, aruco_ids):
        #         # extract the marker corners
        #         corners_ = markerCorner.reshape((4, 2))
        #         topLeft = corners_[0]
        #         topRight = corners_[1]
        #         bottomRight = corners_[2]
        #         bottomLeft = corners_[3]
        #         # convert each of the (x, y)-coordinate pairs to integers
        #         topRight = (int(topRight[0]), int(topRight[1]))
        #         bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        #         bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        #         topLeft = (int(topLeft[0]), int(topLeft[1]))
        #         # draw the bounding box of the ArUCo detection
        #         # cv2.line(self.color_image, topLeft, topRight, (0, 255, 0), 2)
        #         # cv2.line(self.color_image, topRight, bottomRight, (0, 255, 0), 2)
        #         # cv2.line(self.color_image, bottomRight, bottomLeft, (0, 255, 0), 2)
        #         # cv2.line(self.color_image, bottomLeft, topLeft, (0, 255, 0), 2)
        #         # compute and draw the center (x, y)-coordinates of the ArUco
        #         # marker
        #         cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        #         cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        #         cv2.circle(self.color_image, (cX, cY), 4, (0, 0, 255), -1)
        #         # draw the ArUco marker ID on the image
        #         cv2.putText(self.color_image, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #         print("[INFO] ArUco marker ID: {}".format(markerID))
        #         # show the output image
        #         cv2.imshow('result', self.color_image)
        #         cv2.waitKey(0)
        

    def model_eval(self, color_image):
        results = self.model(color_image)
        return results

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

            bX = (boundingBox.xmin + boundingBox.xmax)/2
            bY = (boundingBox.ymin + boundingBox.ymax)/2
            
            boundingBox.Class = box[6]
            if boundingBox.Class == 'water counter':
                boundingBox.Class = 'gauge'
            
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
                    if (distance<=300):
                        boundingBox.id = markerID

            
            if box[6] in self.classes_colors.keys():
                color = self.classes_colors[box[6]]
            else:
                color = np.random.randint(0, 183, 3)
                self.classes_colors[box[6]] = color

            cv2.rectangle(org_img, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (int(color[0]),int(color[1]), int(color[2])), 2)
            # cv2.rectangle(org_img, (int(box[0])+70, int(box[1])+70),
            #               (int(box[2]-70), int(box[3])-70), (int(color[0]),int(color[1]), int(color[2])), 2)
                          
            if box[1] < 20:
                text_pos_y = box[1] + 30
            else:
                text_pos_y = box[1] - 10
                
            cv2.putText(org_img, boundingBox.Class,
                        (int(box[0]), int(text_pos_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(org_img, str(np.round(box[4], 2)),
                        (int(box[0]), int(text_pos_y)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(org_img, 'id '+str(boundingBox.id),
            #             (int(box[0]), int(text_pos_y)+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
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

        if self.flag_for_cropping_gauge:
            if boundingBox.Class=='gauge':
                self.det_gauge_id.data = str(boundingBox.id)
                cropped_img = org_img[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB ) 
                cv2.imwrite('/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/media/'+str(boundingBox.Class)+'_cropped_'+str(boundingBox.id)+'.jpg',cropped_img)
                # cv2.imshow('Cropped gauge', cropped_img)
                
                rospy.loginfo( '%s is cropped and saved', boundingBox.Class)
                # cv2.waitKey(0)
                self.flag_for_cropping_gauge = False
            else:
                return

        elif self.flag_for_cropping_counter:
            if boundingBox.Class=='counter':
                self.det_counter_id.data = str(boundingBox.id)
                print(str(boundingBox.id))                
                cropped_img = org_img[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]
                cv2.imwrite('/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/media/'+str(boundingBox.Class)+'_cropped_'+str(boundingBox.id)+'.jpg',cropped_img)
                # cv2.imshow('Cropped counter', cropped_img)
                rospy.loginfo( '%s is cropped and saved', boundingBox.Class)
                # cv2.waitKey(0)
                self.flag_for_cropping_counter = False
            else:
                return


        elif self.flag_for_cropping_ssdisplay:
            if boundingBox.Class=='ss display':
                self.det_ssdisplay_id.data = str(boundingBox.id)
                cropped_img = org_img[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]
                cv2.imwrite('/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/media/'+str(boundingBox.Class)+'_cropped_'+str(boundingBox.id)+'.jpg',cropped_img)
                
                # cv2.imshow('Cropped ssdisplay', cropped_img)
                rospy.loginfo( '%s is cropped and saved', boundingBox.Class)
                # cv2.waitKey(0)
                self.flag_for_cropping_ssdisplay = False
                
            else:
                return

        self.publish_image(org_img, height, width)

    def publish_image(self, imgdata, height, width):
        # image_temp = Image()
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
