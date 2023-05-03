#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from cv2 import aruco
import torch
import math
import random
import time
import rospy
import numpy as np
import ros_numpy
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image
from yolov5_ros_msgs.msg import BoundingBox, BoundingBoxes
from yolov5_ros_msgs.srv import counter_response_crop, counter_response_cropResponse, gauge_response_crop, gauge_response_cropResponse, ssdisplay_response_crop, ssdisplay_response_cropResponse
import os 

path = '/ros_ws/ws-ros1/src/meter_detect_and_read/'
class Yolo_Dect:

    flag_for_cropping_counter = False
    flag_for_cropping_gauge = False
    flag_for_cropping_ssdisplay = False
    im_rate = 0
    det_counter_id = String()
    det_gauge_id = String()
    det_ssdisplay_id = String()
    boundingBoxes = BoundingBoxes()

    def __init__(self):

        self.msg_ = None
        self.bboxes = []

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
        start_load_model = time.time()
        self.model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local', force_reload=True)
        self.model_digits = torch.hub.load(yolov5_path, 'custom', '/ros_ws/ws-ros1/src/meter_detect_and_read/yolov5_ros/yolov5_ros/yolov5_digits/43.pt', source='local', force_reload=True)
        end_load_model = time.time()
        print('Models are loaded')
        print(end_load_model-start_load_model)
        # self.model.cpu()

        # which device will be used
        if (rospy.get_param('/use_cpu', 'false')):
            self.model.cpu()
        else:
            self.model.cuda()
            self.model_digits.cuda()

        

        self.model.conf = conf
        self.model_digits.conf = conf
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
        print('gauge change flag')
        return gauge_response_cropResponse(success = True, gauge_id = self.det_counter_id)

    def response_srv_ssdisplay(self,request):
        self.flag_for_cropping_ssdisplay = True
        return ssdisplay_response_cropResponse(success = True, ssdisplay_id = self.det_counter_id)

    def image_callback(self, image):

        # tune rate
        # if abs((self.im_rate - image.header.seq))>=1:

        self.getImageStatus = True
            #self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        # self.color_image = ros_numpy.numpify(image)
        # self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        # start_model = time.time()
        # results = self.model(self.color_image)
        # results = self.model_eval(self.color_image)
        self.msg_ = image

        # xmin    ymin    xmax   ymax  confidence  class(number)  name

        # end_model = time.time()
        # print('model time')
        # print(end_model-start_model)

        # boxs = results.pandas().xyxy[0].sort_values(by='confidence').values
        # boxs_ = []
        
        # # create ids list (aruco here)
        # # ids = np.array([349850, 538746])
        # # insert space for id
        # for i, n in enumerate(boxs):
        #     boxes_ = np.append(n, 0)
        #     boxs_ = np.append(boxs_,boxes_)

        # boxs_ = np.reshape(boxs_, (-1, 8))

           
        # self.dectshow(self.color_image, boxs_, image.height, image.width)
        # self.im_rate = image.header.seq
        

    def model_eval(self):
        self.color_image = ros_numpy.numpify(self.msg_)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        results = self.model(self.color_image)
        # print(results.pandas().xyxy[0])
        boxs = results.pandas().xyxy[0].sort_values(by='confidence').values
           
        self.dectshow(self.color_image, boxs, self.msg_.height, self.msg_.width)
        self.im_rate = self.msg_.header.seq
        return results

    def model_digits_eval(self):
        self.color_image = ros_numpy.numpify(self.msg_)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        cropped_msg = self.color_image[self.bboxes[1]:self.bboxes[3],self.bboxes[0]:self.bboxes[2]]
        # cv2.imwrite(os.path.join(path, 'cropped_msg_0.jpg'), cropped_msg)
        results_dig = self.model_digits(cropped_msg)
        print(results_dig.pandas().xyxy[0])
        boxs_dig = results_dig.pandas().xyxy[0].sort_values(by='confidence').values
           
        # self.dectshow(self.color_image, boxs_dig, self.msg_.height, self.msg_.width)
        # self.im_rate = self.msg_.header.seq
        return results_dig

    def dectshow(self, org_img, boxs, height, width):

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
                boundingBox.Class = 'counter'
            
            boundingBox.id = 0

            self.bboxes = [boundingBox.xmin, boundingBox.ymin, boundingBox.xmax, boundingBox.ymax, boundingBox.Class]

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
            cv2.imwrite(os.path.join(path, 'yolov5_ros/yolov5_ros/media/meter_cropped'+str(self.boundingBoxes.bounding_boxes[i].id)+'.jpg') ,cropped_img)
        self.boundingBoxes = BoundingBoxes()
        

        # print(self.flag_for_cropping_gauge)

        if self.flag_for_cropping_gauge:
            if boundingBox.Class=='gauge':
                self.det_gauge_id.data = str(boundingBox.id)
                print('got in if')
                cropped_img = org_img[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB ) 
                cv2.imwrite(os.path.join(path, 'yolov5_ros/yolov5_ros/media/'+str(boundingBox.Class)+'_cropped_'+str(boundingBox.id)+'.jpg'),cropped_img)
                # cv2.imshow('Cropped gauge', cropped_img)
                
                rospy.loginfo( '%s is cropped and saved', boundingBox.Class)
                # cv2.waitKey(0)
                self.flag_for_cropping_gauge = False
            # else:
            #     return

        elif self.flag_for_cropping_counter:
            if boundingBox.Class=='counter':
                self.det_counter_id.data = str(boundingBox.id)
                print(str(boundingBox.id))                
                cropped_img = org_img[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]
                cv2.imwrite(os.path.join(path, 'yolov5_ros/yolov5_ros/media/'+str(boundingBox.Class)+'_cropped_'+str(boundingBox.id)+'.jpg'),cropped_img)                # cv2.imshow('Cropped counter', cropped_img)
                rospy.loginfo( '%s is cropped and saved', boundingBox.Class)
                # cv2.waitKey(0)
                self.flag_for_cropping_counter = False
            # else:
            #     return


        elif self.flag_for_cropping_ssdisplay:
            if boundingBox.Class=='ss display':
                self.det_ssdisplay_id.data = str(boundingBox.id)
                cropped_img = org_img[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]
                cv2.imwrite(os.path.join(path, 'yolov5_ros/yolov5_ros/media/'+str(boundingBox.Class)+'_cropped_'+str(boundingBox.id)+'.jpg'),cropped_img)                
                # cv2.imshow('Cropped ssdisplay', cropped_img)
                rospy.loginfo( '%s is cropped and saved', boundingBox.Class)
                # cv2.waitKey(0)
                self.flag_for_cropping_ssdisplay = False
                
            # else:
            #     return

        self.publish_image(org_img, height, width)

    def publish_image(self, imgdata, height, width):
        # image_temp = Image()
        image_temp = ros_numpy.msgify(Image, imgdata, encoding='bgr8')
        header = Header(stamp=rospy.Time.now())
        header.frame_id = 'camera_frame'
        image_temp.header = header
        self.image_pub.publish(image_temp)        


def main():
    rospy.init_node('yolov5_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    # yolo_dect.model_eval()
    # rospy.spin()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        yolo_dect.model_eval()
        yolo_dect.model_digits_eval()
        rate.sleep()


if __name__ == "__main__":

    main()
