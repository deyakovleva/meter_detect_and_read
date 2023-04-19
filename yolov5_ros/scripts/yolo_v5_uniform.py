#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from cv2 import aruco
import torch
import math
import random

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

    im_rate = 0
    det_counter_id = String()
    det_gauge_id = String()
    det_ssdisplay_id = String()
    boundingBoxes = BoundingBoxes()

    def __init__(self):

        # load parameters
        yolov5_path = rospy.get_param('/yolov5_path', '/home/itmo/yolov5_ws/src/yolov5_ros_wmeters/yolov5_ros/yolov5_ros/yolov5')

        weight_path = rospy.get_param('~weight_path', '/home/itmo/yolov5_ws/src/yolov5_ros_wmeters/yolov5_ros/yolov5_ros/weights/meters_weights.pt')
        image_topic = rospy.get_param(
            '~image_topic', '/camera/color/image_raw')
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

        # if no image messages
        while (not self.getImageStatus) :
            rospy.loginfo("Waiting for image.")
            rospy.sleep(2)


    def image_callback(self, image):

        # tune rate
        # if abs((self.im_rate - image.header.seq))>=1:

        self.getImageStatus = True
            #self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        self.color_image = ros_numpy.numpify(image)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

        results = self.model(self.color_image)
        # xmin    ymin    xmax   ymax  confidence  class(number)  name

        boxs = results.pandas().xyxy[0].sort_values(by='confidence').values
        # boxs_ = []
        
        # create ids list (aruco here)
        # ids = np.array([349850, 538746])
        # insert space for id
        # for i, n in enumerate(boxs):
        #     boxes_ = np.append(n, 0)
        #     boxs_ = np.append(boxs_,boxes_)

        # boxs_ = np.reshape(boxs_, (-1, 8))
            
        self.dectshow(self.color_image, boxs, image.height, image.width)
        self.im_rate = image.header.seq
       

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
            boundingBox.Class = box[6]
            
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
            
            self.boundingBoxes.bounding_boxes.append(boundingBox)

        
        # print('boundingBox.Class') 
        # print(boundingBox.Class)
        # print('start cycle') 
        classes = []
        for i in self.boundingBoxes.bounding_boxes:
            # print('self.boundingBoxes.bounding_boxes.Class') 
            classes.append(i.Class)
        # print('classes')
        # print(classes)

        w_attendence = 'worker' # worker
        matched_indexes_w = []
        i_w = 0
        
        h_attendence = 'helmet' # helmet
        matched_indexes_h = []
        i_h = 0

        v_attendence = 'vest' # vest
        matched_indexes_v = []
        i_v = 0

        while i_w < len(classes):
            if w_attendence == classes[i_w]:
                matched_indexes_w.append(i_w)
            i_w += 1
        # print(f'{w_attendence} is present in {classes} at indexes {matched_indexes_w}')

        while i_h < len(classes):
            if h_attendence == classes[i_h]:
                matched_indexes_h.append(i_h)
            i_h += 1
        # print(f'{h_attendence} is present in {classes} at indexes {matched_indexes_h}')

        while i_v < len(classes):
            if v_attendence == classes[i_v]:
                matched_indexes_v.append(i_v)
            i_v += 1
        # print(f'{v_attendence} is present in {classes} at indexes {matched_indexes_v}')


        # wc = self.boundingBoxes.bounding_boxes[i_wc].Class

        if (matched_indexes_w and matched_indexes_h and matched_indexes_v):

            w_X = (self.boundingBoxes.bounding_boxes[matched_indexes_w[0]].xmin + self.boundingBoxes.bounding_boxes[matched_indexes_w[0]].xmax)/2
            w_Y = (self.boundingBoxes.bounding_boxes[matched_indexes_w[0]].ymin + self.boundingBoxes.bounding_boxes[matched_indexes_w[0]].ymax)/2

            h_X = (self.boundingBoxes.bounding_boxes[matched_indexes_h[0]].xmin + self.boundingBoxes.bounding_boxes[matched_indexes_h[0]].xmax)/2
            h_Y = (self.boundingBoxes.bounding_boxes[matched_indexes_h[0]].ymin + self.boundingBoxes.bounding_boxes[matched_indexes_h[0]].ymax)/2
            
            v_X = (self.boundingBoxes.bounding_boxes[matched_indexes_v[0]].xmin + self.boundingBoxes.bounding_boxes[matched_indexes_v[0]].xmax)/2
            v_Y = (self.boundingBoxes.bounding_boxes[matched_indexes_v[0]].ymin + self.boundingBoxes.bounding_boxes[matched_indexes_v[0]].ymax)/2

            dist_btw_w_h = math.sqrt(abs((w_X - h_X)**2 + (w_Y - h_Y)**2))
            dist_btw_w_v = math.sqrt(abs((w_X - v_X)**2 + (w_Y - v_Y)**2))
            print(dist_btw_w_h)
            print(dist_btw_w_v)

            if dist_btw_w_h<=400:
                print('helmet is on worker') 
            else:
                print('!put on helmet!')

            if dist_btw_w_v<=300:
                print('vest is on worker') 
            else:
                print('!put on vest!')
        elif (matched_indexes_w and matched_indexes_h and not(matched_indexes_v)):
            print('!!!!!put on vest!!!!!')
            w_X = (self.boundingBoxes.bounding_boxes[matched_indexes_w[0]].xmin + self.boundingBoxes.bounding_boxes[matched_indexes_w[0]].xmax)/2
            w_Y = (self.boundingBoxes.bounding_boxes[matched_indexes_w[0]].ymin + self.boundingBoxes.bounding_boxes[matched_indexes_w[0]].ymax)/2

            h_X = (self.boundingBoxes.bounding_boxes[matched_indexes_h[0]].xmin + self.boundingBoxes.bounding_boxes[matched_indexes_h[0]].xmax)/2
            h_Y = (self.boundingBoxes.bounding_boxes[matched_indexes_h[0]].ymin + self.boundingBoxes.bounding_boxes[matched_indexes_h[0]].ymax)/2
            
            dist_btw_w_h = math.sqrt(abs((w_X - h_X)**2 + (w_Y - h_Y)**2))
            print(dist_btw_w_h)
            print(h_Y)
            print(w_Y)

            if ((dist_btw_w_h<=400) and (h_Y<w_Y)):
                print('helmet is on worker') 
            else:
                print('!put on helmet!')

        elif (matched_indexes_w and matched_indexes_v and not(matched_indexes_h)):
            print('!!!!!put on helmet!!!!!')
            w_X = (self.boundingBoxes.bounding_boxes[matched_indexes_w[0]].xmin + self.boundingBoxes.bounding_boxes[matched_indexes_w[0]].xmax)/2
            w_Y = (self.boundingBoxes.bounding_boxes[matched_indexes_w[0]].ymin + self.boundingBoxes.bounding_boxes[matched_indexes_w[0]].ymax)/2

            v_X = (self.boundingBoxes.bounding_boxes[matched_indexes_v[0]].xmin + self.boundingBoxes.bounding_boxes[matched_indexes_v[0]].xmax)/2
            v_Y = (self.boundingBoxes.bounding_boxes[matched_indexes_v[0]].ymin + self.boundingBoxes.bounding_boxes[matched_indexes_v[0]].ymax)/2

            dist_btw_w_v = math.sqrt(abs((w_X - v_X)**2 + (w_Y - v_Y)**2))
            print(dist_btw_w_v)

            if dist_btw_w_v<=300:
                print('vest is on worker') 
            else:
                print('!put on vest!')
        elif (matched_indexes_w and not(matched_indexes_v) and not(matched_indexes_h)):
            print('!!!!!put on helmet and vest!!!!!')
        elif not(matched_indexes_w):
            print('worker is not found')
        else:
            print('something is not found(')

        self.boundingBoxes = BoundingBoxes()


        
        # if (count != 0):
        #     rospy.loginfo("Sensor is detected")
        #     rospy.sleep(2)

        


        # if self.flag_for_cropping_counter:
        #     if boundingBox.Class=='counter':
        #         self.det_counter_id.data = str(boundingBox.id)
        #         print(str(boundingBox.id))                
        #         cropped_img = org_img[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]
        #         cv2.imwrite('/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/media/'+str(boundingBox.Class)+'_cropped_'+str(boundingBox.id)+'.jpg',cropped_img)
        #         rospy.loginfo( '%s is cropped and saved', boundingBox.Class)
        #         self.flag_for_cropping_counter = False
        #     else:
        #         return

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
