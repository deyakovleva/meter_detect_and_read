#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

height = 720
width = 1280
writer = cv2.VideoWriter('aruco_gauge.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (width,height))
bridge = CvBridge()

def video_callback(msg : Image):
    cv_image = bridge.imgmsg_to_cv2(msg)
    cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    writer.write(cv_image_rgb)



def main():



    rospy.init_node('yolov5_ros', anonymous=True)
    # cap = cv2.VideoCapture('b3.mp4')

    video_sub = rospy.Subscriber('/yolov5/detection_image_new', Image, video_callback, queue_size=1000)

    rospy.spin()


if __name__ == "__main__":

    main()
