#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
# sys.path.insert(0, '/ros_ws/ws-ros1/src/meter_detect_and_read/yolov5_ros/yolov5_ros/yolov5/yolov5')
# print(sys.path)
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
from yolov5_ros.yolov5.models.common import DetectMultiBackend
from yolov5_ros.yolov5.utils.torch_utils import select_device
from yolov5_ros.yolov5.utils.dataloaders import LoadImages
from yolov5_ros.yolov5.utils.general import (non_max_suppression, check_img_size)
from pathlib import Path
# (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
#                            increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
#                            strip_optimizer, xyxy2xywh)

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
    seen = 0

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
        
        device = torch.device("cuda")
        # load local repository(YoloV5:v6.0)
        start_load_model = time.time()
        # device = select_device('0')
        # print('device')
        # print(device)

        weights_seg = '/ros_ws/ws-ros1/src/meter_detect_and_read/yolov5_ros/yolov5_ros/yolov5/7_seg.pt'
        data = '/ros_ws/ws-ros1/src/meter_detect_and_read/yolov5_ros/yolov5_ros/yolov5/data/coco128.yaml' 
        self.model_seg = DetectMultiBackend(weights_seg, device = device, dnn=False, data=data, fp16=False )
        self.model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local', force_reload=True)
        self.model.to(device)
        self.model_digits = torch.hub.load(yolov5_path, 'custom', '/ros_ws/ws-ros1/src/meter_detect_and_read/yolov5_ros/yolov5_ros/yolov5/43.pt', source='local', force_reload=True)
        self.model_digits.to(device)
        # self.model_seg = torch.hub.load(yolov5_path, 'custom', '/ros_ws/ws-ros1/src/meter_detect_and_read/yolov5_ros/yolov5_ros/yolov5/7_seg.pt', source='local', force_reload=True)
        
        
        end_load_model = time.time()
        print('Models are loaded')
        print(end_load_model-start_load_model)
        # self.model.cpu()

        # which device will be used
        # if (rospy.get_param('/use_cpu', 'false')):
        #     self.model.cpu()
        # else:
        #     self.model.cuda()
        #     self.model_digits.cuda()
        #     # self.model_seg.cuda()

        

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
        self.msg_ = image

        

    def model_eval(self):
        self.color_image = ros_numpy.numpify(self.msg_)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        results = self.model(self.color_image)
        # print(results.pandas().xyxy[0])
        boxs = results.pandas().xyxy[0].sort_values(by='confidence').values

        ############ for aruco ############
        # boxs_ = []
        
        # # create ids list (aruco here)
        # # ids = np.array([349850, 538746])
        # # insert space for id
        # for i, n in enumerate(boxs):
        #     boxes_ = np.append(n, 0)
        #     boxs_ = np.append(boxs_,boxes_)

        # boxs_ = np.reshape(boxs_, (-1, 8))
           
        self.dectshow(self.color_image, boxs, self.msg_.height, self.msg_.width)
        self.im_rate = self.msg_.header.seq
        return results

    def model_digits_eval(self):
        self.color_image = ros_numpy.numpify(self.msg_)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        cropped_msg = self.color_image[self.bboxes[1]:self.bboxes[3],self.bboxes[0]:self.bboxes[2]]
        # cv2.imwrite(os.path.join(path, 'cropped_msg_0.jpg'), cropped_msg)
        results_dig = self.model_digits(cropped_msg)
        boxs_dig = results_dig.pandas().xyxy[0].sort_values(by='confidence').values
           
        # self.dectshow(self.color_image, boxs_dig, self.msg_.height, self.msg_.width)
        # self.im_rate = self.msg_.header.seq
        return results_dig


    def model_seg_eval(self):
        self.color_image = ros_numpy.numpify(self.msg_)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        cropped_msg = self.color_image[self.bboxes[1]:self.bboxes[3],self.bboxes[0]:self.bboxes[2]]
        cv2.imwrite(os.path.join(path, 'cropped_msg_0.jpg'), self.color_image )
        stride, names, pt = self.model_seg.stride, self.model_seg.names, self.model_seg.pt
        vid_stride = 1
        imgsz=(640, 640)
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        bs = 1
        source = os.path.join(path, 'cropped_msg_0.jpg')
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
        vid_path, vid_writer = [None] * bs, [None] * bs

        for path_, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(self.model_seg.device)
            im = im.half() if self.model_seg.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
                pred, proto = self.model_seg(im)[:2]



        y = np.expand_dims(self.color_image, axis=0) 
        b = np.transpose(y, (0, 3, 1, 2))
        t = torch.from_numpy(b).cuda().float()
        # pred, proto = self.model_seg(y)[:2]

        for i, det in enumerate(pred):  # per image
            self.seen += 1
            agnostic_nms=False
            max_det=1000
            pred = non_max_suppression(pred, self.model.conf, 0.45, 2, agnostic_nms, max_det, nm=32)
            p, im0, frame = path_, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            print('Path')
            print(p)
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # s += '%gx%g ' % im.shape[2:]  # print string
            # imc = im0.copy() if save_crop else im0  # for save_crop
            # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # if len(det):
            #     masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
            #     det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

            #     # Segments
            #     if save_txt:
            #         segments = reversed(masks2segments(masks))
            #         segments = [scale_segments(im.shape[2:], x, im0.shape).round() for x in segments]

            #     # Print results
            #     for c in det[:, 5].unique():
            #         n = (det[:, 5] == c).sum()  # detections per class
            #         s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            #     # Mask plotting
            #     annotator.masks(masks,
            #                     colors=[colors(x, True) for x in det[:, 5]],
            #                     im_gpu=None if retina_masks else im[i])

            #     # Write results
            #     for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
            #         if save_txt:  # Write to file
            #             segj = segments[j].reshape(-1)  # (n,2) to (n*2)
            #             line = (cls, *segj, conf) if save_conf else (cls, *segj)  # label format
            #             with open(f'{txt_path}.txt', 'a') as f:
            #                 f.write(('%g ' * len(line)).rstrip() % line + '\n')

            #             line_bbox = (cls, *xyxy, conf) if save_conf else (cls, *xyxy)  # label format
            #             with open(txt_path + '_bbox.txt', 'a') as f:
            #                 f.write(('%g ' * len(line_bbox)).rstrip() % line_bbox + '\n')

            #         if save_img or save_crop or view_img:  # Add bbox to image
            #             c = int(cls)  # integer class
            #             label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
            #             annotator.box_label(xyxy, label, color=colors(c, True))
            #             # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
            #         if save_crop:
            #             save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
        # print('pred')
        # print(pred)
        # print('proto')
        # print(proto)
        # boxs_seg = results_seg.pandas().xyxy[0].sort_values(by='confidence').values
           
        # self.dectshow(self.color_image, boxs_dig, self.msg_.height, self.msg_.width)
        # self.im_rate = self.msg_.header.seq
        return pred


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
        yolo_dect.model_seg_eval()
        rate.sleep()


if __name__ == "__main__":

    main()
