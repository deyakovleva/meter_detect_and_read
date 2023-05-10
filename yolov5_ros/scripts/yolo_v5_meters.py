#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
start_load_libs = time.time()
import cv2
import torch
import math
import rospy
import numpy as np
import ros_numpy
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image
from yolov5_ros_msgs.msg import BoundingBox, BoundingBoxes
from yolov5_ros_msgs.srv import counter_response_crop, counter_response_cropResponse, gauge_response_crop, gauge_response_cropResponse, ssdisplay_response_crop, ssdisplay_response_cropResponse
import os 
from yolov5_ros.yolov5.models.common import DetectMultiBackend
from pathlib import Path
# from yolov5_ros.yolov5.utils.plots import Annotator
from yolov5_ros.yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5_ros.yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)
from yolov5_ros.yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5_ros.yolov5.utils.segment.general import masks2segments, process_mask
from yolov5_ros.yolov5.utils.torch_utils import select_device, smart_inference_mode
import platform 
from sklearn import linear_model, datasets
import argparse
end_load_libs = time.time()
print(f"Elapsed time to load libs {end_load_libs-start_load_libs}")


class Yolo_Dect:

    flag_for_cropping_counter = False
    flag_for_cropping_gauge = False
    flag_for_cropping_ssdisplay = False
    flag_for_digital_meter_reading = False
    im_rate = 0
    det_counter_id = String()
    det_gauge_id = String()
    det_ssdisplay_id = String()
    boundingBoxes = BoundingBoxes()
    seen = 0
    path_ = '/ros_ws/ws-ros1/src/meter_detect_and_read/'

    def __init__(self):

        self.msg_ = None
        self.bboxes = {}

        # load parameters
        yolov5_path = rospy.get_param('/yolov5_path', '/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolov5')

        weight_path = rospy.get_param('~weight_path', '/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/weights/meters_gauges_ssdisplay.pt')
        image_topic = rospy.get_param(
            '~image_topic', '/camera/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov5/BoundingBoxes')
        # self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_frame')
        conf = rospy.get_param('~conf', '0.5')

        device = torch.device("cuda")
        # load local repository(YoloV5:v6.0)
        start_load_model = time.time()

        weights_seg = '/ros_ws/ws-ros1/src/meter_detect_and_read/yolov5_ros/yolov5_ros/weights/7_seg.pt'
        data = '/ros_ws/ws-ros1/src/meter_detect_and_read/yolov5_ros/yolov5_ros/yolov5/data/coco128.yaml' 
        self.model_seg = DetectMultiBackend(weights_seg, device = device, dnn=False, data=data, fp16=False )
        self.model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local', force_reload=True)
        self.model.to(device)
        self.model_digits_gauge = torch.hub.load(yolov5_path, 'custom', '/ros_ws/ws-ros1/src/meter_detect_and_read/yolov5_ros/yolov5_ros/weights/43.pt', source='local', force_reload=True)
        self.model_digits_gauge.to(device)
        self.model_digits_counter = torch.hub.load(yolov5_path, 'custom', '/ros_ws/ws-ros1/src/meter_detect_and_read/yolov5_ros/yolov5_ros/weights/19.pt', source='local', force_reload=True)
        self.model_digits_counter.to(device)
        self.model_digits_ssdisplay = torch.hub.load(yolov5_path, 'custom', '/ros_ws/ws-ros1/src/meter_detect_and_read/yolov5_ros/yolov5_ros/weights/38.pt', source='local', force_reload=True)
        self.model_digits_ssdisplay.to(device)
        
        end_load_model = time.time()
        print(f"Elapsed time to load models {end_load_model-start_load_model}")
        # self.model.cpu()

        # which device will be used
        # if (rospy.get_param('/use_cpu', 'false')):
        #     self.model.cpu()
        # else:
        #     self.model.cuda()
        #     self.model_digits.cuda()
        #     # self.model_seg.cuda()

        

        self.model.conf = conf
        self.model_digits_gauge.conf = conf
        self.model_digits_counter.conf = conf
        self.model_digits_ssdisplay.conf = 0.2
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
        self.measurement_image_pub = rospy.Publisher(
            '/yolov5/gauge_measuremet_image_new',  Image, queue_size=1)
        self.counter_image_pub = rospy.Publisher(
            '/yolov5/counter_measuremet_image_new',  Image, queue_size=1)
        self.ssdisplay_image_pub = rospy.Publisher(
            '/yolov5/ssdisplay_measuremet_image_new',  Image, queue_size=1)

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

        
    # obj detection func
    def model_eval(self):
        # start_det = time.time()
        self.color_image = ros_numpy.numpify(self.msg_)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        results = self.model(self.color_image)

        return results

    # digits in mano detection func
    def model_digits_gauge_eval(self):
        self.color_image = ros_numpy.numpify(self.msg_)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        g_box = self.bboxes['gauge']
        self.cropped_msg = self.color_image[g_box[1]:g_box[3],g_box[0]:g_box[2]]
        cv2.imwrite(os.path.join(self.path_, 'cropped_msg_0.jpg'), self.cropped_msg)
        results_dig = self.model_digits_gauge(self.cropped_msg)
        boxs_dig = results_dig.pandas().xyxy[0].sort_values(by='confidence').values

        return boxs_dig

    # digits in counter detection func
    def model_digits_counter_eval(self):
        self.color_image = ros_numpy.numpify(self.msg_)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        c_box = self.bboxes['counter']
        self.cropped_msg = self.color_image[c_box[1]:c_box[3],c_box[0]:c_box[2]]
        cv2.imwrite(os.path.join(self.path_, 'cropped_msg_0.jpg'), self.cropped_msg)
        results_dig = self.model_digits_counter(self.cropped_msg)
        boxs_dig = results_dig.pandas().xyxy[0].sort_values(by='xmin').values
        self.flag_for_digital_meter_reading = True
        self.dectshow(self.cropped_msg, boxs_dig, self.cropped_msg.shape[0], self.cropped_msg.shape[1], self.counter_image_pub)

        return boxs_dig

    # digits in counter detection func
    def model_digits_ssdisplay_eval(self):
        self.color_image = ros_numpy.numpify(self.msg_)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        c_box = self.bboxes['ss display']
        self.cropped_msg = self.color_image[c_box[1]:c_box[3],c_box[0]:c_box[2]]
        cv2.imwrite(os.path.join(self.path_, 'cropped_msg_0.jpg'), self.cropped_msg)
        results_dig = self.model_digits_ssdisplay(self.cropped_msg)
        boxs_dig = results_dig.pandas().xyxy[0].sort_values(by='xmin').values
        self.flag_for_digital_meter_reading = True
        self.dectshow(self.cropped_msg, boxs_dig, self.cropped_msg.shape[0], self.cropped_msg.shape[1], self.ssdisplay_image_pub)

        return boxs_dig

    # segmentation func
    def model_seg_eval(self):
        # self.color_image = ros_numpy.numpify(self.msg_)
        # self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        # cropped_msg = self.color_image[self.bboxes[1]:self.bboxes[3],self.bboxes[0]:self.bboxes[2]]
        # cv2.imwrite(os.path.join(self.path_, 'cropped_msg_0.jpg'), cropped_msg )

        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[1]  # YOLOv5 root directory
        ROOT = os.path.join(ROOT, 'yolov5')

        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
        weights=ROOT / '7_seg.pt'  # model.pt path(s)
        source=ROOT / '/ros_ws/ws-ros1/src/meter_detect_and_read/cropped_msg_0.jpg'  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
        imgsz=(640, 640)  # inference size (height, width)
        conf_thres=0.6  # confidence threshold
        iou_thres=0.45  # NMS IOU threshold
        max_det=1000  # maximum detections per image
        # device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True  # show results
        save_txt=True  # save results to *.txt
        save_conf=False  # save confidences in --save-txt labels
        save_crop=False  # save cropped prediction boxes
        nosave=False  # do not save images/videos
        classes=None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False  # class-agnostic NMS
        augment=False  # augmented inference
        visualize=False  # visualize features
        update=False  # update all models
        project=ROOT / 'runs/predict-seg'  # save results to project/name
        name='exp'  # save results to project/name
        exist_ok=False  # existing project/name ok, do not increment
        line_thickness=3  # bounding box thickness (pixels)
        hide_labels=False  # hide labels
        hide_conf=False  # hide confidences
        half=False  # use FP16 half-precision inference
        dnn=False  # use OpenCV DNN for ONNX inference
        vid_stride=1  # video frame-rate stride
        retina_masks=False

        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        stride, names, pt = self.model_seg.stride, self.model_seg.names, self.model_seg.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size


        # Dataloader
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        self.model_seg.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model_seg.device)
                im = im.half() if self.model_seg.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False               
                pred, proto = self.model_seg(im, augment=augment, visualize=visualize)[:2]



            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                if len(det):
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    segj_list = []
                    # Segments
                    if save_txt:
                        segments = reversed(masks2segments(masks))
                        segments = [scale_segments(im.shape[2:], x, im0.shape).round() for x in segments]

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Mask plotting
                    annotator.masks(masks,
                                    colors=[colors(x, True) for x in det[:, 5]],
                                    im_gpu=None if retina_masks else im[i])

                    # Write results
                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):

                        if save_txt:  # Write to file
                            segj = segments[j].reshape(-1)  # (n,2) to (n*2)
                            class_name = cls.detach().cpu().numpy()
                            segj_list.append(segj)
                            segj_list.append(class_name)

                # Show masks results
                im0 = annotator.result()
                cv2.imwrite('/ros_ws/ws-ros1/src/meter_detect_and_read/mask.png', im0)


            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        
        return segj_list 

    def dect_bbox(self, boxes):
        boxs = boxes.pandas().xyxy[0].values
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

            self.position_pub.publish(self.boundingBoxes)
            


    def dectshow(self, org_img, boxs, height, width, pub):

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

            self.bboxes[boundingBox.Class] = [boundingBox.xmin, boundingBox.ymin, boundingBox.xmax, boundingBox.ymax]
            # print(self.bboxes)

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
            # cv2.putText(org_img, str(np.round(box[4], 2)),
            #             (int(box[0]), int(text_pos_y)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(org_img, 'id '+str(boundingBox.id),
            #             (int(box[0]), int(text_pos_y)+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            self.boundingBoxes.bounding_boxes.append(boundingBox)

        for i in range(len(self.boundingBoxes.bounding_boxes)):        
            cropped_img = org_img[self.boundingBoxes.bounding_boxes[i].ymin:self.boundingBoxes.bounding_boxes[i].ymax, self.boundingBoxes.bounding_boxes[i].xmin:self.boundingBoxes.bounding_boxes[i].xmax]
            cv2.imwrite(os.path.join(self.path_, 'yolov5_ros/yolov5_ros/media/meter_cropped'+str(self.boundingBoxes.bounding_boxes[i].id)+'.jpg') ,cropped_img)
        
        display_result = []
        display_result_string = ''

        if self.flag_for_digital_meter_reading:

            for i in range(len(self.boundingBoxes.bounding_boxes)):
                display_result.append(int(self.boundingBoxes.bounding_boxes[i].Class))

            display_result_string = ''.join(str(e) for e in display_result)

            print('Current measurement is %s'%(display_result_string))

            self.flag_for_digital_meter_reading = False
        


        self.boundingBoxes = BoundingBoxes()
        
        if self.flag_for_cropping_gauge:
            if boundingBox.Class=='gauge':
                self.det_gauge_id.data = str(boundingBox.id)
                cropped_img = org_img[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB ) 
                cv2.imwrite(os.path.join(self.path_, 'yolov5_ros/yolov5_ros/media/'+str(boundingBox.Class)+'_cropped_'+str(boundingBox.id)+'.jpg'),cropped_img)
                # cv2.imshow('Cropped gauge', cropped_img)                
                rospy.loginfo( '%s is cropped and saved', boundingBox.Class)


        elif self.flag_for_cropping_counter:
            if boundingBox.Class=='counter':
                self.det_counter_id.data = str(boundingBox.id)
                print(str(boundingBox.id))                
                cropped_img = org_img[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]
                cv2.imwrite(os.path.join(self.path_, 'yolov5_ros/yolov5_ros/media/'+str(boundingBox.Class)+'_cropped_'+str(boundingBox.id)+'.jpg'),cropped_img)                # cv2.imshow('Cropped counter', cropped_img)
                rospy.loginfo( '%s is cropped and saved', boundingBox.Class)



        elif self.flag_for_cropping_ssdisplay:
            if boundingBox.Class=='ss display':
                self.det_ssdisplay_id.data = str(boundingBox.id)
                cropped_img = org_img[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]
                cv2.imwrite(os.path.join(self.path_, 'yolov5_ros/yolov5_ros/media/'+str(boundingBox.Class)+'_cropped_'+str(boundingBox.id)+'.jpg'),cropped_img)                
                # cv2.imshow('Cropped ssdisplay', cropped_img)
                rospy.loginfo( '%s is cropped and saved', boundingBox.Class)

        self.publish_image(org_img, height, width, pub)
        # self.position_pub.publish(self.boundingBoxes)

    def publish_image(self, imgdata, height, width, pub):
        # image_temp = Image()
        image_temp = ros_numpy.msgify(Image, imgdata, encoding='bgr8')
        header = Header(stamp=rospy.Time.now())
        header.frame_id = 'camera_frame'
        image_temp.header = header
        pub.publish(image_temp)



    # functions for calculating


    def masks_reading(self, mask, cls):
        x_pic = []
        y_pic = []
        xy = tuple()
        list_of_coord = mask
        ###########################
        # write upsample
        ##########################
        # print('mask')
        # print(len(list_of_coord))
        # print(list_of_coord)

        for i in range(len(list_of_coord)):
          if i % 2 == 0:
            y_pic.append(int(list_of_coord[i]))
          else:
            x_pic.append(int(list_of_coord[i]))

        if cls == '0':

            contours_of_mask = []
            for i in range(len(y_pic)):
              contours_of_mask.append((y_pic[i],x_pic[i]))

            contours_of_mask_ndarray = np.array(contours_of_mask , dtype=int)
            contours_of_mask_ndarray_tuple = tuple()
            contours_of_mask_ndarray_tuple = (contours_of_mask_ndarray,)

            ellipse_ar = tuple()

            contours_flat = np.vstack(contours_of_mask_ndarray_tuple).squeeze()

            return contours_flat

        else:
            # src_check_needle = self.cropped_msg.copy()
            # for i in range(len(y_pic)):
            #     src_check_needle[x_pic[i],y_pic[i]] = [0,0,255]
            # cv2.imwrite('/ros_ws/ws-ros1/src/meter_detect_and_read/src_check_needle.png',src_check_needle)

            xy = (x_pic, y_pic)

            return xy

    def get_rot_m(self, rows, cols, scale, angle):
        M = cv2.getRotationMatrix2D((rows/2, cols/2), angle, 1)
        M[:,0:2] = np.array([[1,0],[0,scale]]) @ M[:,0:2]
        M[1,2] = M[1,2] * scale # This moves the ellipse so it doesn't end up outside the image (it's not correct to keep the ellipse in the middle of the image)

        return M

    def ransac_pred(self, x, y):

        y_pic_needle_arr = (np.array(y)).reshape(-1, 1)
        x_pic_needle_arr = (np.array(x)).reshape(-1, 1)

        ransac = linear_model.RANSACRegressor()
        ransac.fit(x_pic_needle_arr, y_pic_needle_arr)

        line_y_ransac = ransac.predict(x_pic_needle_arr)

        start_point = (int(line_y_ransac[0]),x_pic_needle_arr[0][0])
        end_point = (int(line_y_ransac[-1]),x_pic_needle_arr[-1][0])
        # print('start_point')
        # print(start_point)
        # print('end_point')
        # print(end_point)
        # src_ransac = self.cropped_msg.copy()
        # cv2.line(src_ransac, (int(start_point[0]), int(start_point[1])), (int(end_point[0]),int(end_point[1])), (255,255,0), 2, cv2.LINE_AA)
        # cv2.imwrite('/ros_ws/ws-ros1/src/meter_detect_and_read/src_ransac.png',src_ransac)


        return (start_point, end_point)

    def recalculate_el_desk(self, M_el2c, M_desk, xy):
        
        # calculate line coordinates for deskewed pic
        # return to ellipse2circle transform
        xy_desk = []
        # print('xy')
        # print(xy)
        for coord in xy:
            coord_arr = np.array([coord[0],coord[1], 1])
            coord_toelpic = M_el2c.dot(coord_arr.T)

            # return to deskewing transform
            coord_toelpic_arr= np.array([coord_toelpic[0],coord_toelpic[1], 1])            
            coord_toelpic_todeskwpic = M_desk.dot(coord_toelpic_arr.T)
            xy_desk.append(coord_toelpic_todeskwpic)

        return xy_desk

    def check_needle_end(self, needle_coord, center):

        d1 = math.sqrt( abs((int(needle_coord[0][0]) - center[0])^2 + (int(needle_coord[0][1]) - center[1])^2))
        d2 = math.sqrt( abs((int(needle_coord[1][0]) - center[0])^2 + (int(needle_coord[1][1]) - center[1])^2))
        # print('d1d2')
        # print(d1)
        # print(d2)

        if d1>d2:
          needle_end_x = int(needle_coord[1][0])
          needle_end_y = int(needle_coord[1][1])
          needle_start_x = int(needle_coord[0][0])
          needle_start_y = int(needle_coord[0][1])
        else:
          needle_end_x = int(needle_coord[0][0])
          needle_end_y = int(needle_coord[0][1])
          needle_start_x = int(needle_coord[1][0])
          needle_start_y = int(needle_coord[1][1])

        return ( (needle_start_x, needle_start_y),(needle_end_x, needle_end_y))

    def find_digits(self, boxes_dig):
        digits_coordinates = []
        digits_meaning = []
        phys_quan_classnum = ''

        for i in boxes_dig:
            if (i[6] != 'bar' and i[6] != 'lbs_in2'):
                x_mid = int((int(i[0]) + int(i[2])) / 2)
                y_mid = int((int(i[1]) + int(i[3])) / 2)
                digits_coordinates.append((x_mid, y_mid))
                digits_meaning.append([int(i[5])])
            else:
                phys_quan_classnum = i[6]

        digits_meaning = np.array(digits_meaning)
        return digits_coordinates, digits_meaning, phys_quan_classnum


    def digits_order(self, coord, dig):
        sorted_digits = []
        # если координата х одной цифры меньше, чем у другой, то цифра левее
        if coord[0][0]<coord[1][0]:
            sorted_digits.append(dig[0][0])
            sorted_digits.append(dig[1][0])
        else:
            sorted_digits.append(dig[1][0])
            sorted_digits.append(dig[0][0])
        return sorted_digits

    def digits_dictionary(self, digits_meaning, undesk_dig, phys_quan_classnum):

        digits_meaning_copy = digits_meaning
        dict_digits = {}
        digits_meaning_list = digits_meaning.tolist()
        digits_coordinates_deskewed_copy = undesk_dig
        black_list = []

        if phys_quan_classnum == 'lbs_in2':
            for c,i in enumerate(digits_meaning_list):
                dist = []
                if c in black_list:
                    continue
                # расстояние между текущей цифрой и остальными, включая текущую
                for j in range(0, len(digits_coordinates_deskewed_copy)):
                    dist.append(math.sqrt( abs((int(digits_coordinates_deskewed_copy[c][0]) - int(digits_coordinates_deskewed_copy[j][0]))^2 + (int(digits_coordinates_deskewed_copy[c][1]) - int(digits_coordinates_deskewed_copy[j][1]))^2)))
                # print(dist)
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
                res = self.digits_order(coord, dig)
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

        return dict_digits

    def gamma_digits(self, w, h, x_1, y_1, dict_digits, src_digits_final):

        gamma_digits = []
        
        contours_of_sector = [(w, 0), (w, h), (x_1, h),  (x_1, 0)]
        contours_of_sector_ndarray = np.array(contours_of_sector, dtype=int)
        contours_of_sector_ndarray_tuple = tuple()
        contours_of_sector_ndarray_tuple = (contours_of_sector_ndarray,)
        contours_of_sector_flat = np.vstack(contours_of_sector_ndarray_tuple).squeeze()

        for i in dict_digits.values():
            # print(i)
            check_point_digits = cv2.pointPolygonTest(contours_of_sector_flat, (int(i[0]), int(i[1])), False)
            # print(check_point_digits)
            d = math.sqrt((int(i[0]) - x_1) ** 2 + (int(i[1]) - h) ** 2)
            e = math.sqrt((int(i[0]) - x_1) ** 2 + (int(i[1]) - y_1) ** 2)
            f = math.sqrt((x_1 - x_1) ** 2 + (y_1 - h) ** 2)

            if int(check_point_digits) == 1:
                gamma_digits.append((180 - math.acos((e ** 2 + f ** 2 - d ** 2) / (2 * e * f)) * 180 / math.pi) + 180)
            else:
                gamma_digits.append(math.acos((e ** 2 + f ** 2 - d ** 2) / (2 * e * f)) * 180 / math.pi)

            cv2.line(src_digits_final, (x_1, y_1), (int(i[0]), int(i[1])), (255, 255, 0), 2, cv2.LINE_AA)

        # print(gamma_digits)
        # cv2_imshow(src_digits_final)

        pts = np.array([[w, 0], [w, h], [x_1, h], [x_1, 0]])
        cv2.polylines(src_digits_final, [pts], True, (200, 100, 200), 2)

        # cv2.imshow('src_digits_final',src_digits_final)
        # cv2.waitKey(0)
        return gamma_digits


    def gauge_calculate(self, boxes_dig, masks):

        mask_dict = {}
        mask_dict[int(masks[1])] = masks[0]
        mask_dict[int(masks[3])] = masks[2]
        # if '1' in mask_dict:
        #     print('Needle is found. Start to evaluate measurement')
        # else:
        #     return

        contours_flat = self.masks_reading(mask_dict[0], '0')
        ellipse = cv2.fitEllipse(contours_flat)
        (x_el, y_el), (MA, ma), angle = cv2.fitEllipse(contours_flat)

        src_with_max_contour = self.cropped_msg.copy()
        cv2.ellipse(src_with_max_contour, ellipse, (0, 255, 0), 3)
        box = cv2.boxPoints(ellipse)
        box = np.intp(box)
        cv2.drawContours(src_with_max_contour, [box], 0, (0,0,255), 3)
        # cv2.imshow('Ellipse',src_with_max_contour)
        # cv2.waitKey(0)

        # convert ellipse to circle
        src_el2c = self.cropped_msg.copy()

        M_el2c = self.get_rot_m(src_el2c.shape[0], src_el2c.shape[1], MA/ma, angle)        

        ellipce2circle = cv2.warpAffine(src_el2c, M_el2c, (src_el2c.shape[0],src_el2c.shape[1]), borderMode=cv2.BORDER_REPLICATE)
        vect_el = np.array([x_el, y_el, 1])
        new_el_coord = M_el2c.dot(vect_el.T)

        # draw circle
        src_with_circle = ellipce2circle.copy()
        cv2.ellipse(src_with_circle, ((new_el_coord[0], new_el_coord[1]), (MA,MA), 180-angle), (0, 255, 0), 3)
        circle = ((new_el_coord[0], new_el_coord[1]), (MA,MA), 180-angle)
        box_circle = cv2.boxPoints(circle)
        box_circle = np.intp(box_circle)
        cv2.drawContours(src_with_circle, [box_circle], 0, (0,0,255), 3)

        # cv2.imshow('ellipce2circle',src_with_circle)
        # cv2.waitKey(0)

        # deskewing rotated box
        src_deskewing = ellipce2circle.copy()

        if circle[1][0]<circle[1][1]:
        # rotate our image by -90 degrees around the image
            # print('w<h')
            angle_d = circle[2]-90
            M_desk = self.get_rot_m(src_deskewing.shape[1], src_deskewing.shape[0], MA/MA, angle_d) 

        else:
            # print('h<w')
            angle_d = circle[2]+180
            M_desk = self.get_rot_m(src_deskewing.shape[1], src_deskewing.shape[0], MA/MA, angle_d) 

        src_deskewing = cv2.warpAffine(src_deskewing, M_desk, (src_deskewing.shape[1], src_deskewing.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # cv2.imshow('src_deskewing',src_deskewing)
        # cv2.waitKey(0)
        # cv2.imwrite('/ros_ws/ws-ros1/src/meter_detect_and_read/src_deskewing.png',src_deskewing)

        # recalculate dots after affine transform

        # recalculate center
        vect = np.array([y_el,x_el,1])
        T = M_el2c.dot(vect.T)
        ellipce2circle[int(T[1]),int(T[0])] = [0,0,255]

        vect_box = np.array([int(T[0]),int(T[1]), 1])
        T_box = M_desk.dot(vect_box.T) # new center

        # cv2.imwrite('/ros_ws/ws-ros1/src/meter_detect_and_read/src_with_helplines.png', src_with_helplines)

        y_1 = int(T_box[1])
        x_1 = int(T_box[0])

        
        xy = self.masks_reading(mask_dict[1], '1')

        start_end_needle_pos = self.ransac_pred(xy[0], xy[1])


        undesk = self.recalculate_el_desk(M_el2c, M_desk, start_end_needle_pos)
        needle_coord = self.check_needle_end(undesk, (x_1, y_1))
        
        # from center to end point of needle
        
        # cv2.imshow('src_deskewing',src_deskewing)
        # cv2.waitKey(0)

        digits_coordinates, digits_meaning, phys_quan_classnum = self.find_digits(boxes_dig)

        undesk_dig = self.recalculate_el_desk(M_el2c, M_desk, digits_coordinates)

        undesk_dig = np.array(undesk_dig)

        dict_digits = self.digits_dictionary(digits_meaning, undesk_dig, phys_quan_classnum)

        src_digits_final = src_deskewing.copy()
        (h, w) = src_digits_final.shape[:2]

        gamma_digits = self.gamma_digits(w, h, x_1, y_1, dict_digits,src_digits_final )

        dict_digits_keys_arr = []
        for i in dict_digits.keys():
          dict_digits_keys_arr.append(i)

        gamma_digits_arr = (np.array(gamma_digits).reshape(-1, 1))
        digits_meaning_arr = (np.array(dict_digits_keys_arr).reshape(-1, 1))
        ransac = linear_model.RANSACRegressor()
        ransac.fit(gamma_digits_arr, digits_meaning_arr)
        line_y_dig_ransac = ransac.predict(gamma_digits_arr)
        d = math.sqrt( (needle_coord[1][0] - x_1 )**2 +  (needle_coord[1][1] - h )**2)
        e = math.sqrt( (needle_coord[1][0] - x_1 )**2 +  (needle_coord[1][1] - y_1)**2)
        f = math.sqrt( (x_1 - x_1 )**2 +  (h - y_1)**2)


        gamma_needle = math.acos( (e**2+f**2-d**2)/(2*e*f) ) *180/math.pi-2
        print(f"Gamma needle: {gamma_needle}")

        vmin = min(line_y_dig_ransac)
        qmin = min(gamma_digits_arr)
        qmax = max(gamma_digits_arr)
        vmax = max(line_y_dig_ransac)
        measurement = vmin + ((gamma_needle-qmin)/(qmax-qmin))*(vmax-vmin)
        src_final = src_deskewing.copy()

        if phys_quan_classnum is not None:
            print('Measurement: %s'%(str(np.round(measurement[0],5))+' '+phys_quan_classnum))
        else:
            print('Measurement: %s'%(str(np.round(measurement[0],5))))

        cv2.putText(src_final, str(np.round(measurement[0],2))+' '+phys_quan_classnum, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.line(src_final, (needle_coord[0][0], needle_coord[0][1]), (x_1,y_1), (255,255,0), 1, cv2.LINE_AA)

        src_final[y_1, x_1] = (0,0,255)
        cv2.imwrite('/ros_ws/ws-ros1/src/meter_detect_and_read/src_final.png', src_final)
        self.publish_image(src_final, src_final.shape[0], src_final.shape[1], self.measurement_image_pub)




def main():

    mode = 1 # choose mode to calculate, online - 0, by service - 

    rospy.init_node('yolov5_ros', anonymous=True)
    yolo_dect = Yolo_Dect()

    
    rate = rospy.Rate(1000)
    while not rospy.is_shutdown():
        bbox = yolo_dect.model_eval()
        det_boxs = bbox.pandas().xyxy[0].values
        yolo_dect.dectshow(yolo_dect.color_image, det_boxs, yolo_dect.msg_.height, yolo_dect.msg_.width, yolo_dect.image_pub)

        if mode == 1:

            if yolo_dect.flag_for_cropping_gauge:
                start_cnns = time.time()
                boxes_dig = yolo_dect.model_digits_gauge_eval()
                masks = yolo_dect.model_seg_eval()
                end_cnns = time.time()
                yolo_dect.gauge_calculate(boxes_dig, masks)
                print(f"Elapsed time to calculate meas {end_cnns-start_cnns}")
                yolo_dect.flag_for_cropping_gauge = False
            elif yolo_dect.flag_for_cropping_counter:

                start_cnns = time.time()
                yolo_dect.model_digits_counter_eval()
                end_cnns = time.time()
                print(f"Elapsed time to calculate meas {end_cnns-start_cnns}")
                yolo_dect.flag_for_cropping_counter = False


            elif yolo_dect.flag_for_cropping_ssdisplay:
                print('calculate ssdisplay')
                start_cnns = time.time()
                yolo_dect.model_digits_ssdisplay_eval()
                end_cnns = time.time()
                print(f"Elapsed time to calculate meas {end_cnns-start_cnns}")
                yolo_dect.flag_for_cropping_ssdisplay = False
        
        else:
            if det_boxs[6] == 'gauge':
                start_cnns = time.time()
                boxes_dig = yolo_dect.model_digits_gauge_eval()
                masks = yolo_dect.model_seg_eval()
                end_cnns = time.time()
                yolo_dect.gauge_calculate(boxes_dig, masks)
                print(f"Elapsed time to calculate meas {end_cnns-start_cnns}")
            elif det_boxs[6] == 'counter':
                print('counter')
            elif det_boxs[6] == 'ssdisplay':
                print('ssdisplay')

        



        rate.sleep()


if __name__ == "__main__":

    main()

