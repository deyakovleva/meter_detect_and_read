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
# from yolov5_ros.yolov5.utils.torch_utils import select_device
# from yolov5_ros.yolov5.utils.dataloaders import LoadImages
# from yolov5_ros.yolov5.utils.general import (non_max_suppression, check_img_size)
from pathlib import Path
# from yolov5_ros.yolov5.utils.plots import Annotator
from yolov5_ros.yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5_ros.yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)
from yolov5_ros.yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5_ros.yolov5.utils.segment.general import masks2segments, process_mask
from yolov5_ros.yolov5.utils.torch_utils import select_device, smart_inference_mode
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
    path_ = '/ros_ws/ws-ros1/src/meter_detect_and_read/'

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
        # device = 'cuda'
        # device = select_device(device)
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
        # cv2.imwrite(os.path.join(self.path_, 'cropped_msg_0.jpg'), cropped_msg)
        results_dig = self.model_digits(cropped_msg)
        boxs_dig = results_dig.pandas().xyxy[0].sort_values(by='confidence').values
        print('boxes in function')
        print(boxs_dig)
           
        # self.dectshow(self.color_image, boxs_dig, self.msg_.height, self.msg_.width)
        # self.im_rate = self.msg_.header.seq
        return boxs_dig


    def model_seg_eval(self):
        self.color_image = ros_numpy.numpify(self.msg_)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        cropped_msg = self.color_image[self.bboxes[1]:self.bboxes[3],self.bboxes[0]:self.bboxes[2]]
        cv2.imwrite(os.path.join(self.path_, 'cropped_msg_0.jpg'), cropped_msg )

        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[1]  # YOLOv5 root directory
        ROOT = os.path.join(ROOT, 'yolov5')
        print(ROOT)

        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
        weights=ROOT / '7_seg.pt'  # model.pt path(s)
        source=ROOT / '/ros_ws/ws-ros1/src/meter_detect_and_read/cropped_msg_0.jpg'  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
        imgsz=(640, 640)  # inference size (height, width)
        conf_thres=0.6  # confidence threshold
        iou_thres=0.45  # NMS IOU threshold
        max_det=1000  # maximum detections per image
        device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False  # show results
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
                    # for c in det[:, 5].unique():
                    #     n = (det[:, 5] == c).sum()  # detections per class
                    #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Mask plotting
                    # annotator.masks(masks,
                    #                 colors=[colors(x, True) for x in det[:, 5]],
                    #                 im_gpu=None if retina_masks else im[i])

                    # Write results
                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):

                        if save_txt:  # Write to file
                            segj = segments[j].reshape(-1)  # (n,2) to (n*2)
                            class_name = [cls]
                            segj_list.append([segj])
                            segj_list.append([class_name])
                            # print(segj_list)
                            # boxj= [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                            # print(boxj)
                    #         line = (cls, *segj, conf) if save_conf else (cls, *segj)  # label format
                    #         with open(f'{txt_path}.txt', 'a') as f:
                    #             f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    #         line_bbox = (cls, *xyxy, conf) if save_conf else (cls, *xyxy)  # label format
                    #         with open(txt_path + '_bbox.txt', 'a') as f:
                    #             f.write(('%g ' * len(line_bbox)).rstrip() % line_bbox + '\n')

                        # if save_img or save_crop or view_img:  # Add bbox to image
                        #     c = int(cls)  # integer class
                        #     label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        #     annotator.box_label(xyxy, label, color=colors(c, True))
                        #     # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
                        # if save_crop:
                        #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                # im0 = annotator.result()
                # if view_img:
                #     if platform.system() == 'Linux' and p not in windows:
                #         windows.append(p)
                #         cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                #     cv2.imshow(str(p), im0)
                #     if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                #         exit()

                # Save results (image with detections)
                # if save_img:
                #     if dataset.mode == 'image':
                #         cv2.imwrite(save_path, im0)
                #     else:  # 'video' or 'stream'
                #         if vid_path[i] != save_path:  # new video
                #             vid_path[i] = save_path
                #             if isinstance(vid_writer[i], cv2.VideoWriter):
                #                 vid_writer[i].release()  # release previous video writer
                #             if vid_cap:  # video
                #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #             else:  # stream
                #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
                #             save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                #         vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        
        return segj_list 


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
            cv2.imwrite(os.path.join(self.path_, 'yolov5_ros/yolov5_ros/media/meter_cropped'+str(self.boundingBoxes.bounding_boxes[i].id)+'.jpg') ,cropped_img)
        self.boundingBoxes = BoundingBoxes()
        

        # print(self.flag_for_cropping_gauge)

        if self.flag_for_cropping_gauge:
            if boundingBox.Class=='gauge':
                self.det_gauge_id.data = str(boundingBox.id)
                print('got in if')
                cropped_img = org_img[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB ) 
                cv2.imwrite(os.path.join(self.path_, 'yolov5_ros/yolov5_ros/media/'+str(boundingBox.Class)+'_cropped_'+str(boundingBox.id)+'.jpg'),cropped_img)
                # cv2.imshow('Cropped gauge', cropped_img)
                
                rospy.loginfo( '%s is cropped and saved', boundingBox.Class)
                # cv2.waitKey(0)
                # self.flag_for_cropping_gauge = False
            # else:
            #     return

        elif self.flag_for_cropping_counter:
            if boundingBox.Class=='counter':
                self.det_counter_id.data = str(boundingBox.id)
                print(str(boundingBox.id))                
                cropped_img = org_img[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]
                cv2.imwrite(os.path.join(self.path_, 'yolov5_ros/yolov5_ros/media/'+str(boundingBox.Class)+'_cropped_'+str(boundingBox.id)+'.jpg'),cropped_img)                # cv2.imshow('Cropped counter', cropped_img)
                rospy.loginfo( '%s is cropped and saved', boundingBox.Class)
                # cv2.waitKey(0)
                self.flag_for_cropping_counter = False
            # else:
            #     return


        elif self.flag_for_cropping_ssdisplay:
            if boundingBox.Class=='ss display':
                self.det_ssdisplay_id.data = str(boundingBox.id)
                cropped_img = org_img[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]
                cv2.imwrite(os.path.join(self.path_, 'yolov5_ros/yolov5_ros/media/'+str(boundingBox.Class)+'_cropped_'+str(boundingBox.id)+'.jpg'),cropped_img)                
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
        if yolo_dect.flag_for_cropping_gauge:
            start_cnns = time.time()
            now = rospy.get_rostime()
            boxes_dig = yolo_dect.model_digits_eval()
            masks = yolo_dect.model_seg_eval()
            print(masks)
            end = rospy.get_rostime()
            end_cnns = time.time()
            print('Evaluation complete')
            print(end_cnns-start_cnns)
            yolo_dect.flag_for_cropping_gauge = False
        rate.sleep()


if __name__ == "__main__":

    main()
