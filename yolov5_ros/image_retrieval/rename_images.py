import cv2
import os

folder = '/home/human/Diana_Iakovleva/datasets/gauge_image_retrieval/pytorch_flower/test/'
images = []
i =0
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    width = 512
    height = 512
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(os.path.join(folder,str(i)+'.jpg'), resized)
    i+=1
    if img is not None:
        images.append(img)

