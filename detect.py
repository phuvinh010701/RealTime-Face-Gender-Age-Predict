import argparse
import numpy as np
import cv2
import torch
import copy
import math
from utils.general import non_max_suppression_face, scale_coords
from models.experimental import attempt_load
import torchvision.transforms as T


IMAGENET_MEAN = 0.485, 0.456, 0.406
IMAGENET_STD = 0.229, 0.224, 0.225  

def classify_transforms(size=224):
    return T.Compose([CenterCrop(size), ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class LetterBox:
    def __init__(self, size=(640, 640), auto=False, stride=32):
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old
        h, w = round(imh * r), round(imw * r)  # resized image
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else self.h, self.w
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top:top + h, left:left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


class CenterCrop:

    def __init__(self, size=640):
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top:top + m, left:left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


class ToTensor:

    def __init__(self, half=False):
        super().__init__()
        self.half = half

    def __call__(self, im):  # im = np.array HWC in BGR order
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im


def get_cordinates(img, xyxy, extrapx=-5):
    h, w = img.shape[:2]
    x1 = max(0, int(xyxy[0]) - extrapx)
    y1 = max(0, int(xyxy[1]) - extrapx)
    x2 = min(w, int(xyxy[2]) + extrapx)
    y2 = min(h, int(xyxy[3]) + extrapx)
    return x1, x2, y1, y2

def show_results(img, x1, x2, y1, y2):
    return img[y1:y2, x1:x2]


def face_preprocessing(img, size=128):
    img = LetterBox(size)(img)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = img.float()
    img /= 255.0
    img = img.unsqueeze(0)

    return img

def get_class(age):
    age = age * 10
    return str(age) + '-' + str(age + 10)

def tensor_preprocessing(img, size):
    img = classify_transforms(size)(img)
    img = img.unsqueeze(0)
    return img


class detect():
    def __init__(self, model_face, model_gender, model_age, device):

        self.model_face = attempt_load(model_face, device)
        self.model_gender = torch.jit.load(model_gender, device)
        self.model_age = torch.jit.load(model_age, device)
        self.img_face = 640
        self.img_gender = 64
        self.conf_thres = 0.6
        self.iou_thres = 0.5
        self.face = None
        self.flag = False
        self.test_age = []
        self.test_gender = []

    def face_inference(self, img):

        im0 = copy.deepcopy(img)
        img = face_preprocessing(img)

        pred = self.model_face(img)[0]
        pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)

        for i, det in enumerate(pred): 

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    x1, x2, y1, y2 = get_cordinates(im0 ,xyxy)
                    self.face = show_results(im0, x1, x2, y1, y2)
        try:
            data = tensor_preprocessing(self.face, 128)
            age = self.model_age(data).numpy()
            age = np.argmax(age)

            gender = self.model_gender(data).numpy()
            gender = np.argmax(gender)
            gender = 'male' if gender == 1 else 'female'

            return x1, x2, y1, y2, gender, age
        except:
            return 0, 720, 0, 1280, 'male', 0
        


import time
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder
    opt = parser.parse_args()

    model_face = 'face.pt'
    model_gender = 'gender.pth'
    model_age = 'age.pth'

    device=torch.device('cpu')
    
    if opt.source == '0':
        a = detect(model_face, model_gender, model_age, device)
        cap = cv2.VideoCapture('rtsp://vjaip001:vjaip2022@192.168.1.145:554/stream1')
        i = 0
        prev_frame_time = 0
        new_frame_time = 0
        pred_age = 'None'
        pred_gender = 'None'

        temp_age = []
        temp_gender = []
        while True:
            ret, frame = cap.read()
            
            if ret:
                i += 1
                if i % 30 == 0:
                    values, counts = np.unique(temp_age, return_counts=True)
                    ind = np.argmax(counts)
                    pred_age = values[ind]
                    pred_age = get_class(pred_age)
                    temp_age = []

                
                s1 = time.time()
                x1, x2, y1, y2, gender, age = a.face_inference(frame)

                temp_age.append(age)

                new_frame_time = time.time()
                fps = int(1/(new_frame_time-prev_frame_time))
                prev_frame_time = new_frame_time

                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)
                frame = cv2.putText(frame, 'gender: ' + gender + ' - age:' + pred_age + ' - fps:' + str(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1, cv2.LINE_AA)   
                
                e1 = time.time()
                cv2.imshow('frame', frame)
                cv2.waitKey(1)

                print(e1 - s1)
            else:
                break

