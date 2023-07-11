import time

import cv2
import numpy as np
import numpy
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from nets.facenet import Facenet
from nets_retinaface.retinaface import RetinaFace
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils import (Alignment_1, compare_faces, letterbox_image,
                         preprocess_input)
from utils.utils_bbox import (decode, decode_landm, non_max_suppression,
                              retinaface_correct_boxes)
import os


# --------------------------------------#
#   写中文需要转成PIL来写。
# --------------------------------------#
def cv2ImgAddText(img, label, left, top, textColor=(255, 255, 255)):
    img = Image.fromarray(np.uint8(img))
    # ---------------#
    #   设置字体
    # ---------------#
    font = ImageFont.truetype(font='model_data/simhei.ttf', size=20)

    draw = ImageDraw.Draw(img)
    label = label.encode('utf-8')
    draw.text((left, top), str(label, 'UTF-8'), fill=textColor, font=font)
    return np.asarray(img)


# --------------------------------------#
#   一定注意backbone和model_path的对应。
#   在更换facenet_model后，
#   一定要注意重新编码人脸。
# --------------------------------------#
class Retinaface(object):
    _defaults = {
        # ----------------------------------------------------------------------#
        #   retinaface训练完的权值路径
        # ----------------------------------------------------------------------#
        "retinaface_model_path": 'model_data/Retinaface_mobilenet0.25.pth',
        # ----------------------------------------------------------------------#
        #   retinaface所使用的主干网络，有mobilenet和resnet50
        # ----------------------------------------------------------------------#
        "retinaface_backbone": "mobilenet",
        # ----------------------------------------------------------------------#
        #   retinaface中只有得分大于置信度的预测框会被保留下来
        # ----------------------------------------------------------------------#
        "confidence": 0.6,
        # ----------------------------------------------------------------------#
        #   retinaface中非极大抑制所用到的nms_iou大小
        # ----------------------------------------------------------------------#
        "nms_iou": 0.4,
        # ----------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        #   输入图像大小会大幅度地影响FPS，想加快检测速度可以减少input_shape。
        #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
        #   会导致检测结果偏差，主干为resnet50不存在此问题。
        #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
        # ----------------------------------------------------------------------#
        "retinaface_input_shape": [640, 640, 3],
        # ----------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        # ----------------------------------------------------------------------#
        "letterbox_image": True,

        # ----------------------------------------------------------------------#
        #   facenet训练完的权值路径
        # ----------------------------------------------------------------------#
        "facenet_model_path": 'model_data/facenet_mobilenet.pth',
        # ----------------------------------------------------------------------#
        #   facenet所使用的主干网络， mobilenet和inception_resnetv1
        # ----------------------------------------------------------------------#
        "facenet_backbone": "mobilenet",
        # ----------------------------------------------------------------------#
        #   facenet所使用到的输入图片大小
        # ----------------------------------------------------------------------#
        "facenet_input_shape": [160, 160, 3],
        # ----------------------------------------------------------------------#
        #   facenet所使用的人脸距离门限
        # ----------------------------------------------------------------------#
        "facenet_threhold": 0.9,

        # --------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # --------------------------------#
        "cuda": False
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Retinaface
    # ---------------------------------------------------#
    def __init__(self, encoding=0, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.model_test = AntiSpoofPredict(0)
        self.image_cropper = CropImage()
        self.model_dir = r"model_data\FAS"
        # ---------------------------------------------------#
        #   不同主干网络的config信息
        # ---------------------------------------------------#
        if self.retinaface_backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50

        # ---------------------------------------------------#
        #   先验框的生成
        # ---------------------------------------------------#
        self.anchors = Anchors(self.cfg, image_size=(
            self.retinaface_input_shape[0], self.retinaface_input_shape[1])).get_anchors()
        self.generate()

        try:
            self.known_face_encodings = np.load(
                "model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone))
            self.known_face_names = np.load("model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone))
        except:
            if not encoding:
                print("载入已有人脸特征失败，请检查model_data下面是否生成了相关的人脸特征文件。")
            pass

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        self.net = RetinaFace(cfg=self.cfg, phase='eval', pre_train=False).eval()
        self.facenet = Facenet(backbone=self.facenet_backbone, mode="predict").eval()
        device = torch.device('cuda' if self.cuda else 'cpu')

        print('Loading weights into state dict...')
        state_dict = torch.load(self.retinaface_model_path, map_location=device)
        self.net.load_state_dict(state_dict)

        state_dict = torch.load(self.facenet_model_path, map_location=device)
        self.facenet.load_state_dict(state_dict, strict=False)

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

            self.facenet = nn.DataParallel(self.facenet)
            self.facenet = self.facenet.cuda()
        print('Finished!')

    def register(self, crop_img, name):
        cv2.imencode('.jpg', crop_img)[1].tofile('face_dataset/'+name+'_1.jpg')
        self.known_face_names = np.append(self.known_face_names, name)
        crop_img = np.array(
            letterbox_image(np.uint8(crop_img), (self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
        crop_img = crop_img.transpose(2, 0, 1)
        crop_img = np.expand_dims(crop_img, 0)
        # ---------------------------------------------------#
        #   利用图像算取长度为128的特征向量
        # ---------------------------------------------------#
        with torch.no_grad():
            crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
            if self.cuda:
                crop_img = crop_img.cuda()

            face_encoding = self.facenet(crop_img)[0].cpu().numpy()
            face_encoding = np.expand_dims(face_encoding, 0)
            self.known_face_encodings = np.append(self.known_face_encodings, face_encoding, axis=0)
        np.save("model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone), self.known_face_encodings)
        np.save("model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone), self.known_face_names)

    def encode_face_dataset(self, image_paths, names):
        face_encodings = []
        for index, path in enumerate(tqdm(image_paths)):
            # ---------------------------------------------------#
            #   打开人脸图片
            # ---------------------------------------------------#
            image = np.array(Image.open(path), np.float32)
            # ---------------------------------------------------#
            #   对输入图像进行一个备份
            # ---------------------------------------------------#
            old_image = image.copy()
            # ---------------------------------------------------#
            #   计算输入图片的高和宽
            # ---------------------------------------------------#
            im_height, im_width, _ = np.shape(image)
            # ---------------------------------------------------#
            #   计算scale，用于将获得的预测框转换成原图的高宽
            # ---------------------------------------------------#
            scale = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
            ]
            scale_for_landmarks = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0]
            ]
            if self.letterbox_image:
                image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
                anchors = self.anchors
            else:
                anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

            # ---------------------------------------------------#
            #   将处理完的图片传入Retinaface网络当中进行预测
            # ---------------------------------------------------#
            with torch.no_grad():
                # -----------------------------------------------------------#
                #   图片预处理，归一化。
                # -----------------------------------------------------------#
                image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(
                    torch.FloatTensor)

                if self.cuda:
                    image = image.cuda()
                    anchors = anchors.cuda()

                loc, conf, landms = self.net(image)
                # -----------------------------------------------------------#
                #   对预测框进行解码
                # -----------------------------------------------------------#
                boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])
                # -----------------------------------------------------------#
                #   获得预测结果的置信度
                # -----------------------------------------------------------#
                conf = conf.data.squeeze(0)[:, 1:2]
                # -----------------------------------------------------------#
                #   对人脸关键点进行解码
                # -----------------------------------------------------------#
                landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])

                # -----------------------------------------------------------#
                #   对人脸检测结果进行堆叠
                # -----------------------------------------------------------#
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

                if len(boxes_conf_landms) <= 0:
                    print(names[index], "：未检测到人脸")
                    continue
                # ---------------------------------------------------------#
                #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
                # ---------------------------------------------------------#
                if self.letterbox_image:
                    boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                                                                 np.array([self.retinaface_input_shape[0],
                                                                           self.retinaface_input_shape[1]]),
                                                                 np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

            # ---------------------------------------------------#
            #   选取最大的人脸框。
            # ---------------------------------------------------#
            best_face_location = None
            biggest_area = 0
            for result in boxes_conf_landms:
                left, top, right, bottom = result[0:4]

                w = right - left
                h = bottom - top
                if w * h > biggest_area:
                    biggest_area = w * h
                    best_face_location = result

            # ---------------------------------------------------#
            #   截取图像
            # ---------------------------------------------------#
            crop_img = old_image[int(best_face_location[1]):int(best_face_location[3]),
                       int(best_face_location[0]):int(best_face_location[2])]
            landmark = np.reshape(best_face_location[5:], (5, 2)) - np.array(
                [int(best_face_location[0]), int(best_face_location[1])])
            crop_img, _ = Alignment_1(crop_img, landmark)

            crop_img = np.array(
                letterbox_image(np.uint8(crop_img), (self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
            crop_img = crop_img.transpose(2, 0, 1)
            crop_img = np.expand_dims(crop_img, 0)
            # ---------------------------------------------------#
            #   利用图像算取长度为128的特征向量
            # ---------------------------------------------------#
            with torch.no_grad():
                crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                if self.cuda:
                    crop_img = crop_img.cuda()

                face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                face_encodings.append(face_encoding)

        np.save("model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone), face_encodings)
        np.save("model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone), names)

    def getFace(self, image):
        # ---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        # ---------------------------------------------------#
        old_image = image.copy()
        # ---------------------------------------------------#
        #   把图像转换成numpy的形式
        # ---------------------------------------------------#
        image = np.array(image, np.float32)

        # ---------------------------------------------------#
        #   Retinaface检测部分-开始
        # ---------------------------------------------------#
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        # ---------------------------------------------------#
        #   计算scale，用于将获得的预测框转换成原图的高宽
        # ---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        # ---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        # ---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
            anchors = self.anchors
        else:
            anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        # ---------------------------------------------------#
        #   将处理完的图片传入Retinaface网络当中进行预测
        # ---------------------------------------------------#
        with torch.no_grad():
            # -----------------------------------------------------------#
            #   图片预处理，归一化。
            # -----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                anchors = anchors.cuda()
                image = image.cuda()

            # ---------------------------------------------------------#
            #   传入网络进行预测
            # ---------------------------------------------------------#
            loc, conf, landms = self.net(image)
            # ---------------------------------------------------#
            #   Retinaface网络的解码，最终我们会获得预测框
            #   将预测结果进行解码和非极大抑制
            # ---------------------------------------------------#
            boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])

            conf = conf.data.squeeze(0)[:, 1:2]

            landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])

            # -----------------------------------------------------------#
            #   对人脸检测结果进行堆叠
            # -----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            if len(boxes_conf_landms) <= 0:
                return False, False, False, False

            # ---------------------------------------------------------#
            #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
            # ---------------------------------------------------------#
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms,
                                                             np.array([self.retinaface_input_shape[0],
                                                                       self.retinaface_input_shape[1]]),
                                                             np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

            sorted(boxes_conf_landms, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]))
            boxes_conf_landm = boxes_conf_landms[-1]

            boxes_conf_landm = np.maximum(boxes_conf_landm, 0)
            crop_img = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]),
                       int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
            landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
                [int(boxes_conf_landm[0]), int(boxes_conf_landm[1])])

            image_bbox = self.model_test.get_bbox(crop_img)
            image_bbox = np.maximum(image_bbox, 0)
            crop_img, _ = Alignment_1(crop_img, landmark)
            flag = False
            if im_height * 0.2 <= image_bbox[3] <= im_height * 0.8 \
                    and im_width * 0.1 <= boxes_conf_landm[0] <= im_width * 0.4:
                flag = True
            if not flag:
                return flag, flag, flag, flag
            return boxes_conf_landm, crop_img, True, image_bbox

    # -----------------------------------------------#
    #   利用活体检测模型，判别是否为活体
    # -----------------------------------------------#
    def isReal(self, old_image, image_box):
        prediction = np.zeros((1, 3))
        # -----------------------------------------------#
        # 通过两个网络进行预测
        # -----------------------------------------------#
        for model_name in os.listdir(self.model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": old_image,
                "bbox": image_box,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = self.image_cropper.crop(**param)
            # -----------------------------------------------#
            # 融合预测结果
            # -----------------------------------------------#
            prediction += self.model_test.predict(img, os.path.join(self.model_dir, model_name))
        label = np.argmax(prediction)
        value = prediction[0][label] / 2
        return label, value

    def identify(self, boxes_conf_landm, crop_img, old_image):
        crop_img = np.array(
            letterbox_image(np.uint8(crop_img), (self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
        crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), 0)
        with torch.no_grad():
            crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
            if self.cuda:
                crop_img = crop_img.cuda()

            # -----------------------------------------------#
            #   利用facenet_model计算长度为128特征向量
            # -----------------------------------------------#
            face_encoding = self.facenet(crop_img)[0].cpu().numpy()
        # -----------------------------------------------------#
        #   与数据库中所有的人脸进行对比，计算得分
        # -----------------------------------------------------#
        matches, face_distances = compare_faces(self.known_face_encodings, face_encoding,
                                                tolerance=self.facenet_threhold)
        name = "Unknown"
        # -----------------------------------------------------#
        #   取出这个最近人脸的评分
        #   取出当前输入进来的人脸，最接近的已知人脸的序号
        # -----------------------------------------------------#
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = self.known_face_names[best_match_index]
        # -----------------------------------------------#
        #   人脸特征比对-结束
        # -----------------------------------------------#

        for i, b in enumerate([boxes_conf_landm]):
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            # ---------------------------------------------------#
            #   b[0]-b[3]为人脸框的坐标，b[4]为得分
            # ---------------------------------------------------#
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # ---------------------------------------------------#
            #   b[5]-b[14]为人脸关键点的坐标
            # ---------------------------------------------------#
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)

            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(old_image, name, (b[0], b[3] - 15), font, 0.75, (255, 255, 255), 2)
            # --------------------------------------------------------------#
            #   cv2不能写中文，加上这段可以，但是检测速度会有一定的下降。
            #   如果不是必须，可以换成cv2只显示英文。
            # --------------------------------------------------------------#
            old_image = cv2ImgAddText(old_image, name, b[0] + 5, b[3] - 25)
        return old_image, name

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        # ---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        # ---------------------------------------------------#
        old_image = image.copy()
        # ---------------------------------------------------#
        #   把图像转换成numpy的形式
        # ---------------------------------------------------#
        image = np.array(image, np.float32)

        # ---------------------------------------------------#
        #   Retinaface检测部分-开始
        # ---------------------------------------------------#
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        # ---------------------------------------------------#
        #   计算scale，用于将获得的预测框转换成原图的高宽
        # ---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        # ---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        # ---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
            anchors = self.anchors
        else:
            anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        # ---------------------------------------------------#
        #   将处理完的图片传入Retinaface网络当中进行预测
        # ---------------------------------------------------#
        with torch.no_grad():
            # -----------------------------------------------------------#
            #   图片预处理，归一化。
            # -----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                anchors = anchors.cuda()
                image = image.cuda()

            # ---------------------------------------------------------#
            #   传入网络进行预测
            # ---------------------------------------------------------#
            loc, conf, landms = self.net(image)
            # ---------------------------------------------------#
            #   Retinaface网络的解码，最终我们会获得预测框
            #   将预测结果进行解码和非极大抑制
            # ---------------------------------------------------#
            boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])

            conf = conf.data.squeeze(0)[:, 1:2]

            landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])

            # -----------------------------------------------------------#
            #   对人脸检测结果进行堆叠
            # -----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            # ---------------------------------------------------#
            #   如果没有预测框则返回原图
            # ---------------------------------------------------#
            if len(boxes_conf_landms) <= 0:
                return old_image

            # ---------------------------------------------------------#
            #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
            # ---------------------------------------------------------#
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                                                             np.array([self.retinaface_input_shape[0],
                                                                       self.retinaface_input_shape[1]]),
                                                             np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

        # ---------------------------------------------------#
        #   Retinaface检测部分-结束
        # ---------------------------------------------------#

        # ---------------------------------------------------#
        #   活体检测部分-开始
        # ---------------------------------------------------#
        model_test = AntiSpoofPredict(0)
        image_cropper = CropImage()
        model_dir = r"model_data/FAS"

        # -----------------------------------------------#
        #   Facenet编码部分-开始
        # -----------------------------------------------#

        face_encodings = []
        ans = np.zeros((0, boxes_conf_landms.shape[1]))
        sorted(boxes_conf_landms, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]))
        boxes_conf_landm = boxes_conf_landms[-1]
        image_bbox = model_test.get_bbox(old_image)
        flag = False
        if im_height * 0.43 <= image_bbox[3] <= im_height * 0.67 and im_width * 0.1 <= image_bbox[0] <= im_width * 0.4:
            flag = True
        if not flag:
            return old_image
        image_bbox = np.maximum(image_bbox, 0)
        crop_img = np.array(old_image)[int(image_bbox[1]):int(image_bbox[1] + image_bbox[3]),
                   int(image_bbox[0]):int(image_bbox[0] + image_bbox[2])]
        landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
            [int(boxes_conf_landm[0]), int(boxes_conf_landm[1])])
        crop_img, _ = Alignment_1(crop_img, landmark)
        prediction = np.zeros((1, 3))
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": old_image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        # draw result of prediction
        label = np.argmax(prediction)
        value = prediction[0][label] / 2
        if label == 1:
            # print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
        else:
            # print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (255, 0, 0)

        b = list(map(int, boxes_conf_landm))
        cx = b[0]
        cy = b[1] - 5
        cv2.putText(
            old_image,
            result_text,
            (cx, cy),
            cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
        if label != 1:
            return old_image
        self.flag = True
        ans = numpy.append(ans, numpy.reshape(boxes_conf_landm, (1, 15)), axis=0)
        # ----------------------#
        #   人脸编码
        # ----------------------#
        crop_img = np.array(
            letterbox_image(np.uint8(crop_img), (self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
        crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), 0)
        with torch.no_grad():
            crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
            if self.cuda:
                crop_img = crop_img.cuda()

            # -----------------------------------------------#
            #   利用facenet_model计算长度为128特征向量
            # -----------------------------------------------#
            face_encoding = self.facenet(crop_img)[0].cpu().numpy()
            face_encodings.append(face_encoding)
        # for i, boxes_conf_landm in enumerate(boxes_conf_landms):
        #     # ----------------------#
        #     #   图像截取，人脸矫正
        #     # ----------------------#
        #     # boxes_conf_landm = np.maximum(boxes_conf_landm, 0)
        #     # crop_img = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]),
        #     #            int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
        #     # landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
        #     #     [int(boxes_conf_landm[0]), int(boxes_conf_landm[1])])
        #     # crop_img, _ = Alignment_1(crop_img, landmark)
        #
        #     image_bbox = model_test.get_bbox(old_image)
        #     flag = False
        #     if im_height*0.43 <= image_bbox[3] <= im_height*0.67 and im_width*0.1 <= image_bbox[0] <= im_width*0.4:
        #         flag = True
        #     if not flag:
        #         continue
        #     image_bbox = np.maximum(image_bbox, 0)
        #     crop_img = np.array(old_image)[int(image_bbox[1]):int(image_bbox[1]+image_bbox[3]),
        #                int(image_bbox[0]):int(image_bbox[0]+image_bbox[2])]
        #     landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
        #         [int(boxes_conf_landm[0]), int(boxes_conf_landm[1])])
        #     crop_img, _ = Alignment_1(crop_img, landmark)
        #     prediction = np.zeros((1, 3))
        #     for model_name in os.listdir(model_dir):
        #         h_input, w_input, model_type, scale = parse_model_name(model_name)
        #         param = {
        #             "org_img": old_image,
        #             "bbox": image_bbox,
        #             "scale": scale,
        #             "out_w": w_input,
        #             "out_h": h_input,
        #             "crop": True,
        #         }
        #         if scale is None:
        #             param["crop"] = False
        #         # param["crop"] = False
        #         img = image_cropper.crop(**param)
        #         prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        #     # draw result of prediction
        #     label = np.argmax(prediction)
        #     value = prediction[0][label] / 2
        #     if label == 1:
        #         # print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
        #         result_text = "RealFace Score: {:.2f}".format(value)
        #         color = (0, 0, 255)
        #     else:
        #         # print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
        #         result_text = "FakeFace Score: {:.2f}".format(value)
        #         color = (255, 0, 0)
        #
        #     b = list(map(int, boxes_conf_landm))
        #     cx = b[0]
        #     cy = b[1] - 5
        #     cv2.putText(
        #         old_image,
        #         result_text,
        #         (cx, cy),
        #         cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
        #     if label != 1:
        #         continue
        #     ans = numpy.append(ans, numpy.reshape(boxes_conf_landm, (1, 15)), axis=0)
        #     # ----------------------#
        #     #   人脸编码
        #     # ----------------------#
        #     crop_img = np.array(
        #         letterbox_image(np.uint8(crop_img), (self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
        #     crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), 0)
        #     with torch.no_grad():
        #         crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
        #         if self.cuda:
        #             crop_img = crop_img.cuda()
        #
        #         # -----------------------------------------------#
        #         #   利用facenet_model计算长度为128特征向量
        #         # -----------------------------------------------#
        #         face_encoding = self.facenet(crop_img)[0].cpu().numpy()
        #         face_encodings.append(face_encoding)
        # -----------------------------------------------#
        #   Facenet编码部分-结束
        # -----------------------------------------------#

        # -----------------------------------------------#
        #   人脸特征比对-开始
        # -----------------------------------------------#
        face_names = []
        for face_encoding in face_encodings:
            # -----------------------------------------------------#
            #   取出一张脸并与数据库中所有的人脸进行对比，计算得分
            # -----------------------------------------------------#
            matches, face_distances = compare_faces(self.known_face_encodings, face_encoding,
                                                    tolerance=self.facenet_threhold)
            name = "Unknown"
            # -----------------------------------------------------#
            #   取出这个最近人脸的评分
            #   取出当前输入进来的人脸，最接近的已知人脸的序号
            # -----------------------------------------------------#
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)
        # -----------------------------------------------#
        #   人脸特征比对-结束
        # -----------------------------------------------#
        for i, b in enumerate(ans):
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            # ---------------------------------------------------#
            #   b[0]-b[3]为人脸框的坐标，b[4]为得分
            # ---------------------------------------------------#
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # ---------------------------------------------------#
            #   b[5]-b[14]为人脸关键点的坐标
            # ---------------------------------------------------#
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)

            name = face_names[i]
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(old_image, name, (b[0], b[3] - 15), font, 0.75, (255, 255, 255), 2)
            # --------------------------------------------------------------#
            #   cv2不能写中文，加上这段可以，但是检测速度会有一定的下降。
            #   如果不是必须，可以换成cv2只显示英文。
            # --------------------------------------------------------------#
            old_image = cv2ImgAddText(old_image, name, b[0] + 5, b[3] - 25)
        return old_image

    def get_FPS(self, image, test_interval):
        # ---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        # ---------------------------------------------------#
        old_image = image.copy()
        # ---------------------------------------------------#
        #   把图像转换成numpy的形式
        # ---------------------------------------------------#
        image = np.array(image, np.float32)

        # ---------------------------------------------------#
        #   Retinaface检测部分-开始
        # ---------------------------------------------------#
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        # ---------------------------------------------------#
        #   计算scale，用于将获得的预测框转换成原图的高宽
        # ---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        # ---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        # ---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
            anchors = self.anchors
        else:
            anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        # ---------------------------------------------------#
        #   将处理完的图片传入Retinaface网络当中进行预测
        # ---------------------------------------------------#
        with torch.no_grad():
            # -----------------------------------------------------------#
            #   图片预处理，归一化。
            # -----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                anchors = anchors.cuda()
                image = image.cuda()

            # ---------------------------------------------------------#
            #   传入网络进行预测
            # ---------------------------------------------------------#
            loc, conf, landms = self.net(image)
            # ---------------------------------------------------#
            #   Retinaface网络的解码，最终我们会获得预测框
            #   将预测结果进行解码和非极大抑制
            # ---------------------------------------------------#
            boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])

            conf = conf.data.squeeze(0)[:, 1:2]

            landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])

            # -----------------------------------------------------------#
            #   对人脸检测结果进行堆叠
            # -----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

        if len(boxes_conf_landms) > 0:
            # ---------------------------------------------------------#
            #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
            # ---------------------------------------------------------#
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                                                             np.array([self.retinaface_input_shape[0],
                                                                       self.retinaface_input_shape[1]]),
                                                             np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

            # ---------------------------------------------------#
            #   Retinaface检测部分-结束
            # ---------------------------------------------------#

            # -----------------------------------------------#
            #   Facenet编码部分-开始
            # -----------------------------------------------#
            face_encodings = []
            for boxes_conf_landm in boxes_conf_landms:
                # ----------------------#
                #   图像截取，人脸矫正
                # ----------------------#
                boxes_conf_landm = np.maximum(boxes_conf_landm, 0)
                crop_img = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]),
                           int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
                landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
                    [int(boxes_conf_landm[0]), int(boxes_conf_landm[1])])
                crop_img, _ = Alignment_1(crop_img, landmark)

                # ----------------------#
                #   人脸编码
                # ----------------------#
                crop_img = np.array(letterbox_image(np.uint8(crop_img),
                                                    (self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
                crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), 0)
                with torch.no_grad():
                    crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                    if self.cuda:
                        crop_img = crop_img.cuda()

                    # -----------------------------------------------#
                    #   利用facenet_model计算长度为128特征向量
                    # -----------------------------------------------#
                    face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                    face_encodings.append(face_encoding)
            # -----------------------------------------------#
            #   Facenet编码部分-结束
            # -----------------------------------------------#

            # -----------------------------------------------#
            #   人脸特征比对-开始
            # -----------------------------------------------#
            face_names = []
            for face_encoding in face_encodings:
                # -----------------------------------------------------#
                #   取出一张脸并与数据库中所有的人脸进行对比，计算得分
                # -----------------------------------------------------#
                matches, face_distances = compare_faces(self.known_face_encodings, face_encoding,
                                                        tolerance=self.facenet_threhold)
                name = "Unknown"
                # -----------------------------------------------------#
                #   取出这个最近人脸的评分
                #   取出当前输入进来的人脸，最接近的已知人脸的序号
                # -----------------------------------------------------#
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                face_names.append(name)
            # -----------------------------------------------#
            #   人脸特征比对-结束
            # -----------------------------------------------#

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------------#
                #   传入网络进行预测
                # ---------------------------------------------------------#
                loc, conf, landms = self.net(image)
                # ---------------------------------------------------#
                #   Retinaface网络的解码，最终我们会获得预测框
                #   将预测结果进行解码和非极大抑制
                # ---------------------------------------------------#
                boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])

                conf = conf.data.squeeze(0)[:, 1:2]

                landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])

                # -----------------------------------------------------------#
                #   对人脸检测结果进行堆叠
                # -----------------------------------------------------------#
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            if len(boxes_conf_landms) > 0:
                # ---------------------------------------------------------#
                #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
                # ---------------------------------------------------------#
                if self.letterbox_image:
                    boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                                                                 np.array([self.retinaface_input_shape[0],
                                                                           self.retinaface_input_shape[1]]),
                                                                 np.array([im_height, im_width]))

                boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
                boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

                # ---------------------------------------------------#
                #   Retinaface检测部分-结束
                # ---------------------------------------------------#

                # -----------------------------------------------#
                #   Facenet编码部分-开始
                # -----------------------------------------------#
                face_encodings = []
                for boxes_conf_landm in boxes_conf_landms:
                    # ----------------------#
                    #   图像截取，人脸矫正
                    # ----------------------#
                    boxes_conf_landm = np.maximum(boxes_conf_landm, 0)
                    crop_img = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]),
                               int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
                    landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
                        [int(boxes_conf_landm[0]), int(boxes_conf_landm[1])])
                    crop_img, _ = Alignment_1(crop_img, landmark)

                    # ----------------------#
                    #   人脸编码
                    # ----------------------#
                    crop_img = np.array(letterbox_image(np.uint8(crop_img), (
                        self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
                    crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), 0)
                    with torch.no_grad():
                        crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                        if self.cuda:
                            crop_img = crop_img.cuda()

                        # -----------------------------------------------#
                        #   利用facenet_model计算长度为128特征向量
                        # -----------------------------------------------#
                        face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                        face_encodings.append(face_encoding)
                # -----------------------------------------------#
                #   Facenet编码部分-结束
                # -----------------------------------------------#

                # -----------------------------------------------#
                #   人脸特征比对-开始
                # -----------------------------------------------#
                face_names = []
                for face_encoding in face_encodings:
                    # -----------------------------------------------------#
                    #   取出一张脸并与数据库中所有的人脸进行对比，计算得分
                    # -----------------------------------------------------#
                    matches, face_distances = compare_faces(self.known_face_encodings, face_encoding,
                                                            tolerance=self.facenet_threhold)
                    name = "Unknown"
                    # -----------------------------------------------------#
                    #   取出这个最近人脸的评分
                    #   取出当前输入进来的人脸，最接近的已知人脸的序号
                    # -----------------------------------------------------#
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                    face_names.append(name)
                # -----------------------------------------------#
                #   人脸特征比对-结束
                # -----------------------------------------------#
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
