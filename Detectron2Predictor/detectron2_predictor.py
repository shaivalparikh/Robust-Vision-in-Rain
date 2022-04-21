GOOGLE_COLAB = False

from detectron2_dataset import Detectron2CustomDataset

import torch

import matplotlib.pyplot as plt

import time
import cv2
if GOOGLE_COLAB:
    from google.colab.patches import cv2_imshow

import detectron2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# from detectron2.projects.point_rend import add_pointrend_config

def get_dict():
    return {}
default_dataset = Detectron2CustomDataset('train', 'val', get_dict, get_dict)

"""Detectron2 heads

ObjectDetection
SemanticSegmentation
InstanceSegmentation
PanopticSegmentation
"""
class Detectron2Predictor:
    def __init__(self, head='SemanticSegmentation', model_path=None):

        self.head = head
        
        self.cfg = get_cfg()

        # Model Zoo
        # https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

        if self.head == 'ObjectDetection':
            config_file = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml' # 1531 MiB
            config_path = model_zoo.get_config_file(config_file)
            model_path  = model_zoo.get_checkpoint_url(config_file)
            
            self.cfg.merge_from_file(config_path)

        elif self.head == 'SemanticSegmentation':
            config_path = 'configs/Misc/semantic_R_50_FPN_1x.yaml'
            self.cfg.merge_from_file(config_path)
            
            if model_path is None:
                model_path = 'https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl'
                
            self.cfg.DATASETS.TRAIN = ('train')
            self.cfg.DATASETS.TEST = ('val')
            
            classes = MetadataCatalog.get('train').stuff_classes
            self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(classes)
            
            # # PointRend
            # # https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend
            # add_pointrend_config(self.cfg)
            # config_path = 'configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml'
            # model_path  = 'detectron2://PointRend/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes/202576688/model_final_cf6ac1.pkl'
        
        elif self.head == 'InstanceSegmentation':
            config_file = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml' # 1539 MiB
            config_path = model_zoo.get_config_file(config_file)
            model_path  = model_zoo.get_checkpoint_url(config_file)
            self.cfg.merge_from_file(config_path)

        elif self.head == 'PanopticSegmentation':
            config_file = 'COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml' # 720: 1945 MiB # 360: 1755 MiB
            # config_file = 'COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml' # 2019 MiB
            config_path = model_zoo.get_config_file(config_file)
            model_path  = model_zoo.get_checkpoint_url(config_file)
            self.cfg.merge_from_file(config_path)

        # print(config_path)
        # print(model_path)

        self.cfg.MODEL.WEIGHTS = model_path

        # self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = 'cuda'

        self.predictor = DefaultPredictor(self.cfg)

    def test_image(self, image, show_original=False):

        # print(image.shape)
        
        if show_original:
            if GOOGLE_COLAB:
                cv2_imshow(image)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(20, 10))
                plt.imshow(image)
                plt.show()

        if self.head == 'SemanticSegmentation':
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN))
            start_time = time.time()
            outputs = self.predictor(image)
            stop_time = time.time()
            sem_seg = torch.argmax(outputs['sem_seg'], dim=0)
            # print(outputs['sem_seg'].shape)
            # print(sem_seg.shape)
            # print(sem_seg)

            out = v.draw_sem_seg(sem_seg.to('cpu'))

        elif self.head == 'PanopticSegmentation':
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            start_time = time.time()
            outputs = self.predictor(image)
            stop_time = time.time()
            panoptic_seg, segments_info = outputs['panoptic_seg']
            # print(outputs['panoptic_seg'])
            # print(panoptic_seg)
            # print(segments_info)

            out = v.draw_panoptic_seg_predictions(panoptic_seg.to('cpu'), segments_info)

        else:
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            start_time = time.time()
            outputs = self.predictor(image)
            stop_time = time.time()
            # print(outputs['instances'])
            # print(outputs['instances'].pred_classes)
            # print(outputs['instances'].pred_boxes)

            out = v.draw_instance_predictions(outputs['instances'].to('cpu'))

        if GOOGLE_COLAB:
            cv2_imshow(out.get_image()[:, :, ::-1])
        else:
            # cv2.imshow('Detectron2 Predictor', out.get_image()[:, :, ::-1])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            image = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(20, 10))
            plt.imshow(image)
            plt.show()

        print(f'Time = {stop_time - start_time}, freq = {1/(stop_time - start_time)}')

    def test_image_file(self, image_path, show_original=False):
        image = cv2.imread(image_path)
        self.test_image(image, show_original)

    def test_video_file(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if cap.isOpened() == False:
            print('Video opening error!')
            return
        
        (success, image) = cap.read()
        while success:
            self.test_image(image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
            (success, image) = cap.read()


##################################################
# Test the predictors

if __name__ == '__main__':
    
    predictor = Detectron2Predictor(head='SemanticSegmentation', model_path='output_all_40k/model_final.pth')
    sample_file_path = '/home/tunx404/Miscellaneous/data/CarlaNight/night_packaging/package10/1649876436.6209295_clear.png'
    predictor.test_image_file(sample_file_path)

    # sample_video_file_path = 'data/videos/video-clip.mp4'
    # predictor.test_video_file(sample_video_file_path)