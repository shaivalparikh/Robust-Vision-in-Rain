from utilities import imshow_jupyter

import cv2, random

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

class Detectron2CustomDataset:
    train_classes = ['Unlabeled', 'Building', 'Fence', 'Pedestrian', 'Pole', 'Road', 'SideWalk', 'Vegetation', 'Vehicles', 'Wall', 'TrafficSign', 'Sky', 'TrafficLight', 'Terrain']
    train_colors = [(0, 0, 0), (70, 70, 70), (100, 40, 40), (220, 20, 60), (153, 153, 153), (128, 64, 128), (244, 35, 232), (107, 142, 35), (0, 0, 142), (102, 102, 156), (220, 220, 0), (70, 130, 180), (250, 170, 30), (145, 170, 100)]
    
    def __init__(self, train_dataset_name, val_dataset_name, get_train_dicts_fn, get_val_dicts_fn, classes=None, colors=None, ignore_label=0):
        
        if classes is None:
            self.classes = self.train_classes
        if colors is None:
            self.colors = self.train_colors
        print(f'Number of classes: {len(self.classes)}')
        
        self.train_dataset_name = train_dataset_name
        self.val_dataset_name = val_dataset_name
        
        self.get_train_dicts_fn = get_train_dicts_fn
        self.get_val_dicts_fn = get_val_dicts_fn

        print(f'Number of train images: {len(self.get_train_dicts_fn())}')
        print(f'Number of val images:   {len(self.get_val_dicts_fn())}')
        
        DatasetCatalog.register(self.train_dataset_name, self.get_train_dicts_fn)
        DatasetCatalog.register(self.val_dataset_name, self.get_val_dicts_fn)

        MetadataCatalog.get(self.train_dataset_name).stuff_classes = self.classes
        MetadataCatalog.get(self.train_dataset_name).stuff_colors = self.colors
        MetadataCatalog.get(self.train_dataset_name).ignore_label = 0

        MetadataCatalog.get(self.val_dataset_name).stuff_classes = self.classes
        MetadataCatalog.get(self.val_dataset_name).stuff_colors = self.colors
        MetadataCatalog.get(self.val_dataset_name).ignore_label = ignore_label
        
    def visualize_train_dataset(self, num_samples=1, size=(12, 6)):
        train_metadata = MetadataCatalog.get(self.train_dataset_name)
        data_train_dicts = self.get_train_dicts_fn()

        for file_dict in random.sample(data_train_dicts, num_samples):
            image = cv2.imread(file_dict['file_name'])
            visualizer = Visualizer(image[:, :, ::-1], metadata=train_metadata, scale=0.5)
            output = visualizer.draw_dataset_dict(file_dict)
            image = cv2.cvtColor(output.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
            imshow_jupyter(image, size=size)

    def visualize_val_dataset(self, predictor, num_samples=1, size=(12, 6)):
        val_metadata = MetadataCatalog.get(self.val_dataset_name)
        data_val_dicts = self.get_val_dicts_fn()

        for file_dict in random.sample(data_val_dicts, num_samples):
            image = cv2.imread(file_dict['file_name'])
            visualizer = Visualizer(image[:, :, ::-1], metadata=val_metadata, scale=0.5)
            
            print('Ground truth')
            output = visualizer.draw_dataset_dict(file_dict)
            target_image = cv2.cvtColor(output.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
            imshow_jupyter(target_image)
            
            print('Predicted')
            outputs = predictor(image)
            sem_seg = torch.argmax(outputs['sem_seg'], dim=0)
            output = visualizer.draw_sem_seg(sem_seg.to('cpu'))
            predicted_image = cv2.cvtColor(output.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
            imshow_jupyter(predicted_image)
            
            print()