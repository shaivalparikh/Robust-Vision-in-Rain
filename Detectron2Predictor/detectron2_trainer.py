import os

import detectron2

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import SemSegEvaluator

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference')
        return SemSegEvaluator(dataset_name, output_folder)

class Detectron2Trainer:
    def __init__(self, train_dataset_name, val_dataset_name, output_folder):
        
        config_path = 'configs/Misc/semantic_R_50_FPN_1x.yaml'
        # model_path = 'https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl'

        # All configs: https://detectron2.readthedocs.io/en/latest/modules/config.html
        
        self.train_dataset_name = train_dataset_name
        self.val_dataset_name = val_dataset_name

        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)

        # self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.DEVICE = 'cuda'
        # self.cfg.MODEL.DEVICE = 'cpu'

        self.cfg.DATASETS.TRAIN = (self.train_dataset_name,)
        # self.cfg.DATASETS.TEST = (self.train_dataset_name,)
        self.cfg.DATASETS.TEST = (self.val_dataset_name,)

        self.cfg.DATALOADER.NUM_WORKERS = 2

        self.cfg.INPUT.MIN_SIZE_TRAIN = 720
        self.cfg.INPUT.MAX_SIZE_TRAIN = 2048
        self.cfg.INPUT.MIN_SIZE_TEST = 720
        self.cfg.INPUT.MAX_SIZE_TEST = 2048

        # Number of images per batch across all machines. This is also the number
        # of training images per step (i.e. per iteration).
        self.cfg.SOLVER.IMS_PER_BATCH = 8
        self.cfg.SOLVER.BASE_LR = 0.01
        self.cfg.SOLVER.GAMMA = 0.1
        
        # self.cfg.SOLVER.MAX_ITER = 20000
        # # The iteration number to decrease learning rate by GAMMA.
        # self.cfg.SOLVER.STEPS = (10000, 15000, 18000, 19000)
        # # Save a checkpoint after every this number of iterations
        # self.cfg.SOLVER.CHECKPOINT_PERIOD = 1000
        # self.cfg.TEST.EVAL_PERIOD = 1000

        self.cfg.SOLVER.MAX_ITER = 40000
        self.cfg.SOLVER.STEPS = (20000, 30000, 36000, 38000)
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 2000
        self.cfg.TEST.EVAL_PERIOD = 2000

        self.cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 0
        classes = MetadataCatalog.get(self.train_dataset_name).stuff_classes
        self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(classes)
        
        # Directory where output files are written
        self.cfg.OUTPUT_DIR = output_folder
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        # self.trainer = DefaultTrainer(self.cfg)
        self.trainer = MyTrainer(self.cfg)
        
        self.evaluator = None
        self.predictor = None
        
    def load(self):
        self.trainer.resume_or_load(resume=True)
        
    def train(self):
        self.trainer.train()
        
    def get_predictor(self, output_folder=None, last_checkpoint=None):
        if output_folder is None:
            output_folder = self.cfg.OUTPUT_DIR
        if last_checkpoint is None:
            last_checkpoint = 'model_final.pth'
            with open(os.path.join(output_folder, 'last_checkpoint')) as file:
                last_checkpoint = file.read()
        print('Last checkpoint: ' + last_checkpoint)
            
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, last_checkpoint)
        self.predictor = DefaultPredictor(self.cfg)
        
    def test(self, output_folder=None, last_checkpoint=None):
        if output_folder is None:
            output_folder = self.cfg.OUTPUT_DIR
        self.get_predictor(output_folder, last_checkpoint)
        self.evaluator = SemSegEvaluator(self.val_dataset_name, output_dir=os.path.join(output_folder, 'inference'))
        self.trainer.test(self.cfg, self.predictor.model, self.evaluator)