import cv2
import numpy as np
import detectron2
import os
import sys
import time

sys.path.append('../Detectron2Predictor/')
import detectron2_predictor as d2
import PIL

sys.path.append('../RainRemoval/')
import RainRemoval

folder = './_out/'


#detectron = d2.Detectron2Predictor('SemanticSegmentation', model_path='../Detectron2Predictor/output/clear_model_final.pth')
detectron = d2.Detectron2Predictor('SemanticSegmentation', model_path='../Detectron2Predictor/output/com_model_final.pth')
remover = RainRemoval.RainRemoval('../Rain Removal edited/ckpt/DGNLNet/40000_carla.pth')

count = 0
start = time.time()
for name in os.listdir(folder):
        array = cv2.imread(folder + name)
        
        #array = remover.infer(array)
        #array = detectron.test_image(array, output_numpy=True)
        #cv2.imshow('test', array)
        #cv2.waitKey(1)
        
        count += 1
end = time.time()
print(count)
print('FPS:', count / (end - start))
