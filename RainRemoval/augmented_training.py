import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import glob
import time
import sys
import cv2

sys.path.append('../Detectron2Predictor/')
import detectron2_predictor as d2

path = "../DrivingScripts/carla_data"
output_path = path
detectron = d2.Detectron2Predictor('SemanticSegmentation', model_path='../com_model_final.pth')
count = 0
total = 0
for file in glob.glob(path+"/*.png"):
    name = file.split("/")[-1]
    if "_rain_" in file:
        count += 1
        start = time.time()
        array = Image.open(file)
        array = np.asarray(array)
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        output = detectron.test_image(array)
        end = time.time()

        total += end - start
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        op = np.hstack((array,output))

        img = Image.fromarray(op,"RGB")

        img.save(output_path+"/"+name.split()[0]+"_at.png")

print("FPS: ",count/total)
