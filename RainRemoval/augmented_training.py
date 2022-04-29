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

sys.path.append('../Detectron2Predictor/')
import detectron2_predictor as d2

path = ""
output_path = path
detectron = Detectron2Predictor('SemanticSegmentation', model_path='../clear_model_final.pth')
count = 0
total = 0
for file in glob.glob(path+"/*.png"):
    name = file.split("/")[-1]
    if "_rain_" in file:
        count += 1
        start = time.time()
        array = Image.open(file)
        array = np.asarray(array)
        output = detectron.test_image(array)
        end = time.time()

        total += end - start
        op = np.hstack((array[:,:,:-1],output))

        img = Image.fromarray(op,"RGB")

        img.save(output_path+"/"+name.split()[0]+"_at.png")

print("FPS: ",count/total)