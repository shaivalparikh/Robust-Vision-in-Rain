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
sys.path.append('../RainRemoval/')
import RainRemoval

sys.path.append('../Detectron2Predictor/')
import detectron2_predictor as d2

path = "../DrivingScripts/carla_data"
output_path = path
detectron = d2.Detectron2Predictor('SemanticSegmentation', model_path='../clear_model_final.pth')
remover = RainRemoval.RainRemoval('../40000_carla.pth')
count = 0
total = 0
for file in glob.glob(path+"/*.png"):
    name = file.split("/")[-1]
   
    if "_rain_" in file:
        count += 1
        print(name)
        array = Image.open(file)
        array = np.asarray(array)
        start = time.time()
        report = remover.infer(array)
        array_rr = report[0]
        output = detectron.test_image(array_rr)
        end = time.time()
        total += end - start
        # fig = plt.figure(figsize=(14, 7))
        
        # fig.add_subplot(1,3,1)
        # plt.axis("off")
        # plt.imshow(array)
        
        # fig.add_subplot(1,3,2)
        # plt.axis("off")
        # plt.imshow(array_rr)
        
        # fig.add_subplot(1,3,3)
        # plt.axis("off")
        # plt.imshow(output)
        
        # plt.savefig(output_path+"/"+name)

        op = np.hstack((array[:,:,:-1],array_rr,output))

        img = Image.fromarray(op,"RGB")

        img.save(output_path+"/"+name.split(".")[0]+"_rr.png")
        

print("FPS:", count/total)
