import os
from tkinter import Tcl

# Create a list of all files within a directory
def create_file_list(dir='/', sub_dir=''):
    file_name_list = os.listdir(os.path.join(dir, sub_dir))
    file_name_list = list(Tcl().call('lsort', '-dict', file_name_list)) # Sort by filename
    file_path_list = file_name_list.copy()
    for i in range(len(file_path_list)):
        file_path_list[i] = os.path.join(sub_dir, file_name_list[i])
    return file_name_list, file_path_list

import numpy as np
import matplotlib.pyplot as plt
import cv2

def imshow_jupyter(image, type='RGB', size=(12, 6)):
    # print(f'Shape = {image.shape}, min value = {np.min(image)}, max value = {np.max(image)}')
    image = np.reshape(image, (image.shape[0], image.shape[1], -1))
    plt.figure(figsize=size)
    if image.shape[2] == 3:
        if type == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')
    plt.show()