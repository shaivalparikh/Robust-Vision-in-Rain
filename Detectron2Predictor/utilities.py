import os
from tkinter import Tcl

# Create a list of all files within a directory
def create_file_list(dir):
    file_name_list = os.listdir(dir)
    file_name_list = list(Tcl().call('lsort', '-dict', file_name_list)) # Sort by filename
    file_path_list = file_name_list.copy()
    for i in range(len(file_path_list)):
        file_path_list[i] = os.path.join(dir, file_name_list[i])
    return file_name_list, file_path_list