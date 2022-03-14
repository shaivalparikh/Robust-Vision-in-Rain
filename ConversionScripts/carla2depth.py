import sys
import os
from PIL import Image
import PIL

def main():
    if len(sys.argv) < 2:
        print('Usage: argument of path of folder with depth images [-l for logdepth images]')
        return
    tag = 'depth'
    if len(sys.argv) == 3:
        tag = 'logdepth'
    folder_name = sys.argv[1]
    files = os.listdir(folder_name)
    #print(files)
    
    new_dir = "converted_images"
    if new_dir not in os.listdir():
        os.mkdir(new_dir)
    
    for data in files:
        label = '_' + tag + '.png'
        if label in data:
            filename = folder_name + '/' + data
            with Image.open(filename) as image:
                new_filename = filename[:-4] + '_converted' + '.png'
                new_filename = new_filename.replace(folder_name, new_dir)
                #print(new_filename)
                image = PIL.ImageOps.invert(image)
                image.save(new_filename)
            
    
if __name__ == '__main__':
    main()
