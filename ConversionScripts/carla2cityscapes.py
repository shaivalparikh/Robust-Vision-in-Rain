import sys
import os
from PIL import Image
import PIL
import carla_converter

def main():
    if len(sys.argv) < 2:
        print('Usage: argument of path of folder with semantic segmentation images')
        return
    tag = 'semantic'
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
                conv_image = image.copy()
                conv_image = conv_image.split()[0]
                pixels = conv_image.load()
                for i in range(conv_image.size[0]):
                    for j in range(conv_image.size[1]):
                        pixels[i, j] = carla_converter.carla2cityscapes(pixels[i, j])
                conv_image.save(new_filename)
                visual_image = image.copy()
                colors = visual_image.load()
                for i in range(visual_image.size[0]):
                    for j in range(conv_image.size[1]):
                        colors[i, j] = carla_converter.citysemantic2citycolor(pixels[i, j])
                visual_filename = new_filename.replace('_converted', '_color')
                visual_image.save(visual_filename)
            
    
if __name__ == '__main__':
    main()
