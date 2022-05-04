from utilities import create_file_list, imshow_jupyter

import numpy as np

import os, cv2

from tqdm import tqdm

##################################################
# Carla

def get_carla_file_list(data_dir, packages=[], levels=[]):
    file_list = []
    
    for package in packages:
        temp_file_name_list, temp_file_path_list = create_file_list(data_dir, package)
        
        for i in range(len(temp_file_name_list)):
            file_name_split = temp_file_name_list[i].split('_')
            if len(levels) > 0:
                if len(file_name_split) == 3: # [id, type, level.png]
                    file_id = file_name_split[0]
                    level = file_name_split[2].replace('.png', '')
                    if level in levels:
                        file_list.append((file_id, level, package))
            else:
                if len(file_name_split) == 2: # [id, type.png]
                    file_id = file_name_split[0]
                    level = ''
                    file_list.append((file_id, level, package))
                    
    print(f'Number of images: {len(file_list)}')
    
    return file_list

def get_carla_dicts(file_list, data_dir, clear=True, rain=True):
    dicts = []
    
    for file in file_list:
        file_id, level, package = file

        image_clear_path = os.path.join(data_dir, package, file_id + '_clear.png')
        image_rain_path = os.path.join(data_dir, package, file_id + '_rain_' + level + '.png')
        # image_semantic_path = os.path.join(data_dir, package, file_id + '_semantic_single.png') # Carla ID
        image_semantic_path = os.path.join(data_dir, package, file_id + '_semantic_train.png') # Train ID

        if clear == True:
            record = {}
            record['file_name'] = image_clear_path
            record['height'] = 720 # shape[0]
            record['width'] = 1280 # shape[1]
            # record['height'] = 512 # shape[0]
            # record['width'] = 1024 # shape[1]
            record['image_id'] = file_id + '_clear'
            record['sem_seg_file_name'] = image_semantic_path
            dicts.append(record)
        
        if rain == True:
            record = {}
            record['file_name'] = image_rain_path
            record['height'] = 720 # shape[0]
            record['width'] = 1280 # shape[1]
            record['image_id'] = file_id + '_rain'
            record['sem_seg_file_name'] = image_semantic_path
            dicts.append(record)

    return dicts

def get_carla_dicts2(file_list, data_dir, clear=True, rain=True):
    dicts = []
    
    for file in file_list:
        file_id, level, package = file

        image_clear_path = os.path.join(data_dir, package, file_id + '_clear.png')
        image_rain_path = os.path.join(data_dir, package, file_id + '_rain_' + level + '.png')
        # image_semantic_path = os.path.join(data_dir, package, file_id + '_semantic_single.png') # Carla ID
        image_semantic_path = os.path.join(data_dir, package, file_id + '_semantic_train.png') # Train ID

        if clear == True:
            record = {}
            record['file_name'] = image_clear_path
            # record['height'] = 720 # shape[0]
            # record['width'] = 1280 # shape[1]
            record['height'] = 512 # shape[0]
            record['width'] = 1024 # shape[1]
            # record['height'] = 256 # shape[0]
            # record['width'] = 512 # shape[1]
            record['image_id'] = file_id + '_clear'
            record['sem_seg_file_name'] = image_semantic_path
            dicts.append(record)
        
        if rain == True:
            record = {}
            record['file_name'] = image_rain_path
            record['height'] = 720 # shape[0]
            record['width'] = 1280 # shape[1]
            record['image_id'] = file_id + '_rain'
            record['sem_seg_file_name'] = image_semantic_path
            dicts.append(record)

    return dicts

##################################################
# Cityscapes

def get_cityscapes_file_list(data_dir, cities=[], levels=[]):
    file_list = []
    
    for city in cities:
        temp_file_name_list, _ = create_file_list(data_dir, city)
        
        for file_name in temp_file_name_list:
            file_list.append((file_name, city))
                    
    print(f'Number of images: {len(file_list)}')
    
    return file_list

def get_cityscapes_dicts(file_list, data_dir_main, data_dir_rain, anno_dir, clear=True, rain=True, levels=[]):
    dicts = []
    
    # for file in file_list:
    for index, file in enumerate(file_list):
        file_name, city = file

        image_clear_path = os.path.join(data_dir_main, city, file_name)
        
        file_name_split = file_name.split('_')
        # anno_name = file_name_split[0] + '_' + file_name_split[1] + '_' + file_name_split[2] + '_gtFine_labelIds.png' # Default
        image_id = file_name_split[0] + '_' + file_name_split[1] + '_' + file_name_split[2]
        anno_name = image_id + '_train.png' # Mapped
        image_semantic_path = os.path.join(anno_dir, city, anno_name)

        if clear == True:
            record = {}
            record['file_name'] = image_clear_path
            record['height'] = 1024 # shape[0]
            record['width'] = 2048 # shape[1]
            record['image_id'] = image_id + '_clear'
            record['sem_seg_file_name'] = image_semantic_path
            dicts.append(record)

        if rain == True:
            num_levels = len(levels)
            # for level in levels: # All 3 levels
            for level in [levels[index%num_levels]]: # 1 level
                image_rain_name = image_id + '_' + level + '.png'
                image_rain_path = os.path.join(data_dir_rain, city, image_rain_name)

                record = {}
                record['file_name'] = image_rain_path
                record['height'] = 1024
                record['width'] = 2048
                record['image_id'] = image_id + '_rain'
                record['sem_seg_file_name'] = image_semantic_path
                dicts.append(record)

    return dicts

##################################################
# Label conversion

map_cityscapes_id_to_carla_id = { 
     0:  0, # unlabeled           
     1:  0, # ego vehicle         
     2:  0, # rectification border
     3:  0, # out of roi          
     4: 19, # static              
     5: 20, # dynamic             
     6: 14, # ground              
     7:  7, # road                
     8:  8, # sidewalk            
     9:  0, # parking             
    10: 16, # rail track          
    11:  1, # building            
    12: 11, # wall                
    13:  2, # fence               
    14: 17, # guard rail          
    15: 15, # bridge              
    16:  0, # tunnel              
    17:  5, # pole                
    18:  0, # polegroup           
    19: 18, # traffic light       
    20: 12, # traffic sign        
    21:  9, # vegetation          
    22: 22, # terrain             
    23: 13, # sky                 
    24:  4, # person              
    25:  4, # rider               
    26: 10, # car                 
    27: 10, # truck               
    28: 10, # bus                 
    29:  0, # caravan             
    30:  0, # trailer             
    31: 10, # train               
    32: 10, # motorcycle          
    33: 10, # bicycle             
    -1:  0  # license plate       
}

map_carla_id_to_train_id = {
    0 :  0, # Unlabeled    # Unlabeled    # (0, 0, 0)
    1 :  1, # Building     # Building     # (70, 70, 70)
    2 :  2, # Fence        # Fence        # (100, 40, 40)
    3 :  0, # Other        #
    4 :  3, # Pedestrian   # Pedestrian   # (220, 20, 60)
    5 :  4, # Pole         # Pole         # (153, 153, 153)
    6 :  5, # RoadLine     #
    7 :  5, # Road         # Road         # (128, 64, 128)
    8 :  6, # SideWalk     # SideWalk     # (244, 35, 232)
    9 :  7, # Vegetation   # Vegetation   # (107, 142, 35)
    10:  8, # Vehicles     # Vehicles     # (0, 0, 142)
    11:  9, # Wall         # Wall         # (102, 102, 156)
    12: 10, # TrafficSign  # TrafficSign  # (220, 220, 0)
    13: 11, # Sky          # Sky          # (70, 130, 180)
    14:  0, # Ground       #
    15:  0, # Bridge       #
    16:  0, # RailTrack    #
    17:  0, # GuardRail    #
    18: 12, # TrafficLight # TrafficLight # (250, 170, 30)
    19:  0, # Static       #
    20:  0, # Dynamic      #
    21:  0, # Water        #
    22: 13  # Terrain      # Terrain      # (145, 170, 100)
}

def encode_labels(mask, map):
    label_mask = np.zeros_like(mask)
    for k in map:
        label_mask[mask == k] = map[k]
    return label_mask

def convert_carla(file_list, data_dir):
    for i in tqdm(range(len(file_list))):
        file_id, level, package = file_list[i]
        
        output_file_name = os.path.join(data_dir, package, file_id + '_semantic_train.png')
        
        if not os.path.exists(output_file_name):
            image_semantic_path = os.path.join(data_dir, package, file_id + '_semantic.png')
            image_semantic = cv2.imread(image_semantic_path)[:, :, 2] # HxWxC, BGR
            image_semantic = encode_labels(image_semantic, map=map_carla_id_to_train_id)

            # print(image_semantic)
            # imshow_jupyter(image_semantic)
            
            # print(output_file_name)
            cv2.imwrite(output_file_name, image_semantic)
            
def convert_cityscapes(file_list, anno_dir, output_dir):
    for i in tqdm(range(len(file_list))):
        file_name, city = file_list[i]
        file_name_split = file_name.split('_')
        
        output_file_name = file_name_split[0] + '_' + file_name_split[1] + '_' + file_name_split[2] + '_train.png'
        output_file_name = os.path.join(output_dir, city, output_file_name)
        
        if not os.path.exists(output_file_name):
            anno_name = file_name_split[0] + '_' + file_name_split[1] + '_' + file_name_split[2] + '_gtFine_labelIds.png'
            image_semantic_path = os.path.join(anno_dir, city, anno_name)
            image_semantic = cv2.imread(image_semantic_path)[:, :, 0] # HxWxC, BGR
            image_semantic = encode_labels(image_semantic, map=map_cityscapes_id_to_carla_id)
            image_semantic = encode_labels(image_semantic, map=map_carla_id_to_train_id)
        
            # print(image_semantic)
            # imshow_jupyter(image_semantic)
            
            # print(output_file_name)
            cv2.imwrite(output_file_name, image_semantic)