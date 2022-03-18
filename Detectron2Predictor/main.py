from detectron2_predictor import Detectron2Predictor
from utilities import create_file_list

##################################################
# Test the predictors

if __name__ == '__main__':

    """Detectron2 heads

    ObjectDetection
    SemanticSegmentation
    InstanceSegmentation
    PanopticSegmentation
    """
    predictor = Detectron2Predictor(head='ObjectDetection')






    main_dir = './' # Local Jupyter

    data_dir_cityscapes = main_dir + 'data/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/'
    anno_dir_cityscapes = main_dir + 'data/Cityscapes/gtFine_trainvaltest/gtFine/'

    sample_cityscapes_file_name_list, sample_cityscapes_file_path_list = create_file_list(data_dir_cityscapes + 'train/cologne')

    print(sample_cityscapes_file_name_list[:5])
    print(sample_cityscapes_file_path_list[:5])

    # for image_path in sample_cityscapes_file_path_list[:5]:
    #     predictor.test_image_file(image_path)






    # # sample_video_file_path = 'data/videos/video-clip.mp4'
    # # predictor.test_video_file(sample_video_file_path)


    image_path = 'data/Carla/test360.png'
    predictor.test_image_file(image_path)