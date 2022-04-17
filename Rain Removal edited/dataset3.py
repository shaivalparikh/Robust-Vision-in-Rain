import os
import os.path
import numpy as np
import torch.utils.data as data
from PIL import Image


def make_dataset(root, is_train):
    if is_train:

        input = open(os.path.join(root, 'data/train_images.txt'))
        ground_t = open(os.path.join(root, 'data/train_gt.txt'))
        depth_t = open(os.path.join(root, 'data/train_depth.txt'))
        image = [(os.path.join(root, 'data/images', img_name.strip('\n'))) for img_name in
                 input]
        gt = [(os.path.join(root, 'data/gt', img_name.strip('\n'))) for img_name in
                 ground_t]
        depth = [(os.path.join(root, 'data/depth', img_name.strip('\n'))) for img_name in
              depth_t]

        input.close()
        ground_t.close()
        depth_t.close()


        return [[image[i], gt[i], depth[i]]for i in range(len(image))]

    else:

        input = open(os.path.join(root, 'data/test_images.txt'))
        ground_t = open(os.path.join(root, 'data/test_gt.txt'))
        depth_t = open(os.path.join(root, 'data/test_depth.txt'))

        image = [(os.path.join(root, 'data/images', img_name.strip('\n'))) for img_name in
                 input]
        gt = [(os.path.join(root, 'data/gt', img_name.strip('\n'))) for img_name in
              ground_t]
        depth = [(os.path.join(root, 'data/depth', img_name.strip('\n'))) for img_name in
                 depth_t]

        input.close()
        ground_t.close()
        depth_t.close()

        return [[image[i], gt[i], depth[i]]for i in range(len(image))]



class ImageFolder(data.Dataset):
    def __init__(self, root, triple_transform=None, transform=None, target_transform=None, is_train=True):
        self.root = root
        self.imgs = make_dataset(root, is_train)
        self.triple_transform = triple_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path, depth_path = self.imgs[index]

        img = Image.open(img_path)
        target = Image.open(gt_path)
        depth = Image.open(depth_path)

        if len(img.getbands()) == 4:
          temp = np.asarray(img)
          temp = temp[:,:,:3]
          img = Image.fromarray(temp)
        if len(target.getbands()) == 4:
          temp = np.asarray(target)
          temp = temp[:,:,:3]
          target = Image.fromarray(temp)


        if self.triple_transform is not None:
            img, target, depth = self.triple_transform(img, target, depth)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            depth = self.target_transform(depth)

        return img, target, depth

    def __len__(self):
        return len(self.imgs)
