import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from nets import DGNLNet

class RainRemoval:
    def __init__(self,model):
        self.model = model

    def infer(self,img,flag = True):

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        torch.manual_seed(2019)
        torch.cuda.set_device(0)

        transform = transforms.Compose([
            transforms.Resize([512,1024]),
            transforms.ToTensor() ])

        to_pil = transforms.ToPILImage()    


        if type(img) == str:
          img = np.asarray(Image.open(img))
          
        net = DGNLNet().cuda()

        net.load_state_dict(torch.load(self.model,map_location=lambda storage,loc: storage.cuda(0)))

        net.eval()

        if img.shape[-1] == 4:
            img = img[:,:,:-1]

        self.img_infer = Image.fromarray(img)

        with torch.no_grad():

            w, h = self.img_infer.size
            img_var = Variable(transform(self.img_infer).unsqueeze(0)).cuda()
            
            res = net(img_var)

            torch.cuda.synchronize()

            self.result = transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu()))

            self.result_np = np.array(self.result)
        
        if flag:
            return self.result_np
        else:
            return self.result
    
    def displayRes(self):

        fig = plt.figure(figsize=(14, 7))
        
        fig.add_subplot(1,2,1)
        plt.axis("off")
        plt.imshow(self.img_infer)
        
        fig.add_subplot(1,2,2)
        plt.axis("off")
        plt.imshow(self.result)
        