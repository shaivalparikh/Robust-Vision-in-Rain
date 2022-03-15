import os
import time
import sys
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
from nets import DGNLNet_fast, DGNLNet
import matplotlib.pyplot as plt

ckpt = "./ckpt/40000.pth"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2019)
torch.cuda.set_device(0)

transform = transforms.Compose([
    transforms.Resize([512,1024]),
    transforms.ToTensor() ])

to_pil = transforms.ToPILImage()



if __name__ == '__main__':


    img = sys.argv[1]

    net = DGNLNet().cuda()

    net.load_state_dict(torch.load(ckpt,map_location=lambda storage,loc: storage.cuda(0)))

    net.eval()

    if isinstance(img,Image.Image):
        img = img.convert("RGB")
    else:
        img = Image.open(Path(img))
        img = img.convert("RGB")
    with torch.no_grad():

        w, h = img.size
        img_var = Variable(transform(img).unsqueeze(0)).cuda()

        res = net(img_var)

        torch.cuda.synchronize()

        result = transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu()))

        plt.axis("off")
        plt.imshow(result)

        result.save("output_img.png")