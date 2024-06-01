import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import copy

# argument parser for command-line options
parser = argparse.ArgumentParser(description='Neural Style Transfer')
parser.add_argument('--style_image', type=str, required=True, help='Path to the style image')
parser.add_argument('--content_image', type=str, required=True, help='Path to the content image')

# parse arguments
args = parser.parse_args()

# determine device to run the network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)  # add a dimension to the tensor
    return image.to(device, torch.float)

style_img = image_loader(args.style_image)
content_img = image_loader(args.content_image)

assert style_img.size() == content_img.size(), \
    "We need to import style and content images of the same size"
