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

# parser for command-line options
parser = argparse.ArgumentParser(description='Neural Style Transfer with Multiple Style Images')
parser.add_argument('--style_images', type=str, nargs='+', required=True, help='Paths to the style images')
parser.add_argument('--content_image', type=str, required=True, help='Path to the content image')

args = parser.parse_args()

# device to run the network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU

# load images and convert to tensors
def load_images(image_paths, imsize):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB') 
        image = transforms.Resize((imsize, imsize))(image)  # ensure images are the same size
        image = transforms.ToTensor()(image).unsqueeze(0)
        images.append(image.to(device, torch.float))
    return images

# load and resize images to the new_size
style_imgs = load_images(args.style_images, imsize)
content_img = load_images([args.content_image], imsize)[0]

unloader = transforms.ToPILImage()  # reconvert into PIL image
plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# display the images
for i, style_img in enumerate(style_imgs):
    plt.figure()
    imshow(style_img, title=f'Style Image {i+1}')

plt.figure()
imshow(content_img, title='Content Image')

# content and style loss

# content loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_features):
        super(StyleLoss, self).__init__()
        self.targets = [gram_matrix(target).detach() for target in target_features]
        self.weights = [1.0 / len(target_features)] * len(target_features)

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = sum([F.mse_loss(G, target) * weight for target, weight in zip(self.targets, self.weights)])
        return input

# pretrained VGG network
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

# normalization mean and standard deviation of RGB
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# style and content layers
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# building the style transfer model
def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_imgs, content_img):
    normalization = Normalization(normalization_mean, normalization_std)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    i = 0  # initialize counter for layers

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers_default:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers_default:
            # add style loss:
            target_features = [model(style_img).detach() for style_img in style_imgs]
            style_loss = StyleLoss(target_features)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    # this line shows that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_imgs, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_imgs, content_img)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            # correct the values of updated input image using 'clamp', not 'clamp_'
            with torch.no_grad():
                input_img.data = input_img.data.clamp(0, 1)
            
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            loss = style_score * style_weight + content_score * content_weight
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
            return loss

        optimizer.step(closure)

    # a last correction without using in-place operation
    with torch.no_grad():
        input_img.data = input_img.data.clamp(0, 1)

    return input_img


# execution of style transfer
input_img = content_img.clone()
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_imgs, input_img)

plt.figure()
imshow(output, title='Output Image')
plt.ioff()
plt.show()
