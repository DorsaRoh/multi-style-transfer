# Neural Style Transfer for Multiple Images

### What is it?

Style transfer (for multiple images) is the synthesis of images, creating an output that has the content of one image and the styles of  other images.

## How it Works

### Concept

The goal of neural style transfer is to minimize how different the content and style are between images. This involves taking the content of one image and blending it with the artistic style of others.

### Prerequisite Knowledge

#### Tensors
Tensors are multi-dimensional arrays, essential for storing data in neural networks. They allow the network to process and learn from the data efficiently.
- **Analogy**: Imagine one block equals one number. A row of these blocks is a vector. Stacking many rows of blocks on top of each other forms a matrix. Stacking many matrices results in a tensor.

#### GPUs vs. CPUs for Machine Learning
- **CPUs**: Good for tasks that require sequential computing and multi-threading.
- **GPUs**: Better suited for tasks that require parallel computing, like performing large scale matrix multiplications essential in neural networks.

## Steps to Perform Style Transfer

### 1. Add Desired Images
Place your style and content images in the 'images' folder.

### 2. Set the Images
Run the neural style transfer script by specifying the paths to your images, where:

`content_image.jpg` is the name of the content image
<br>
`style_image.jpg` are the names of the style images

<i>note: the images do <b>not</b> need to be .jpg</i>

```bash
python neural_style.py --style_images images/style_image1.jpg images/style-images/style_image2.jpg --content_image images/content_image.jpg
```
replace <u>style_images1.jpg</u> and <u>style_images2.jpg</u> with the names of your style images. Feel free to add more images as well

### 3. Image Preprocessing
Images need to be preprocessed to fit the neural network's requirements. This typically includes resizing and normalization.
```bash
from torchvision import transforms
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()       # transform it into a torch tensor
])
```

Add a batch dimension required by neural networks for batch processing:
```bash
.unsqueeze(0)
```

### 4. Content Loss
Content loss measures how much the content of the generated image differs from the content of the target image. It's calculated using a pre-trained CNN, like VGG19, which extracts feature maps from both images.
```bash
import torch.nn.functional as F
class ContentLoss(torch.nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()  # detach target from the graph to treat as a constant reference
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)  # calculate the mean squared error
        return input  # return input image unchanged
```

### 5. Running the Style Transfer
Optimize the generated image to minimize the content and style losses:

### Result
After running the style transfer, you will get an image that combines the content of the content image with the artistic style of the style image - a successful style transfer!

## Connect with Me

[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/DorsaRoh)
[![Twitter](https://img.shields.io/badge/-Twitter-1DA1F2?style=for-the-badge&logo=twitter)](https://twitter.com/Dorsa_Rohani)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/dorsarohani/)

Feel free to contribute 😊



