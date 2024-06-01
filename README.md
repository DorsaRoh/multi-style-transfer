STEPS:

1. add desired images (style and content) to 'image' folder

to set the style and content images:

2. run: python neural_style.py --style_image path/to/style_image.jpg --content_image path/to/content_image.jpg




what is it?
Style transfer is the synthesis of two images, creating output with the content of one image, and the style of the other.


How it works:

concept:
to minimize how how different the content and style are between two images. 


basic prerequisite knowledge:
- tensors: 
analogy: let one block equal one number. a row of these blocks is a vector. if you stack many rows of blocks on top of each other, you get a matrix. 

if you stack many matrices on top of each other, you get a tensor.

tensors are like the memory of the neural network. they hold lots of data for the network to use and learn, by trying to find patterns. for instance, in an image classification example: if you give the neural network pictures of cats and dogs, the tensors enable the network to classify them to be 'dog' or 'cat' by remembering lots of details.

thus, tensors are the inputs to neural networks.

- gpus vs cpus for machine learning:
cpus: good for sequential computing and multi-threading
gpus: good for parallel computing (can perform a ton of linear algebra and matrix multiplication)


now let's dive in.

```
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor
```

we often need to preprocess images into consistent formats before feeding them into a neural network. this includes resizing, normaization, and converting the images into tensors.


```.unsqueeze(0)```

adds a batch dimension to the tensor (required by the neural network). 

neural networks process data in batches to leverage parallelism and improve computational efficiency. 


- content loss

the content loss determines how different the generated image is from the target content image (best achieved output).

how is it calculated? it uses a pre-trained CNN, in this case, VGG19, to extract feature maps (feature maps capture levels of abstraction in the image like edges, textures, shapes, and objects)

the difference is calculated using the mean squared error.


```
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()  # detach: don't track changes to this image - need to treat as a constant reference (not something that changes during the training process)

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)  # calculate the mean squared error (measures how similar two images are by comparing the difference in their pixel values - the smaller the loss, the more similar the images)
        return input  # return input image unchanged

```
ContentLoss: Inherits from nn.Module and calculates the content loss.
detach(): Detaches the target tensor from the computation graph, treating it as a constant.
F.mse_loss(input, self.target): Computes the mean squared error between the input and the target.
return input: Returns the input tensor unchanged.


