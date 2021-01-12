# ***************************************
# In this script we run over all pretrained imageNet models in torchVision and compare their accuracy (top-1 error and top-5 error),
# inference times, model size and finally a multidimensional comparison of all these metrics
# 
# Data visualization software: pyRoot
# ***************************************

from torchvision import models
import torch
#model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
dir(models)

#models to choose from:
# ['AlexNet', 'DenseNet', 'GoogLeNet', 'GoogLeNetOutputs', 'Inception3', 'InceptionOutputs', 'MNASNet', 'MobileNetV2', 'ResNet', 'ShuffleNetV2', 'SqueezeNet', 'VGG', '_GoogLeNetOutputs', '_InceptionOutputs', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', '_utils', 'alexnet', 'densenet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'detection', 'googlenet', 'inception', 'inception_v3', 'mnasnet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet', 'mobilenet_v2', 'quantization', 'resnet', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext50_32x4d', 'segmentation', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'shufflenetv2', 'squeezenet', 'squeezenet1_0', 'squeezenet1_1', 'utils', 'vgg', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'video', 'wide_resnet101_2', 'wide_resnet50_2']


# download the model weights
#model = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
#model = torch.hub.load('pytorch/vision', 'resnet101', pretrained=True)
model = torch.hub.load('pytorch/vision', 'vgg16', pretrained=True)
#model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
model.eval()

print("model details: ", model)

# Download an example image from the pytorch website
#import urllib
#url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
#try: urllib.URLopener().retrieve(url, filename)
#except: urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)
from PIL import Image
img = Image.open("images/Aspen4.jpg")

from torchvision import transforms
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
        
with torch.no_grad():
    # inference step
    output = model(input_batch)
    print("inference: ", output.shape)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
#print(output[0])

# read and store labels 
with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
# find index where max score is in output. Use this index to find the prediction
_, index = torch.max(output, 1)
percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
#print(labels[index[0]], percentage[index[0]].item())

# find other classes the model predicted
_, indices = torch.sort(output, descending=True)
#[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
for idx in indices[0][:5]:
    print("class: ", labels[idx], " likelihood: ", percentage[idx].item())

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
#print(torch.max(torch.nn.functional.softmax(output[0], dim=0)))
                    
